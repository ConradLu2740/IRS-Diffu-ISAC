"""
verify_fm_distill_diversity.py — D1：1 步蒸馏学生的样本多样性（模式坍塌检测）

背景：§7.30 的渐进蒸馏学生以 1 步前向在 CD 上胜过 teacher 23.4%，但少步/蒸馏
生成的潜在代价是多样性坍塌——只报 CD/F-Score/IoU 不足以排除模式坍塌。
本脚本复用 verify_gen_hardening 的多样性协议（同 cond × 16 个初始噪声，
CFG w=2.0）：pairwise CD（样本间）vs GT-vs-后验均值 CD，比率 ≈0 提示坍塌。

协议一致性：
  - 测试批按 --eval_seed 重播种后生成（与 train_fm_distill 的 eval_seed 相同），
    单样本 CD 复现蒸馏评估协议（x0 seed 999），不同训练预算的 student
    可在同一测试批上配对比较；
  - teacher 与学生消费同一批 x0 噪声（种子 1000+s），配对比较。

预注册命题（docs/optimization_roadmap.md §7.31）：
  D1 无不成比例坍塌：student 多样性比率 ≥ 0.7 × teacher 比率
     （比率 = pairwise CD / GT-vs-后验均值 CD；v1.11 的 teacher 侧证书约 2.7）
  D2 样本间确有差异：student pairwise CD ≥ 0.5 × teacher pairwise CD
  D3 同批质量一致性：student 对 GT 的 CD ≤ teacher 对 GT 的 CD

CFG 尺度扫描（--cfg_list，默认 0 1 2）附加命题（§7.32）：
  M1 CFG 驱动坍塌：teacher 比率(w=0) ≥ 3 × 比率(w=2)（去引导恢复多样性）
  M1b 引导买到质量：teacher CD(w=2) ≤ CD(w=0)
  M2 同 w=2 下学生 pairwise > teacher pairwise（多样性优势是结构性的）
  注：学生以 w=2.0 的 teacher 输出为训练目标，w≠2 属离分布评估。
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import chamfer_distance_loss
from fm_utils import sample_conditional_FM
from compare_gen import _PairView
from torch.utils.data import DataLoader


@torch.no_grad()
def one_step(vnet, condenc, x0, cond, device, cfg_scale=2.0):
    """NFE=1 单步 Euler + CFG（与学生/teacher 的 1 步输出同构）。"""
    B = cond.size(0)
    c = condenc(cond)
    c_null = condenc(torch.zeros_like(cond))
    t = torch.zeros(B, device=device)
    v = vnet(x0, t, c_null) + cfg_scale * (vnet(x0, t, c) - vnet(x0, t, c_null))
    return x0 + v


@torch.no_grad()
def diversity(vae, vnet, condenc, cond, pc_gt, z_mean, z_std,
              n_samples=16, device="cpu", cfg_scale=2.0):
    """同 cond × n_samples 个初始噪声：pairwise CD + GT-vs-后验均值 CD。"""
    outs = []
    for s in range(n_samples):
        g = torch.Generator(device=device).manual_seed(1000 + s)
        x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
        z0 = one_step(vnet, condenc, x0, cond, device, cfg_scale=cfg_scale) * z_std + z_mean
        pc, _ = vae.decode(z0)
        outs.append(pc)
    pcs = torch.stack(outs)                       # [S, B, N, 3]
    S, B = pcs.shape[:2]
    pw = []
    for b in range(B):
        for i in range(S):
            for j in range(i + 1, S):
                pw.append(chamfer_distance_loss(pcs[i, b:b+1], pcs[j, b:b+1]).item())
    gt_cd = [chamfer_distance_loss(pc_gt[b:b+1], pcs[:, b].mean(0, keepdim=True)).item()
             for b in range(B)]
    return {"pairwise_cd_mean": float(np.mean(pw)), "pairwise_cd_std": float(np.std(pw)),
            "gt_vs_postmean_cd": float(np.mean(gt_cd))}


def main(args):
    device = args.device
    random.seed(args.eval_seed); np.random.seed(args.eval_seed); torch.manual_seed(args.eval_seed)

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    src = os.path.join(args.ckpt_dir, "sat")

    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(src, "vae_best.pth"), map_location=device))
    stats = torch.load(os.path.join(src, "latent_stats.pth"), map_location=device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]
    ce_sd = torch.load(os.path.join(src, "condenc_fm_best.pth"), map_location=device)
    cond_dim = int(ce_sd["lstm.weight_ih_l0"].shape[1])

    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    condenc.load_state_dict(ce_sd)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    vnet.load_state_dict(torch.load(os.path.join(src, "vnet_fm_best.pth"), map_location=device))
    student = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                    depth=args.depth, num_heads=8).to(device)
    student.load_state_dict(torch.load(os.path.join(args.student_dir, "student_1step.pth"),
                                       map_location=device))
    for p in (list(vae.parameters()) + list(condenc.parameters())
              + list(vnet.parameters()) + list(student.parameters())):
        p.requires_grad_(False)
    vae.eval(); condenc.eval(); vnet.eval(); student.eval()

    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            cond_feat=args.cond_feat)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)

    # ---- 单样本 CD（复现 train_fm_distill 评估协议：teacher NFE=1/2，student x0 seed 999）----
    with torch.no_grad():
        pc_t1 = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                      device=device, cfg_scale=2.0, nfe=1)
        cd_t1 = chamfer_distance_loss(pc_gt, pc_t1).item()
        pc_t2 = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                      device=device, cfg_scale=2.0, nfe=2, solver="midpoint")
        cd_t2 = chamfer_distance_loss(pc_gt, pc_t2).item()
        g = torch.Generator(device=device).manual_seed(999)
        x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
        pc_stu, _ = vae.decode(one_step(student, condenc, x0, cond, device) * z_std + z_mean)
        cd_stu = chamfer_distance_loss(pc_gt, pc_stu).item()

    # ---- 多样性 headline（w=2.0，D1/D2/D3；复现蒸馏评估协议）----
    div_t = diversity(vae, vnet, condenc, cond, pc_gt, z_mean, z_std,
                      n_samples=args.n_samples, device=device)
    div_s = diversity(vae, student, condenc, cond, pc_gt, z_mean, z_std,
                      n_samples=args.n_samples, device=device)
    ratio_t = div_t["pairwise_cd_mean"] / max(div_t["gt_vs_postmean_cd"], 1e-9)
    ratio_s = div_s["pairwise_cd_mean"] / max(div_s["gt_vs_postmean_cd"], 1e-9)

    verdicts = {
        "D1_student_ratio_ge_0.7x_teacher": bool(ratio_s >= 0.7 * ratio_t),
        "D2_student_pairwise_ge_0.5x_teacher": bool(
            div_s["pairwise_cd_mean"] >= 0.5 * div_t["pairwise_cd_mean"]),
        "D3_student_cd_le_teacher_cd": bool(cd_stu <= cd_t1),
    }

    # ---- CFG 尺度扫描：坍塌是 CFG 假象还是条件本身？----
    # 配对协议：teacher 与学生消费同一 x0（seed 999）；每个 w 独立评估。
    # 学生以 w=2.0 的 teacher 输出为训练目标，w≠2 属离分布评估（如实记账）。
    sweep = {}
    for cfg in args.cfg_list:
        with torch.no_grad():
            pc_t, _ = vae.decode(one_step(vnet, condenc, x0, cond, device,
                                          cfg_scale=cfg) * z_std + z_mean)
            cd_t = chamfer_distance_loss(pc_gt, pc_t).item()
            pc_s, _ = vae.decode(one_step(student, condenc, x0, cond, device,
                                          cfg_scale=cfg) * z_std + z_mean)
            cd_s = chamfer_distance_loss(pc_gt, pc_s).item()
        d_t = diversity(vae, vnet, condenc, cond, pc_gt, z_mean, z_std,
                        n_samples=args.n_samples, device=device, cfg_scale=cfg)
        d_s = diversity(vae, student, condenc, cond, pc_gt, z_mean, z_std,
                        n_samples=args.n_samples, device=device, cfg_scale=cfg)
        r_t = d_t["pairwise_cd_mean"] / max(d_t["gt_vs_postmean_cd"], 1e-9)
        r_s = d_s["pairwise_cd_mean"] / max(d_s["gt_vs_postmean_cd"], 1e-9)
        sweep[f"w={cfg:g}"] = {"teacher": {"cd_gt": cd_t, **d_t, "ratio": float(r_t)},
                               "student": {"cd_gt": cd_s, **d_s, "ratio": float(r_s)}}

    if 0.0 in args.cfg_list and 2.0 in args.cfg_list:
        rt0, rt2 = sweep["w=0"]["teacher"]["ratio"], sweep["w=2"]["teacher"]["ratio"]
        cd0, cd2 = sweep["w=0"]["teacher"]["cd_gt"], sweep["w=2"]["teacher"]["cd_gt"]
        verdicts["M1_cfg_drives_collapse_ratio_w0_ge_3x_w2"] = bool(rt0 >= 3.0 * rt2)
        verdicts["M1b_guidance_buys_quality_cd_w2_le_w0"] = bool(cd2 <= cd0)
        verdicts["M2_student_more_diverse_at_matched_w2"] = bool(
            sweep["w=2"]["student"]["pairwise_cd_mean"]
            > sweep["w=2"]["teacher"]["pairwise_cd_mean"])

    print(f"\n{'=' * 70}")
    print("单样本 CD（同测试批，复现蒸馏评估协议）")
    print(f"  teacher NFE=1 : CD={cd_t1:.4f}")
    print(f"  teacher NFE=2 : CD={cd_t2:.4f}")
    print(f"  student 1-step: CD={cd_stu:.4f}  (vs teacher1 {(cd_stu/cd_t1-1)*100:+.1f}%)")
    print(f"\n样本多样性 headline（同 cond × {args.n_samples} 初始噪声，CFG w=2.0）")
    print(f"  teacher: pairwise CD {div_t['pairwise_cd_mean']:.4f} ± {div_t['pairwise_cd_std']:.4f}"
          f"  GT-vs-postmean {div_t['gt_vs_postmean_cd']:.4f}  比率 {ratio_t:.2f}")
    print(f"  student: pairwise CD {div_s['pairwise_cd_mean']:.4f} ± {div_s['pairwise_cd_std']:.4f}"
          f"  GT-vs-postmean {div_s['gt_vs_postmean_cd']:.4f}  比率 {ratio_s:.2f}")
    print(f"\nCFG 尺度扫描（同一 x0 配对；学生 w≠2 为离分布评估）")
    for k, v in sweep.items():
        print(f"  {k:>5}: teacher CD={v['teacher']['cd_gt']:.4f} ratio={v['teacher']['ratio']:.2f}"
              f" | student CD={v['student']['cd_gt']:.4f} ratio={v['student']['ratio']:.2f}")
    print(f"\n裁决: {verdicts}")

    out = {"cd_teacher_nfe1": cd_t1, "cd_teacher_nfe2": cd_t2, "cd_student": cd_stu,
           "diversity_teacher": {**div_t, "ratio": float(ratio_t)},
           "diversity_student": {**div_s, "ratio": float(ratio_s)},
           "cfg_sweep": sweep,
           "n_samples": args.n_samples, "n_eval": args.n_eval,
           "eval_seed": args.eval_seed, "student_dir": args.student_dir,
           "verdicts": verdicts}
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "distill_diversity_result.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D1：蒸馏学生样本多样性验证")
    parser.add_argument("--ckpt_dir", type=str, default="./sat_model_c1")
    parser.add_argument("--student_dir", type=str, default="./sat_model_distill")
    parser.add_argument("--save_dir", type=str, default="./sat_model_distill")
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                        default="hrrp")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=999)
    parser.add_argument("--cfg_list", type=float, nargs="+", default=[0.0, 1.0, 2.0],
                        help="CFG 尺度扫描列表（坍塌机制分析）")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
