"""
train_fm_distill.py — 渐进蒸馏：C1 HRRP 条件 FM 的 1 步学生（fresh-data 协议）

背景：C1 的 HRRP 条件 FM 在 NFE=1 已达 CD 0.2269，但 NFE=1 输出是
x0 + v(x0, 0, c) 的单步 Euler——积分误差存在。渐进蒸馏让学生直接回归
teacher（midpoint NFE=2）的输出：同 1 次前向成本，去掉积分误差。

协议（遵守 §7.25 的数据冻结教训）：
  - 逐 epoch 新样本（SatROIDataset 逐样本生成，不物化）
  - teacher 输出无梯度；学生参数独立
  - 与学生同构：z_student = x0 + v_s(x0, 0, c)，loss = MSE(z_student, z_teacher)

预注册命题：
  P1 学生 1 步 CD ≤ teacher 1 步 CD − 3%（积分误差被消除）
  P2 学生 CD ≤ teacher 2 步 CD（学生逼近 teacher 的高精度输出）
  P3 训练损失下降（学习发生）
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import train_PointVAE, estimate_latent_stats, chamfer_distance_loss
from fm_utils import CFMScheduler, T_EMB_SCALE, sample_conditional_FM
from compare_gen import _PairView


def _integrate(condenc, vnet, x0, cond, nfe, solver, cfg_scale, device):
    """teacher 的 ODE 积分（无梯度）。"""
    B = cond.size(0)
    c = condenc(cond)
    c_null = condenc(torch.zeros_like(cond))

    def v_cfg(x, t01):
        t = torch.full((B,), t01 * T_EMB_SCALE, device=device)
        return vnet(x, t, c_null) + cfg_scale * (vnet(x, t, c) - vnet(x, t, c_null))

    x = x0
    dt = 1.0 / nfe
    for i in range(nfe):
        t0 = i / nfe
        if solver == "midpoint" and nfe > 1:
            k1 = v_cfg(x, t0)
            x = x + dt * v_cfg(x + 0.5 * dt * k1, t0 + 0.5 * dt)
        else:
            x = x + dt * v_cfg(x, t0)
    return x


def main(args):
    device = args.device
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

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
    for p in list(condenc.parameters()) + list(vnet.parameters()):
        p.requires_grad_(False)
    vae.eval(); condenc.eval(); vnet.eval()
    print(f"teacher 加载自 {src} (cond_dim={cond_dim})")

    # 学生：独立参数的 vnet（同构），condenc 与 teacher 共享（冻结）
    student = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                    depth=args.depth, num_heads=8).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    # fresh-data 训练（不物化）
    train_ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                             device=device, tau=args.tau, phase_mode=args.phase_mode,
                             cond_feat=args.cond_feat)
    loader = DataLoader(_PairView(train_ds), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    gen = torch.Generator(device=device).manual_seed(args.x0_seed)

    hist = []
    for ep in range(1, args.epochs + 1):
        student.train()
        tot = 0.0
        for pc, cond in loader:
            cond = cond.to(device)
            B = cond.size(0)
            x0 = torch.randn((B, 256), device=device, generator=gen)
            with torch.no_grad():
                z_teacher = _integrate(condenc, vnet, x0, cond, args.teacher_nfe,
                                       "midpoint", 2.0, device)
            c = condenc(cond)
            c_null = condenc(torch.zeros_like(cond))
            t = torch.zeros(B, device=device)
            v_s = student(x0, t, c_null) + 2.0 * (student(x0, t, c) - student(x0, t, c_null))
            loss = F.mse_loss(x0 + v_s, z_teacher)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        hist.append(tot / len(loader))
        if ep % 10 == 0 or ep == 1:
            print(f"[distill] epoch {ep}/{args.epochs} loss={hist[-1]:.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.save_dir, "student_1step.pth"))

    # ---- 评估：同一测试批，teacher vs student ----
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            cond_feat=args.cond_feat)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)

    student.eval()
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(999)
        x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
        c = condenc(cond); c_null = condenc(torch.zeros_like(cond))
        t = torch.zeros(cond.size(0), device=device)
        v_s = student(x0, t, c_null) + 2.0 * (student(x0, t, c) - student(x0, t, c_null))
        pc_stu, _ = vae.decode((x0 + v_s) * z_std + z_mean)
        cd_stu = chamfer_distance_loss(pc_gt, pc_stu).item()

        pc_t1 = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                      device=device, cfg_scale=2.0, nfe=1)
        cd_t1 = chamfer_distance_loss(pc_gt, pc_t1).item()
        pc_t2 = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                      device=device, cfg_scale=2.0, nfe=2, solver="midpoint")
        cd_t2 = chamfer_distance_loss(pc_gt, pc_t2).item()

    verdicts = {
        "P1_student_beats_teacher1": bool(cd_stu <= 0.97 * cd_t1),
        "P2_student_beats_teacher2": bool(cd_stu <= cd_t2),
        "P3_loss_decreased": bool(hist[-1] < hist[0]),
    }
    print(f"\n{'=' * 66}")
    print(f"teacher NFE=1 : CD={cd_t1:.4f}")
    print(f"teacher NFE=2 : CD={cd_t2:.4f}")
    print(f"student 1-step: CD={cd_stu:.4f}  (vs teacher1 {(cd_stu/cd_t1-1)*100:+.1f}%)")
    print(f"裁决: {verdicts}")
    out = {"cd_teacher_nfe1": cd_t1, "cd_teacher_nfe2": cd_t2, "cd_student": cd_stu,
           "epochs": args.epochs, "loss_first": hist[0], "loss_last": hist[-1],
           "verdicts": verdicts}
    with open(os.path.join(args.save_dir, "distill_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {os.path.join(args.save_dir, 'distill_result.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C1 FM 渐进蒸馏")
    parser.add_argument("--ckpt_dir", type=str, default="./sat_model_c1",
                        help="teacher checkpoint 目录（含 sat/ 子目录）")
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                        default="hrrp")
    parser.add_argument("--train_data", type=int, default=1024)
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher_nfe", type=int, default=2)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./sat_model_distill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x0_seed", type=int, default=123)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
