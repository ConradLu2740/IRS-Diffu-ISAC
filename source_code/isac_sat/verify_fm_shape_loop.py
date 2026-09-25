"""
verify_fm_shape_loop.py — FM 生成形状替代手工盒子先验的闭环升级（G-κ 门的修复验证）

背景（G-κ 门实测）：闭环的真瓶颈不是相位优化（CA 已达认证上界 88%），而是
`demo.py:estimate_roi_from_pos` 的手工盒子 ROI——它比真实模板低 ~57% 的相位设计
功率，是 η_sense 损失的主要来源。

本脚本把 FM 生成模型从"只算 CD 的装饰"变成闭环真实组件：
  - 策略 box      : 现有手工盒子（基线）
  - 策略 fm_shape : FM NFE=1 采样形状 → 体素化 → 平移到 MLP 估计位置（新）
  - 策略 true     : 真实模板（oracle）
全部策略在同一场景/同一感知位置估计/同一相位设计器下对比。

预注册命题（roadmap §7，延续 G-κ 门的修复路径）：
  S1 盒子先验的 ℓ1 误差 ∈ [70, 150] 体素（G-κ 门记录的范围）
  S2 FM 形状先验的 ℓ1 误差 ≤ 盒子的 60%
  S3 η_sense(box) ≈ 0.87 复现；η_sense(fm_shape) ≥ 0.90（总达成率 → ≥0.71）
  S4 若 FM 形状分类与真实类别不一致（形状采样错误），功率不得劣于盒子+随机
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatScenarioChannels, SatROIDataset, generate_ground_target_sample, _SIGNAL1
from phase_optimizer_sat import PhaseOptimizerSat
from demo import estimate_roi_from_pos
from train_sensing import SensingMLP
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from fm_utils import sample_conditional_FM


def get_roi_and_cond(ds, idx=0):
    """复刻 SatROIDataset.__getitem__ 的 RNG 顺序，但保留 ROI 体素。

    返回 (ROI_np [16,16,16] float32, cond [tau, cond_dim] tensor)。
    """
    import math
    from data_sat import (calculate_value_sat, data_progress_amp_phase_db,
                          extract_point_cloud_from_voxel)
    ROI_np, class_id, angle, targets = ds._make_roi()
    ROI_voxel = torch.tensor(ROI_np).reshape(-1)
    X_cpu = ds.X_fixed.detach().cpu()
    X_amp_db = 20.0 * torch.log10(torch.abs(X_cpu).reshape(-1) + 1e-12)
    X_phase = torch.angle(X_cpu).reshape(-1)
    X_feat = torch.cat([X_amp_db, torch.sin(X_phase), torch.cos(X_phase)], dim=0).float()
    cond_list = []
    phases_seq = ds._frame_phases(ROI_voxel)
    for t in range(ds.tau):
        frame = ds.ch.channels_per_frame[t]
        t_rel = ds.ch.frames[t]["t_sec"]
        phases = phases_seq[t]
        Y_t = calculate_value_sat(ROI_voxel, phases, ds.X_fixed, frame,
                                  ds.power_sigma, t_rel, ds.ch.wavelength_m)
        Y_feat = data_progress_amp_phase_db(Y_t.detach().cpu())
        phases_cpu = phases.detach().cpu()
        if len(phases_cpu) > 0:
            IRS_feat = torch.cat([torch.sin(phases_cpu), torch.cos(phases_cpu)], dim=0).float()
        else:
            IRS_feat = torch.tensor([])
        dyn = torch.tensor([
            frame["f_d_bs_roi"] / 1e3, frame["delay_s"] * 1e3,
            frame["d_sat_target"] / 1e2, frame["elevation_deg"] / 90.0,
        ], dtype=torch.float32)
        y_pow = torch.log10(Y_t.detach().cpu().abs().pow(2).mean() + 1e-12)
        cond_t = torch.cat([X_feat, Y_feat, IRS_feat, dyn, y_pow.view(1)], dim=0).float()
        cond_list.append(cond_t)
    return ROI_np.astype(np.float32), torch.stack(cond_list).float()


def pc_to_voxels(pc, res=16, thresh=0.5):
    """点云 [N,3]（归一化 [-1,1]）→ 占据体素 [res³]（多数投票阈值）。"""
    idx = ((pc - (-1.0)) / 2.0 * res).astype(int).clip(0, res - 1)
    vox = np.zeros((res, res, res), dtype=np.float32)
    for i, j, k in idx:
        vox[i, j, k] += 1.0
    vox = (vox >= thresh * vox.max() if vox.max() > 0 else vox).astype(np.float32)
    return vox


def translate_voxels(vox, shift_xy):
    """把体素质心平移到归一化坐标 shift_xy（整数网格平移）。"""
    res = vox.shape[0]
    occ = np.argwhere(vox > 0.5)
    if len(occ) == 0:
        return vox
    cx, cy = (occ.mean(axis=0)[:2] / (res - 1)) * 2.0 - 1.0
    dx = int(round((shift_xy[0] - cx) / 2.0 * (res - 1)))
    dy = int(round((shift_xy[1] - cy) / 2.0 * (res - 1)))
    out = np.zeros_like(vox)
    xs = np.clip(occ[:, 0] + dx, 0, res - 1).astype(int)
    ys = np.clip(occ[:, 1] + dy, 0, res - 1).astype(int)
    zs = occ[:, 2].astype(int)
    out[xs, ys, zs] = 1.0
    return out


def load_fm(args, device, cond_dim):
    for cand in [os.path.join(args.save_dir, "sat"),
                 os.path.join("./sat_model_cmp", "sat")]:
        try:
            vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
            vae.load_state_dict(torch.load(os.path.join(cand, "vae_best.pth"), map_location=device))
            condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                          hidden_size=128, out_emb=256).to(device)
            condenc.load_state_dict(torch.load(os.path.join(cand, "condenc_fm_best.pth"), map_location=device))
            vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                         depth=args.depth, num_heads=8).to(device)
            vnet.load_state_dict(torch.load(os.path.join(cand, "vnet_fm_best.pth"), map_location=device))
            stats = torch.load(os.path.join(cand, "latent_stats.pth"), map_location=device)
            vae.eval(); condenc.eval(); vnet.eval()
            print(f"  FM checkpoint: {cand}")
            return vae, condenc, vnet, stats["z_mean"], stats["z_std"]
        except FileNotFoundError:
            continue
    raise FileNotFoundError("no FM checkpoint found")


def run_seed(seed, args, fm_models, ckpt):
    device = "cpu"
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]

    # 场景真实目标 + 其窄带 cond（供 FM 生成）
    ds = SatROIDataset(1, channels, num_points=args.num_points, device=device,
                       tau=args.tau, phase_mode=args.phase_mode)
    roi_true, cond = get_roi_and_cond(ds, 0)
    cond = cond.unsqueeze(0).to(device)

    # MLP 感知（宽带 HRRP → 位置），协议与 demo/分解一致
    from data_sat import compute_range_profile
    rp = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0,
                               align=True, sat_ecef=mid["sat_pos"])
    with torch.no_grad():
        logits, pred_pos = model(torch.from_numpy(rp).float().unsqueeze(0).to(device))
    pos_pred = pred_pos[0].cpu().numpy()
    pc_true = np.argwhere(roi_true > 0.5).astype(np.float32)
    pos_true = (pc_true.mean(axis=0) / 16.0 * 2.0 - 1.0)[:2]

    # FM 形状 → 体素 → 平移到感知位置
    vae, condenc, vnet, z_mean, z_std = fm_models
    with torch.no_grad():
        pc_hat = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                       device=device, cfg_scale=2.0, nfe=args.nfe)
    vox_fm = translate_voxels(pc_to_voxels(pc_hat[0].cpu().numpy()), pos_pred)
    vox_box = estimate_roi_from_pos(pos_pred)

    # 功率对比
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    def power_of(roi_est_np):
        """相位用估计 ROI 设计、功率在**真实 ROI** 上评估（与
        verify_optimality_decomposition.py 同口径：估计设计/真实评估）。"""
        roi_est_t = torch.tensor(roi_est_np.astype(np.float32)).reshape(-1)
        roi_true_t = torch.tensor(roi_true.astype(np.float32)).reshape(-1)
        ps = []
        for Ht in channels.channels_per_frame:
            ph = opt.optimize_frame(Ht, roi_est_t, X)
            ps.append(opt._power(Ht, roi_true_t, X, ph))
        return float(np.mean(ps))

    p_true = power_of(roi_true)
    p_box = power_of(vox_box)
    p_fm = power_of(vox_fm)

    l1_box = float(np.abs(vox_box - roi_true).sum())
    l1_fm = float(np.abs(vox_fm - roi_true).sum())
    return {"seed": seed, "pos_err": float(np.linalg.norm(pos_pred - pos_true)),
            "p_true": p_true, "p_box": p_box, "p_fm": p_fm,
            "l1_box": l1_box, "l1_fm": l1_fm,
            "eta_box": p_box / p_true, "eta_fm": p_fm / p_true}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"感知 checkpoint: cond_dim={ckpt['cond_dim']}, wideband={ckpt['wideband']}")

    # cond_dim 探测（建一次场景）
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    sc = ss.SatISACScenario(tau=args.tau); frames = sc.build_frames()
    ch0 = SatScenarioChannels(frames, irs_mode=args.irs_mode, device="cpu",
                              bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    fm_models = load_fm(args, "cpu", ch0.frame_cond_dim())

    rows = [run_seed(s, args, fm_models, ckpt) for s in range(args.n_seeds)]
    for r in rows:
        print(f"  seed {r['seed']}: pos_err={r['pos_err']:.3f} "
              f"eta_box={r['eta_box']:.3f} eta_fm={r['eta_fm']:.3f} "
              f"l1: box={r['l1_box']:.0f} fm={r['l1_fm']:.0f}")

    eta_box = np.mean([r["eta_box"] for r in rows])
    eta_fm = np.mean([r["eta_fm"] for r in rows])
    l1_box = np.mean([r["l1_box"] for r in rows])
    l1_fm = np.mean([r["l1_fm"] for r in rows])
    print(f"\n{'=' * 70}")
    print(f"FM 形状 vs 手工盒子（{args.n_seeds} seeds, irs={args.irs_mode}）:")
    print(f"  eta_sense: box={eta_box:.3f}  fm_shape={eta_fm:.3f}  "
          f"(+{(eta_fm / eta_box - 1) * 100:.1f}%)")
    print(f"  L1 体素误差: box={l1_box:.0f}  fm_shape={l1_fm:.0f}  "
          f"(fm/box = {l1_fm / max(l1_box, 1):.2f})")
    print(f"{'=' * 70}")
    print("预注册: S1 box L1 in [70,150]; S2 fm L1 <= 0.6*box; "
          f"S3 eta_fm >= 0.90 (box 复现 ~0.87); S4 实测 eta_box={eta_box:.3f}, eta_fm={eta_fm:.3f}")

    out = {"rows": rows,
           "summary": {"eta_box": float(eta_box), "eta_fm": float(eta_fm),
                       "l1_box": float(l1_box), "l1_fm": float(l1_fm),
                       "n_seeds": args.n_seeds, "nfe": args.nfe,
                       "fm_dir": args.save_dir},
           "verdicts": {
               "S1_box_l1_in_range": bool(70 <= l1_box <= 150),
               "S2_fm_l1_below_60pct": bool(l1_fm <= 0.6 * l1_box),
               "S3_eta_fm_ge_090": bool(eta_fm >= 0.90),
               "S3b_eta_box_reproduced": bool(0.80 <= eta_box <= 0.95)}}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "fm_shape_loop.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM 形状替代盒子先验的闭环验证")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--save_dir", type=str, default="./sat_model_g15",
                        help="FM checkpoint 目录（默认 G15 修复版）")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
