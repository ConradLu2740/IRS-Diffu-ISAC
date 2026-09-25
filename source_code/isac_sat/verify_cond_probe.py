"""
verify_cond_probe.py — 条件信息充分性门禁（G15 的前置决策实验）

问题：G15 修复坍塌后 Δ(t)≈0。两种可能：
  (a) cond 里**有**信息，但训练/优化没利用（⇒ 值得做 drop=0.5 重训/辅助损失）
  (b) cond 里**没有**（足够）信息（⇒ 重训无用，应换条件输入，如 HRRP）

判决实验：训练探针 MLP  cond → x1（VAE 潜变量），测留出集 R²。
对照：HRRP → x1 探针（宽带有无更多信息）。
  R²(cond) ≥ 0.30 ⇒ 信息存在，问题在优化（走 (a)）
  R²(cond) < 0.10 ⇒ 信息不足，走 (b)（改条件输入）
  0.10–0.30 ⇒ 边缘，报告两种解读

协议：SatROIDataset 的 (pc, cond) 对；x1 = VAE 编码（后验均值，逐维白化口径
与训练一致）；1024 训练 / 128 留出；探针为 2 层 MLP；HRRP 探针用
compute_range_profile 的 512 维特征。
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

from torch.utils.data import DataLoader
import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels, compute_range_profile
from models import PointVAE


class ProbeMLP(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def collect(args, vae, z_mean, z_std, device):
    """收集严格配对的 (cond, x1, hrrp)：同一 ROI 生成三者。"""
    from verify_fm_shape_loop import get_roi_and_cond
    from data_sat import extract_point_cloud_from_voxel
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    sc = ss.SatISACScenario(tau=args.tau)
    frames = sc.build_frames()
    ch = SatScenarioChannels(frames, irs_mode="sat", device=device)
    ds = SatROIDataset(args.n_total, ch, num_points=args.num_points, device=device,
                       tau=args.tau, phase_mode=args.phase_mode)
    mid = frames[len(frames) // 2]
    conds, hrps, x1s = [], [], []
    with torch.no_grad():
        for i in range(args.n_total):
            roi_np, cond = get_roi_and_cond(ds, i)
            pc = extract_point_cloud_from_voxel(roi_np, num_points=args.num_points,
                                                voxel_size=ds.ch.voxel_size)
            max_extent = ds.ch.roi_res * ds.ch.voxel_size
            pc = (pc / max_extent) * 2.0 - 1.0
            pc_t = torch.tensor(pc, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = vae.encode(pc_t)
            x1 = ((mu - z_mean) / z_std).squeeze(0)
            rp = compute_range_profile(roi_np, mid["target_pos"], mid["ground_pos"],
                                       ch.wavelength_m, snr_db=args.snr_db, seed=0,
                                       align=True, sat_ecef=mid["sat_pos"])
            conds.append(cond); x1s.append(x1)
            hrps.append(torch.from_numpy(rp).float())
    return torch.stack(conds), torch.stack(x1s), torch.stack(hrps), ch, ds


def ridge_r2(X, Y, lam=1.0):
    """闭式岭回归探针（无过拟合容量问题，公平比较不同特征的信息量）。"""
    X, Y = X.double().cpu(), Y.double().cpu()
    n_tr = int(0.8 * X.shape[0])
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xte, Yte = X[n_tr:], Y[n_tr:]
    keep = Xtr.std(0) > 1e-6
    X, Xtr, Xte = X[:, keep], Xtr[:, keep], Xte[:, keep]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(n_tr, 1)], dim=1)
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.pinv(A) @ Xtr.T @ Ytr
    pred = Xte @ W
    ss_res = F.mse_loss(pred, Yte).item()
    ss_tot = Yte.var(unbiased=False).item()
    return float(1.0 - ss_res / ss_tot)


def train_probe(X, Y, args, device):
    n_tr = int(0.8 * X.shape[0])
    Xtr, Ytr = X[:n_tr].to(device), Y[:n_tr].to(device)
    Xte, Yte = X[n_tr:].to(device), Y[n_tr:].to(device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    probe = ProbeMLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        probe.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, 64):
            idx = perm[i:i + 64]
            loss = F.mse_loss(probe(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xte)
        ss_res = F.mse_loss(pred, Yte).item()
        ss_tot = Yte.var(unbiased=False).item()
    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # VAE（缩放版，逐维白化口径）
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(args.vae_dir, "vae_best.pth"),
                                   map_location=device))
    stats = torch.load(os.path.join(args.vae_dir, "latent_stats.pth"), map_location=device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]
    vae.eval()
    print(f"VAE: {args.vae_dir}, z_mean shape {tuple(z_mean.shape)}")

    conds, x1s, hrps, ch, ds = collect(args, vae, z_mean, z_std, device)
    n = conds.shape[0]
    print(f"配对样本: {n}, cond dim {conds.shape[1] * conds.shape[2]}")

    X_cond = conds.reshape(n, -1)
    r2_cond = train_probe(X_cond, x1s, args, device)
    print(f"\nR²(cond → x1) = {r2_cond:.4f}")

    r2_hrp = train_probe(hrps, x1s, args, device)
    print(f"R²(HRRP → x1) = {r2_hrp:.4f}  (严格配对, n={n})")

    # 岭探针（无过拟合容量，公平比较）
    r2_cond_ridge = ridge_r2(X_cond, x1s)
    r2_hrp_ridge = ridge_r2(hrps, x1s)
    print(f"\n岭探针: R2(cond) = {r2_cond_ridge:.4f}  R2(HRRP) = {r2_hrp_ridge:.4f}"
          f"  (参照: 类别标签单独解释 0.2449)")

    # 容量无关的参照：类别标签单独解释 24.5% 潜方差（解析计算，无探针容量问题）
    r2_class_ref = 0.2449
    if r2_cond >= 0.30:
        verdict = "信息存在 ⇒ 走 (a) drop=0.5 重训/辅助损失"
    elif r2_cond < r2_class_ref:
        verdict = (f"cond 信息量({r2_cond:.3f})低于类别标签单独解释率({r2_class_ref:.3f})"
                   f" ⇒ 条件输入信息贫乏，训练技巧无用 ⇒ 走 (b) 换条件输入（HRRP）")
    else:
        verdict = "边缘 ⇒ 记录在案，优先 (b)"
    print(f"\n裁决: {verdict}")

    out = {"r2_cond_mlp": r2_cond, "r2_hrp_mlp": r2_hrp,
           "r2_cond_ridge": r2_cond_ridge, "r2_hrp_ridge": r2_hrp_ridge,
           "r2_class_label_ref": r2_class_ref, "n": n,
           "verdict": verdict, "vae_dir": args.vae_dir}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "cond_probe.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="条件信息充分性门禁")
    parser.add_argument("--vae_dir", type=str, default="./sat_model_scale_42/sat")
    parser.add_argument("--n_total", type=int, default=512)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
