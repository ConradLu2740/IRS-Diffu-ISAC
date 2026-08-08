"""
train_sat.py — 星-地 ISAC 对比训练（none / sat / ground 三种 IRS 模式）

复用现有架构：
  - models.py: PointVAE / AdvancedCondEncoder / LatentDiT1D_CrossAttn
  - train.py:  DDPMScheduler / train_PointVAE / train_1D_DDPM /
               estimate_latent_stats / sample_conditional_1D / chamfer_distance_loss
  - data_sat.py: SatROIDataset（动态星-地样本）

用法（smoke 示例）：
  python train_sat.py --irs_mode sat --train_data 32 --test_data 8 --vae_epochs 1 --ldm_epochs 1
"""

import os
import math
import argparse
import numpy as np
import random
import torch
import sys
from torch.utils.data import DataLoader

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels, P_SNR as SAT_P_SNR
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import (
    DDPMScheduler,
    train_PointVAE,
    train_1D_DDPM,
    estimate_latent_stats,
    sample_conditional_1D,
    chamfer_distance_loss,
)


def run_mode(args, irs_mode):
    """训练并评估一种 IRS 模式，返回平均 CD。"""
    device = args.device
    print(f"\n{'=' * 70}")
    print(f"[{irs_mode}] 构建星-地 ISAC 场景与数据 (SNR≈{SAT_P_SNR}dB)")
    print(f"{'=' * 70}")

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=irs_mode, device=device)

    train_ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                             device=device, tau=args.tau, phase_mode=args.phase_mode)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cond_dim = channels.frame_cond_dim()
    print(f"[{irs_mode}] cond_dim={cond_dim}, train={len(train_ds)}, test={len(test_ds)}")

    save_dir = os.path.join(args.save_dir, irs_mode)
    os.makedirs(save_dir, exist_ok=True)

    # ---- 模型 ----
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    epsnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                   depth=args.depth, num_heads=8).to(device)
    sched = DDPMScheduler(T=args.T, device=device)
    print(f"[{irs_mode}] VAE={sum(p.numel() for p in vae.parameters())/1e6:.2f}M "
          f"CondEnc={sum(p.numel() for p in condenc.parameters())/1e6:.2f}M "
          f"DiT={sum(p.numel() for p in epsnet.parameters())/1e6:.2f}M")

    # ---- 阶段 1：PointVAE ----
    print(f"[{irs_mode}] Stage 1: PointVAE (epochs={args.vae_epochs}, kl_weight={args.kl_weight}, warmup={args.kl_warmup})")
    train_PointVAE(vae, train_loader, test_loader, device=device,
                   epochs=args.vae_epochs, lr=1e-3, kl_weight=args.kl_weight,
                   kl_warmup_epochs=args.kl_warmup, save_dir=save_dir)

    # ---- 潜在统计 + 阶段 2：LDM ----
    print(f"[{irs_mode}] Stage 2: Latent Diffusion (epochs={args.ldm_epochs}, T={args.T})")
    z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device)
    torch.save({"z_mean": z_mean, "z_std": z_std}, os.path.join(save_dir, "latent_stats.pth"))
    print(f"[{irs_mode}] latent_stats 已保存")
    train_1D_DDPM(vae, condenc, epsnet, sched, train_loader, test_loader,
                  z_mean, z_std, device=device, epochs=args.ldm_epochs,
                  save_dir=save_dir)

    # ---- 条件采样 + CD 评估 ----
    print(f"[{irs_mode}] Conditional sampling ({args.n_eval} samples, CFG=2.0)")
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)
    pc_hat = sample_conditional_1D(vae, condenc, epsnet, sched, cond,
                                   z_mean, z_std, device=device, cfg_scale=2.0)

    with torch.no_grad():
        cd = chamfer_distance_loss(pc_gt, pc_hat)
    print(f"[{irs_mode}] eval CD (GT vs sampled) = {cd.item():.6f}")
    return cd.item()


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    for mode in args.modes:
        results[mode] = run_mode(args, mode)

    print(f"\n{'=' * 70}")
    print("对比结果汇总 (CD 越小越好):")
    for mode, cd in results.items():
        print(f"  {mode:>7}: CD = {cd:.6f}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星-地 ISAC 对比训练")
    parser.add_argument("--modes", nargs="+", choices=["none", "sat", "ground"],
                        default=["none", "sat", "ground"])
    parser.add_argument("--train_data", type=int, default=64)
    parser.add_argument("--test_data", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--vae_epochs", type=int, default=1)
    parser.add_argument("--ldm_epochs", type=int, default=1)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="VAE KL 权重")
    parser.add_argument("--kl_warmup", type=int, default=5, help="VAE KL 退火 epoch 数（0=关闭）")
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./sat_model")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random",
                        help="IRS 相位模式：random（随机）或 tracked（解析跟踪优化）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, modes={args.modes}")
    main(args)
