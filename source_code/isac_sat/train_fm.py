"""
train_fm.py — 星-地 ISAC Flow Matching 训练（none / sat / ground 三种 IRS 模式）

与 train_sat.py（条件潜空间扩散）同构、同数据、同架构、同超参：
  - models.py:  PointVAE / AdvancedCondEncoder / LatentDiT1D_CrossAttn（共享）
  - train.py:   train_PointVAE / estimate_latent_stats / chamfer_distance_loss（共享）
  - fm_utils.py: CFMScheduler / train_1D_FM / sample_conditional_FM（FM 目标 + ODE 采样）
  - data_sat.py: SatROIDataset（动态星-地样本）

差异仅在生成范式：ε-预测扩散 → OT-CFM 速度场回归，采样用 ODE 积分（NFE 可配）。

用法（smoke 示例）：
  python train_fm.py --irs_mode sat --train_data 32 --test_data 8 --vae_epochs 1 --fm_epochs 1
"""

import os
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
    train_PointVAE,
    estimate_latent_stats,
    chamfer_distance_loss,
)
from fm_utils import train_1D_FM, sample_conditional_FM


def run_mode(args, irs_mode):
    """训练并评估一种 IRS 模式，返回 {nfe: CD} 字典。"""
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

    # ---- 模型（与 train_sat.py 完全相同）----
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    print(f"[{irs_mode}] VAE={sum(p.numel() for p in vae.parameters())/1e6:.2f}M "
          f"CondEnc={sum(p.numel() for p in condenc.parameters())/1e6:.2f}M "
          f"DiT={sum(p.numel() for p in vnet.parameters())/1e6:.2f}M")

    # ---- 阶段 1：PointVAE（与扩散版共享训练器）----
    print(f"[{irs_mode}] Stage 1: PointVAE (epochs={args.vae_epochs}, kl_weight={args.kl_weight}, warmup={args.kl_warmup})")
    train_PointVAE(vae, train_loader, test_loader, device=device,
                   epochs=args.vae_epochs, lr=1e-3, kl_weight=args.kl_weight,
                   kl_warmup_epochs=args.kl_warmup, save_dir=save_dir)

    # ---- 潜在统计 + 阶段 2：Flow Matching ----
    print(f"[{irs_mode}] Stage 2: Latent Flow Matching (epochs={args.fm_epochs})")
    z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device)
    torch.save({"z_mean": z_mean, "z_std": z_std}, os.path.join(save_dir, "latent_stats.pth"))
    print(f"[{irs_mode}] latent_stats 已保存")
    train_1D_FM(vae, condenc, vnet, train_loader, test_loader,
                z_mean, z_std, device=device, epochs=args.fm_epochs,
                lr_cond=args.lr_cond, save_dir=save_dir)

    # ---- 条件采样 + CD 评估（NFE 扫描）----
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)

    results = {}
    for nfe in args.nfe_list:
        pc_hat = sample_conditional_FM(vae, condenc, vnet, cond,
                                       z_mean, z_std, device=device,
                                       cfg_scale=2.0, nfe=nfe, solver=args.solver)
        with torch.no_grad():
            cd = chamfer_distance_loss(pc_gt, pc_hat)
        print(f"[{irs_mode}] eval CD (NFE={nfe}) = {cd.item():.6f}")
        results[nfe] = cd.item()
    return results


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    for mode in args.modes:
        results[mode] = run_mode(args, mode)

    print(f"\n{'=' * 70}")
    print("对比结果汇总 (CD 越小越好):")
    for mode, cd_by_nfe in results.items():
        row = "  ".join(f"NFE{nfe}={cd:.4f}" for nfe, cd in cd_by_nfe.items())
        print(f"  {mode:>7}: {row}")
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星-地 ISAC Flow Matching 训练")
    parser.add_argument("--modes", nargs="+", choices=["none", "sat", "ground"],
                        default=["none", "sat", "ground"])
    parser.add_argument("--train_data", type=int, default=64)
    parser.add_argument("--test_data", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--vae_epochs", type=int, default=1)
    parser.add_argument("--fm_epochs", type=int, default=1)
    parser.add_argument("--lr_cond", type=float, default=1e-4,
                        help="条件编码器学习率（G15: 1e-3 会导致条件坍塌，默认 1e-4）")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="VAE KL 权重")
    parser.add_argument("--kl_warmup", type=int, default=5, help="VAE KL 退火 epoch 数（0=关闭）")
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--nfe_list", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50],
                        help="ODE 采样步数（NFE）扫描列表")
    parser.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    parser.add_argument("--save_dir", type=str, default="./sat_model_fm")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random",
                        help="IRS 相位模式：random（随机）或 tracked（解析跟踪优化）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, modes={args.modes}")
    main(args)
