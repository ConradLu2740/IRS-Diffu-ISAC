"""
compare_gen.py — 扩散 vs Flow Matching 等算力公平对比（星-地 ISAC 3D 重建）

严格公平协议（对齐仓库 physics-grounded 调性）：
  - 每个 IRS 模式：场景 / 数据集 / 随机种子 / PointVAE 只构建并训练一次
  - DDPM 与 FM 在**同一批归一化潜变量**上训练（同 epoch、同优化器超参、同 CFG 约定）
  - 同一测试批评估 CD / F-Score / Voxel IoU；采样质量-NFE 曲线：
      DDPM 取原生 T 步（NFE=T），FM 扫描 nfe_list（ODE 步数可任意缩小）
  - 结果落盘 JSON + 曲线图，可直接写进 README / TECH_REPORT

用法：
  python compare_gen.py --modes sat --train_data 64 --test_data 16 \
      --vae_epochs 10 --gen_epochs 10
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import sys
from torch.utils.data import DataLoader

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from fm_utils import train_1D_FM, sample_conditional_FM
from eval_sat import f_score, voxel_iou

class _PairView(torch.utils.data.Dataset):
    """把 (pc, cond, *rest) 数据集包装成 (pc, cond) 对（宽带特征丢弃）。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        out = self.base[i]
        return out[0], out[1]




def new_gen_models(cond_dim, args, device):
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    net = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                depth=args.depth, num_heads=8).to(device)
    return condenc, net


def eval_metrics(pc_gt, pc_hat, roi_res):
    cd = chamfer_distance_loss(pc_gt, pc_hat).item()
    fs1 = float(np.mean([f_score(pc_gt[i], pc_hat[i], 0.1) for i in range(pc_gt.shape[0])]))
    fs2 = float(np.mean([f_score(pc_gt[i], pc_hat[i], 0.2) for i in range(pc_gt.shape[0])]))
    iou = float(np.mean([voxel_iou(pc_gt[i], pc_hat[i], roi_res) for i in range(pc_gt.shape[0])]))
    return {"cd": cd, "fs_0.1": fs1, "fs_0.2": fs2, "iou": iou}


def run_mode(args, irs_mode):
    device = args.device
    print(f"\n{'=' * 74}")
    print(f"[{irs_mode}] 等算力对比：DDPM vs Flow Matching (SNR≈{SAT_P_SNR}dB)")
    print(f"{'=' * 74}")

    # ---- 场景与数据（两者共享，只构建一次）----
    tle = ss.STARLINK_TLE if args.sat == "starlink" else ss.ISS_TLE
    scenario = ss.SatISACScenario(tau=args.tau, tle_lines=tle, sat_name=args.sat)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=irs_mode, device=device)
    train_ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                             device=device, tau=args.tau, phase_mode=args.phase_mode,
                             cond_feat=args.cond_feat)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            cond_feat=args.cond_feat)
    train_loader = DataLoader(_PairView(train_ds), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    cond_dim = train_ds[0][1].shape[-1]   # 由实际样本推断（hrrp 模式为 512）

    save_dir = os.path.join(args.save_dir, irs_mode)
    os.makedirs(save_dir, exist_ok=True)

    # ---- 共享 PointVAE（--vae_ckpt 提供则复用，跳过训练）----
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    if args.vae_ckpt:
        sd_src = os.path.join(args.vae_ckpt, irs_mode)
        vae.load_state_dict(torch.load(os.path.join(sd_src, "vae_best.pth"),
                                       map_location=device))
        stats = torch.load(os.path.join(sd_src, "latent_stats.pth"), map_location=device)
        z_mean, z_std = stats["z_mean"], stats["z_std"]
        print(f"[{irs_mode}] Stage 1 (shared): PointVAE 复用自 {sd_src}（跳过训练）")
    else:
        print(f"[{irs_mode}] Stage 1 (shared): PointVAE (epochs={args.vae_epochs})")
        train_PointVAE(vae, train_loader, test_loader, device=device,
                       epochs=args.vae_epochs, lr=1e-3, kl_weight=args.kl_weight,
                       kl_warmup_epochs=args.kl_warmup, save_dir=save_dir)
        z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device,
                                              per_dim=args.whiten == "perdim")
    torch.save({"z_mean": z_mean, "z_std": z_std}, os.path.join(save_dir, "latent_stats.pth"))

    # ---- 两种生成模型：同潜空间、同 epoch、同优化器超参 ----
    print(f"[{irs_mode}] Stage 2a: DDPM (epochs={args.gen_epochs}, T={args.T})")
    condenc_d, epsnet = new_gen_models(cond_dim, args, device)
    sched = DDPMScheduler(T=args.T, device=device)
    train_1D_DDPM(vae, condenc_d, epsnet, sched, train_loader, test_loader,
                  z_mean, z_std, device=device, epochs=args.gen_epochs,
                  lr_cond=args.lr_cond, posterior_sample=args.posterior_sample,
                  save_dir=save_dir)

    print(f"[{irs_mode}] Stage 2b: Flow Matching (epochs={args.gen_epochs})")
    condenc_f, vnet = new_gen_models(cond_dim, args, device)
    train_1D_FM(vae, condenc_f, vnet, train_loader, test_loader,
                z_mean, z_std, device=device, epochs=args.gen_epochs,
                lr_cond=args.lr_cond, posterior_sample=args.posterior_sample,
                save_dir=save_dir)

    # ---- 同一测试批评估 ----
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)

    results = {"ddpm": {}, "fm": {}}

    pc_hat_d = sample_conditional_1D(vae, condenc_d, epsnet, sched, cond,
                                     z_mean, z_std, device=device, cfg_scale=2.0)
    with torch.no_grad():
        results["ddpm"][args.T] = eval_metrics(pc_gt, pc_hat_d, args.roi_res)
    print(f"[{irs_mode}] DDPM  NFE={args.T}: CD={results['ddpm'][args.T]['cd']:.6f}")

    for nfe in args.nfe_list:
        pc_hat_f = sample_conditional_FM(vae, condenc_f, vnet, cond,
                                         z_mean, z_std, device=device,
                                         cfg_scale=2.0, nfe=nfe, solver=args.solver)
        with torch.no_grad():
            results["fm"][nfe] = eval_metrics(pc_gt, pc_hat_f, args.roi_res)
        print(f"[{irs_mode}] FM    NFE={nfe}: CD={results['fm'][nfe]['cd']:.6f}")

    # ---- VAE 上界参考（重建天花板）----
    with torch.no_grad():
        mu, _ = vae.encode(pc_gt)
        pc_vae, _ = vae.decode(mu)
        results["vae_oracle"] = eval_metrics(pc_gt, pc_vae, args.roi_res)
    print(f"[{irs_mode}] VAE oracle CD={results['vae_oracle']['cd']:.6f}")

    return results


def plot_nfe_curve(all_results, save_dir):
    """质量-NFE 曲线（FM 连续曲线 + DDPM 单点）。"""
    for mode, res in all_results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        nfes = sorted(int(k) for k in res["fm"].keys())
        cds = [res["fm"][str(nfe) if str(nfe) in res["fm"] else nfe]["cd"] for nfe in nfes]
        ax.plot(nfes, cds, "o-", label="Flow Matching (ODE)", color="crimson")
        for nfe_ddpm, m in res["ddpm"].items():
            ax.scatter([int(nfe_ddpm)], [m["cd"]], marker="*", s=180,
                       color="steelblue", label=f"DDPM (NFE={nfe_ddpm})", zorder=5)
        ax.axhline(res["vae_oracle"]["cd"], ls="--", c="gray", alpha=0.6,
                   label="VAE reconstruction ceiling")
        ax.set_xscale("log")
        ax.set_xlabel("NFE (network evaluations)")
        ax.set_ylabel("Chamfer Distance ↓")
        ax.set_title(f"Sampling efficiency — IRS mode = {mode}")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        path = os.path.join(save_dir, f"nfe_curve_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[{mode}] NFE 曲线已保存: {path}")


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}
    for mode in args.modes:
        all_results[mode] = run_mode(args, mode)

    # 汇总表
    print(f"\n{'=' * 74}")
    print("等算力对比汇总 (CD 越小越好；DDPM NFE=T，FM NFE 可配):")
    for mode, res in all_results.items():
        for nfe, m in res["ddpm"].items():
            print(f"  {mode:>7} DDPM NFE={nfe:>4}: CD={m['cd']:.4f} "
                  f"FS@0.1={m['fs_0.1']:.4f} IoU={m['iou']:.4f}")
        for nfe in sorted(int(k) for k in res["fm"].keys()):
            m = res["fm"][nfe]
            print(f"  {mode:>7} FM   NFE={nfe:>4}: CD={m['cd']:.4f} "
                  f"FS@0.1={m['fs_0.1']:.4f} IoU={m['iou']:.4f}")
        print(f"  {mode:>7} VAE oracle   : CD={res['vae_oracle']['cd']:.4f}")
    print(f"{'=' * 74}")

    # JSON 落盘（int key → str）
    out = {}
    for mode, res in all_results.items():
        out[mode] = {
            "ddpm": {str(k): v for k, v in res["ddpm"].items()},
            "fm": {str(k): v for k, v in res["fm"].items()},
            "vae_oracle": res["vae_oracle"],
            "config": {
                "train_data": args.train_data, "test_data": args.test_data,
                "vae_epochs": args.vae_epochs, "gen_epochs": args.gen_epochs,
                "T": args.T, "seed": args.seed, "phase_mode": args.phase_mode,
                "torch": torch.__version__,
            },
        }
    json_path = os.path.join(args.save_dir, "compare_gen.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {json_path}")

    plot_nfe_curve({m: {**r, "fm": {str(k): v for k, v in r["fm"].items()}}
                    for m, r in all_results.items()}, args.save_dir)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM vs Flow Matching 等算力对比")
    parser.add_argument("--modes", nargs="+", choices=["none", "sat", "ground"],
                        default=["none", "sat", "ground"])
    parser.add_argument("--train_data", type=int, default=64)
    parser.add_argument("--test_data", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--vae_epochs", type=int, default=10)
    parser.add_argument("--gen_epochs", type=int, default=10)
    parser.add_argument("--lr_cond", type=float, default=1e-4,
                        help="条件编码器学习率（G15: 1e-3 会导致条件坍塌，默认 1e-4）")
    parser.add_argument("--posterior_sample", type=int, default=1,
                        help="1=传输目标用后验样本 z~q（ELBO 一致性，默认）；0=后验均值 μ（旧行为）")
    parser.add_argument("--whiten", choices=["scalar", "perdim"], default="perdim",
                        help="潜空间归一化：perdim=逐维白化（默认，恢复逐维高斯恒等式）；scalar=旧标量")
    parser.add_argument("--vae_ckpt", type=str, default=None,
                        help="提供 VAE checkpoint 目录则复用（跳过 VAE 训练，隔离生成侧变量）")
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp"], default="narrowband",
                        help="条件输入：narrowband（默认）或 hrrp（C1：宽带距离像广播）")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--kl_warmup", type=int, default=5)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--roi_res", type=int, default=16)
    parser.add_argument("--nfe_list", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sat", choices=["iss", "starlink"], default="iss",
                        help="卫星（泛化：ISS vs Starlink TLE）")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, modes={args.modes}")
    main(args)
