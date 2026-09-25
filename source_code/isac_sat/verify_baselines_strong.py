"""
verify_baselines_strong.py — 强 baseline 同口径对比（堵住"参数化偏差"攻击面）

评审缺口：FM 的少步优势可能被归因为"v-参数化 + ODE 采样器"而非目标函数本身。
本脚本在同一 VAE / 同一测试批 / 同一 CFG=2.0 下对比四类方法：

  1. DDPM-T      : 原生 T=100 步祖先采样（现状基线）
  2. DDIM-k      : 同一训练好的 DDPM，确定性 DDIM 重采样到 k 步（**零重训练**的
                   标准少步扩散基线；k∈{1,2,4,10}）
  3. FM-k        : OT-CFM + Euler/midpoint k 步（本项目方法）
  4. FM-distill  : 渐进蒸馏——teacher = FM midpoint NFE=2，student = 1 步
                   （x0 + v_θ(x0,0,c) 的系数网络）回归 teacher 输出；
                   两轮蒸馏到 NFE=1。**需要训练**（与 train_fm.py 同算力量级）。

公平协议：共享 vae_best/condenc/vnet/epsnet checkpoints；同一测试批；同一 CFG；
同一 CD 评估。训练侧 DDIM/FM 零成本，distill 与 FM 同 epoch 同数据。
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

from torch.utils.data import DataLoader

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import DDPMScheduler, chamfer_distance_loss
from fm_utils import T_EMB_SCALE, sample_conditional_FM
from verify_fm_bounds import build_eval_batch, load_models


# ----------------------------------------------------------------------
# DDIM：同一 DDPM 的确定性少步重采样（零重训练）
# ----------------------------------------------------------------------

@torch.no_grad()
def sample_ddim(vae, condenc, epsnet, sched, cond, z_mean, z_std, n_steps,
                device="cuda", cfg_scale=2.0):
    """DDIM（η=0）在训练好的 DDPM 上以 n_steps 步确定性采样。"""
    vae.eval(); condenc.eval(); epsnet.eval()
    B = cond.size(0)
    zt = torch.randn((B, 256), device=device)
    c = condenc(cond); c_null = condenc(torch.zeros_like(cond))

    ts = np.linspace(0, sched.T - 1, n_steps).astype(int)[::-1]   # T-1 → 0
    for i, ti in enumerate(ts):
        t = torch.full((B,), ti, device=device, dtype=torch.long)
        eps = epsnet(zt, t, c_null) + cfg_scale * (epsnet(zt, t, c) - epsnet(zt, t, c_null))
        abar_t = sched.abar[ti]
        z0_pred = (zt - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
        if i == len(ts) - 1:
            zt = z0_pred
        else:
            t_prev = ts[i + 1]
            abar_p = sched.abar[t_prev]
            zt = torch.sqrt(abar_p) * z0_pred + torch.sqrt(1 - abar_p) * eps
    z0 = zt * z_std + z_mean
    pc_hat, _ = vae.decode(z0)
    return pc_hat


# ----------------------------------------------------------------------
# 渐进蒸馏：teacher(FM midpoint NFE=2) → student(1 步)
# ----------------------------------------------------------------------

@torch.no_grad()
def teacher_target(vae, condenc, vnet, x0, cond, z_mean, z_std, nfe, device):
    """teacher 的潜变量输出（目标，无梯度）。"""
    from fm_utils import sample_conditional_FM
    z_hat, _ = _fm_integrate(condenc, vnet, x0, cond, nfe=nfe, solver="midpoint",
                             cfg_scale=2.0, device=device)
    return z_hat


def _fm_integrate(condenc, vnet, x0, cond, nfe, solver, cfg_scale, device):
    """FM ODE 积分（可训练版 student 用：student = x0 + v(x0, 0, c)，1 步）。"""
    B = cond.size(0)
    c = condenc(cond); c_null = condenc(torch.zeros_like(cond))

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
    return x, None


def train_distilled_student(vae, condenc, vnet, train_loader, z_mean, z_std,
                            device, epochs=60, lr=1e-4, teacher_nfe=2,
                            save_dir="./sat_model_cmp/distill"):
    """student 复用 vnet 结构但独立参数：z_student = x0 + v_s(x0, 0, c)，
    回归 teacher(FM midpoint NFE=2) 的潜变量输出。"""
    os.makedirs(save_dir, exist_ok=True)
    student = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                    depth=vnet.depth if hasattr(vnet, "depth") else 2,
                                    num_heads=8).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(123)

    for ep in range(1, epochs + 1):
        tot = 0.0
        student.train()
        condenc.eval()                                  # 条件编码器冻结（LSTM 勿入训练图）
        for pc, cond in train_loader:
            cond = cond.to(device)
            B = cond.size(0)
            x0 = torch.randn((B, 256), device=device, generator=g)
            with torch.no_grad():
                z_teacher, _ = _fm_integrate(condenc, vnet, x0, cond,
                                             nfe=teacher_nfe, solver="midpoint",
                                             cfg_scale=2.0, device=device)
                c = condenc(cond); c_null = condenc(torch.zeros_like(cond))
            t = torch.zeros(B, device=device)
            v = student(x0, t, c_null) + 2.0 * (student(x0, t, c) - student(x0, t, c_null))
            z_student = x0 + v
            loss = F.mse_loss(z_student, z_teacher)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"  [distill] epoch {ep}/{epochs} loss={tot / len(train_loader):.6f}")
    torch.save(student.state_dict(), os.path.join(save_dir, "student_1step.pth"))
    return student


@torch.no_grad()
def sample_student(vae, condenc, student, x0, cond, z_mean, z_std, device):
    c = condenc(cond); c_null = condenc(torch.zeros_like(cond))
    t = torch.zeros(cond.size(0), device=device)
    v = student(x0, t, c_null) + 2.0 * (student(x0, t, c) - student(x0, t, c_null))
    z0 = (x0 + v) * z_std + z_mean
    pc_hat, _ = vae.decode(z0)
    return pc_hat


def main(args):
    device = args.device
    pc_gt, cond, cond_dim = build_eval_batch(args, device)
    vae, condenc_d, epsnet, condenc_f, vnet, z_mean, z_std = load_models(args, device, cond_dim)
    sched = DDPMScheduler(T=args.T, device=device)

    # 蒸馏训练数据（与 compare_gen 同协议）
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    train_loader = DataLoader(SatROIDataset(args.train_data, channels,
                                            num_points=args.num_points, device=device,
                                            tau=args.tau, phase_mode=args.phase_mode),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)

    def cd_of(pc_hat):
        return chamfer_distance_loss(pc_gt, pc_hat).item()

    results = {}

    # 1) DDPM 原生 T 步
    results["DDPM_T"] = {"nfe": args.T,
                         "cd": cd_of(sample_ddim(vae, condenc_d, epsnet, sched, cond,
                                                 z_mean, z_std, args.T, device))}
    # 2) DDIM 少步（零重训练）
    for k in args.ddim_steps:
        results[f"DDIM_{k}"] = {"nfe": k,
                                "cd": cd_of(sample_ddim(vae, condenc_d, epsnet, sched, cond,
                                                        z_mean, z_std, k, device))}
    # 3) FM 少步
    for k in args.fm_steps:
        pc = sample_conditional_FM(vae, condenc_f, vnet, cond, z_mean, z_std,
                                   device=device, cfg_scale=2.0, nfe=k, solver="euler")
        results[f"FM_{k}"] = {"nfe": k, "cd": cd_of(pc)}
    # 4) 渐进蒸馏 1 步
    print("\n训练蒸馏 student（teacher=FM midpoint NFE=2）...")
    student = train_distilled_student(vae, condenc_f, vnet, train_loader, z_mean, z_std,
                                      device, epochs=args.distill_epochs,
                                      save_dir=os.path.join(args.save_dir, "distill"))
    g = torch.Generator(device=device).manual_seed(999)
    x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
    results["FM_distill_1"] = {"nfe": 1,
                               "cd": cd_of(sample_student(vae, condenc_f, student, x0, cond,
                                                          z_mean, z_std, device))}

    print(f"\n{'=' * 70}")
    print("强 baseline 同口径对比（CD ↓，同一测试批/CFG=2.0）:")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["nfe"]):
        print(f"  {name:>14} NFE={r['nfe']:>4}: CD={r['cd']:.4f}")
    print(f"{'=' * 70}")
    print("解读：若 FM_k 在所有 k 上 ≤ DDIM_k 且 ≤ distill_k，则少步优势来自目标函数")
    print("（直线 OT 路径）而非采样器技巧；若 distill 追平 FM，则优势部分可由蒸馏获得。")

    out = {"results": results, "protocol": {"T": args.T, "cfg_scale": 2.0,
                                            "distill_epochs": args.distill_epochs,
                                            "teacher_nfe": 2}}
    out_path = os.path.join(args.save_dir, "strong_baselines.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="强 baseline 同口径对比")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_data", type=int, default=256)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim_steps", nargs="+", type=int, default=[1, 2, 4, 10])
    parser.add_argument("--fm_steps", nargs="+", type=int, default=[1, 2, 4, 10])
    parser.add_argument("--distill_epochs", type=int, default=60)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
