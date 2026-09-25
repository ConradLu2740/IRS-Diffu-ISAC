"""
verify_fm_bounds.py — Flow Matching 优化有效性量化验证（数学性质 → 数值证书）

对已训练的 sat 模式模型（compare_gen.py 产物），验证三类可证性质：

  Claim 1 [逼近界·收敛阶]：ODE 采样误差随 NFE 的衰减斜率。
      Lipschitz 理论：若 v_θ 为 L-Lipschitz，Euler/midpoint 积分的
      Wasserstein 误差 ≤ C·L·Δt^p（Euler p=1，midpoint p=2）。
      → log-log 下 |CD(NFE) − CD_ref| vs NFE 的拟合斜率应 ≈ −1 / −2。
      观测斜率 = 理论斜率 即误差界的数值验证。

  Claim 2 [路径直线性]：OT-CFM 学到的轨迹是近似直线传输路径，
      离散二阶差分（曲率代理）应显著小于 DDPM 反向扩散轨迹。
      曲率小 → 大步长积分不漂移 → 少步采样有效的**机理证据**。

  Claim 3 [等算力 crossover]：FM Euler 采样的 CD 首次 ≤ DDPM 原生
      T=100 步采样的 CD 所需 NFE。该 NFE 即"质量打平时采样算力
      节省倍数" = T / NFE_crossover。

协议（与 compare_gen.py 一致的可复现设定）：
  - 固定 seed；场景/信道/测试集同 compare_gen sat 模式配置；
  - 加载 ./sat_model_cmp/sat 的 *_best.pth（VAE / DDPM / FM 各自最优）；
  - 同一测试批（n_eval 个样本）、同一 CFG scale=2.0；
  - CD 参考值 = FM midpoint NFE=1000（高阶细步长采样）。
"""

import os
import json
import math
import argparse
import numpy as np
import random
import torch
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import DDPMScheduler, chamfer_distance_loss
from fm_utils import T_EMB_SCALE


def build_eval_batch(args, device, mode="sat"):
    """重建指定模式的场景测试批（seed 固定，协议与 compare_gen.py 一致）。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tle = ss.STARLINK_TLE if args.sat == "starlink" else ss.ISS_TLE
    scenario = ss.SatISACScenario(tau=args.tau, tle_lines=tle, sat_name=args.sat)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=mode, device=device)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pc_gt, cond = next(iter(test_loader))
    return pc_gt[:args.n_eval].to(device), cond[:args.n_eval].to(device), channels.frame_cond_dim()


def load_models(args, device, cond_dim):
    save_dir = os.path.join(args.save_dir, args.mode)
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(save_dir, "vae_best.pth"), map_location=device))
    condenc_d = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                    hidden_size=128, out_emb=256).to(device)
    epsnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                   depth=args.depth, num_heads=8).to(device)
    condenc_d.load_state_dict(torch.load(os.path.join(save_dir, "condenc_best.pth"), map_location=device))
    epsnet.load_state_dict(torch.load(os.path.join(save_dir, "epsnet_best.pth"), map_location=device))
    condenc_f = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                    hidden_size=128, out_emb=256).to(device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    condenc_f.load_state_dict(torch.load(os.path.join(save_dir, "condenc_fm_best.pth"), map_location=device))
    vnet.load_state_dict(torch.load(os.path.join(save_dir, "vnet_fm_best.pth"), map_location=device))
    stats = torch.load(os.path.join(save_dir, "latent_stats.pth"), map_location=device)
    return vae, condenc_d, epsnet, condenc_f, vnet, stats["z_mean"], stats["z_std"]


@torch.no_grad()
def sample_ddpm_traj(vae, condenc, epsnet, sched, cond, z_mean, z_std, device,
                     cfg_scale=2.0):
    """DDPM 原生 T 步反向采样（记录潜空间轨迹）。"""
    vae.eval(); condenc.eval(); epsnet.eval()
    B = cond.size(0)
    zt = torch.randn((B, 256), device=device)
    c = condenc(cond); c_null = condenc(torch.zeros_like(cond))
    traj = [zt.clone()]
    for ti in reversed(range(sched.T)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)
        eps = epsnet(zt, t, c_null) + cfg_scale * (epsnet(zt, t, c) - epsnet(zt, t, c_null))
        beta, alpha, abar = sched.betas[ti], sched.alphas[ti], sched.abar[ti]
        mean = (1.0 / torch.sqrt(alpha)) * (zt - (beta / torch.sqrt(1 - abar + 1e-8)) * eps)
        zt = mean + torch.sqrt(beta) * torch.randn_like(zt) if ti > 0 else mean
        traj.append(zt.clone())
    return zt * z_std + z_mean, traj


@torch.no_grad()
def sample_fm_traj(vae, condenc, vnet, cond, z_mean, z_std, device,
                   cfg_scale=2.0, nfe=20, solver="euler", x0=None):
    """FM ODE 采样（记录潜空间轨迹）。x0 给定则用固定噪声（配对比较）。"""
    vae.eval(); condenc.eval(); vnet.eval()
    B = cond.size(0)
    x = torch.randn((B, 256), device=device) if x0 is None else x0.clone()
    c = condenc(cond); c_null = condenc(torch.zeros_like(cond))

    def v_cfg(xt, t01):
        t = torch.full((B,), t01 * T_EMB_SCALE, device=device)
        return vnet(xt, t, c_null) + cfg_scale * (vnet(xt, t, c) - vnet(xt, t, c_null))

    dt = 1.0 / nfe
    traj = [x.clone()]
    for i in range(nfe):
        t0 = i / nfe
        if solver == "midpoint":
            k1 = v_cfg(x, t0)
            x = x + dt * v_cfg(x + 0.5 * dt * k1, t0 + 0.5 * dt)
        else:
            x = x + dt * v_cfg(x, t0)
        traj.append(x.clone())
    return x * z_std + z_mean, traj


def cd_of(pc_gt, z0, vae):
    with torch.no_grad():
        pc_hat, _ = vae.decode(z0)
        return chamfer_distance_loss(pc_gt, pc_hat).item()


def discrete_curvature(traj):
    """轨迹 [N+1, B, D] → 每步离散二阶差分范数均值（曲率代理，dt=1/N 已归一）。"""
    xs = torch.stack(traj)                      # [N+1, B, D]
    d2 = xs[2:] - 2 * xs[1:-1] + xs[:-2]        # [N-1, B, D]
    n = xs.shape[0] - 1
    return float((d2.norm(dim=-1)).mean().item()) * (n ** 2)   # /dt², dt=1/n


def main(args):
    device = args.device
    print(f"Device: {device}")
    pc_gt, cond, cond_dim = build_eval_batch(args, device, mode=args.mode)
    vae, condenc_d, epsnet, condenc_f, vnet, z_mean, z_std = load_models(args, device, cond_dim)
    sched = DDPMScheduler(T=args.T, device=device)

    # ---------- Claim 1: 收敛阶 ----------
    print(f"\n{'=' * 70}")
    print("Claim 1 — ODE 采样收敛阶（误差界 C·L·Δt^p 的数值验证）")
    print(f"{'=' * 70}")
    g = torch.Generator(device=device).manual_seed(args.x0_seed)
    x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
    _, traj_ref = sample_fm_traj(vae, condenc_f, vnet, cond, z_mean, z_std,
                                 device, nfe=2000, solver="midpoint", x0=x0)
    cd_ref = cd_of(pc_gt, traj_ref[-1], vae)
    z_ref = traj_ref[-1]
    print(f"参考: FM midpoint NFE=2000（固定初始噪声 x0 seed={args.x0_seed}）, CD_ref = {cd_ref:.6f}")

    rows = []
    for solver in ["euler", "midpoint"]:
        for nfe in args.nfe_list:
            _, traj = sample_fm_traj(vae, condenc_f, vnet, cond, z_mean, z_std,
                                     device, nfe=nfe, solver=solver, x0=x0)
            cd = cd_of(pc_gt, traj[-1], vae)
            # 配对潜空间端点误差（ODE 积分误差的直接度量，配对设计降噪）
            err_z = float((traj[-1] - z_ref).norm(dim=-1).mean().item())
            rows.append({"solver": solver, "nfe": nfe, "cd": cd, "err": err_z})
            print(f"  {solver:>8} NFE={nfe:>4}: CD={cd:.6f}  潜空间配对误差={err_z:.6f}")

    slopes = {}
    for solver in ["euler", "midpoint"]:
        errs = [r["err"] for r in rows if r["solver"] == solver]
        nfes = [r["nfe"] for r in rows if r["solver"] == solver]
        pos = [(n, e) for n, e in zip(nfes, errs) if e > 1e-9]
        if len(pos) >= 2:
            slope = float(np.polyfit(np.log([p[0] for p in pos]),
                                     np.log([p[1] for p in pos]), 1)[0])
        else:
            slope = float("nan")
        slopes[solver] = slope
        print(f"  → {solver} 潜空间误差-NFE log-log 拟合斜率 = {slope:.2f}（理论 Euler −1.0, midpoint −2.0）")

    # ---------- Claim 2: 轨迹直线性 ----------
    print(f"\n{'=' * 70}")
    print("Claim 2 — 轨迹直线性（离散曲率代理，FM vs DDPM）")
    print(f"{'=' * 70}")
    k_fm = discrete_curvature(traj_ref)
    _, traj_d = sample_ddpm_traj(vae, condenc_d, epsnet, sched, cond, z_mean, z_std, device)
    k_ddpm = discrete_curvature(traj_d)
    print(f"  FM    轨迹平均离散曲率 = {k_fm:.4f}")
    print(f"  DDPM  轨迹平均离散曲率 = {k_ddpm:.4f}")
    print(f"  → FM/DDPM 曲率比 = {k_fm / k_ddpm:.3f}（越小直线性越好）")

    # ---------- Claim 3: 等质量 crossover NFE ----------
    print(f"\n{'=' * 70}")
    print("Claim 3 — 等质量 crossover：FM Euler 追上 DDPM(T=100) 所需 NFE")
    print(f"{'=' * 70}")
    cd_ddpm = cd_of(pc_gt, traj_d[-1], vae)
    cd_by_nfe = {r["nfe"]: r["cd"] for r in rows if r["solver"] == "euler"}
    crossover = None
    for nfe in sorted(cd_by_nfe):
        if cd_by_nfe[nfe] <= cd_ddpm:
            crossover = nfe
            break
    print(f"  DDPM NFE=100 CD = {cd_ddpm:.6f}")
    print(f"  FM euler NFE={crossover} CD = {cd_by_nfe.get(crossover, float('nan')):.6f}")
    if crossover:
        print(f"  → 采样算力节省 = 100 / {crossover} ≈ {100 / crossover:.0f}×")

    # ---------- 汇总 ----------
    summary = {
        "claim1_convergence_order": {
            "slope_euler": slopes["euler"], "theory_euler": -1.0,
            "slope_midpoint": slopes["midpoint"], "theory_midpoint": -2.0,
            "cd_ref_midpoint_nfe1000": cd_ref,
        },
        "claim2_straightness": {
            "fm_curvature": k_fm, "ddpm_curvature": k_ddpm,
            "ratio": k_fm / k_ddpm,
        },
        "claim3_crossover": {
            "cd_ddpm_t100": cd_ddpm, "crossover_nfe": crossover,
            "speedup_x": (100 / crossover) if crossover else None,
        },
        "protocol": {"seed": args.seed, "x0_seed": args.x0_seed, "n_eval": args.n_eval,
                     "T": args.T, "cfg_scale": 2.0, "torch": torch.__version__},
    }
    out = os.path.join(args.save_dir, "verify_fm_bounds.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM 优化有效性量化验证")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--nfe_list", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x0_seed", type=int, default=7,
                        help="初始噪声种子（common random numbers 配对设计）")
    parser.add_argument("--mode", choices=["none", "sat", "ground"], default="sat",
                        help="加载哪个 IRS 模式的 checkpoint")
    parser.add_argument("--sat", choices=["iss", "starlink"], default="iss",
                        help="卫星（泛化：ISS vs Starlink TLE）")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
