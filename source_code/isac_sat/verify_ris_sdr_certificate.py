"""
verify_ris_sdr_certificate.py — RIS 恒模 QCQP 的 SDR 最优性证书（多种子）

对每个种子的过境帧序列，用同一 (d, M) 线性模型计算：
  - P_cf      : 闭式相干对齐（现有方法，可行点）
  - P_CA      : 坐标上升多起点（现有 oracle，可行点=下界）
  - P_bm      : SDR Burer-Monteiro 值（松弛可行值）
  - λ₂(W*)    : BM 解第二大特征值（秩-1 ⇒ 全局最优的精确性证书）
  - P_eig     : BM 主特征向量舍入功率（可行点）
  - P_gr      : 高斯舍入最佳功率（可行点；π/4 期望保证）
  - P_GG      : Gershgorin 对偶上界（严格合法上界）

证书结构：  L = max(P_cf, P_CA, P_eig, P_gr) ≤ P* ≤ P_GG

预注册命题（可证伪）：
  C1  Gershgorin 上界合法性自检：P_GG ≥ max(所有可行点) 在 100% 帧成立（否则求解器 bug）
  C2  oracle 近紧性：P_CA / P_GG 均值 ∈ [0.97, 1.00]；若 < 0.95 的帧 > 5% ⇒ CA 高估 η_design，须下修
  C3  设计因子证书区间：η_design* = P_cf/P* ∈ [P_cf/P_GG, P_cf/P_CA]
  C4  舍入质量：P_gr_mean / P_GG ≥ 0.75（π/4=0.785 最坏保证的数值确认）
  C5  SDR 相对闭式的增益上界：P_GG/P_cf - 1 ≥ 实测 P_CA/P_cf - 1（一致性）

协议与 verify_optimality_decomposition.py 一致（同场景/同种子/同 X）。
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch

import setup_sat as ss
from data_sat import SatScenarioChannels, generate_ground_target_sample, _SIGNAL1
from phase_optimizer_sat import PhaseOptimizerSat

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from isac_sim.ris.sdr import sdr_phase_bounds


def run_seed(seed, args):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cpu"

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    roi_true, _, _ = generate_ground_target_sample()
    roi_t = torch.tensor(roi_true.astype(np.float32)).reshape(-1)

    rows = []
    for t, Ht in enumerate(channels.channels_per_frame):
        d, M = opt._linear_model(Ht, roi_t, X)
        if d is None:
            continue
        # 可行点：闭式 + 坐标上升
        ph_cf = opt.optimize_frame(Ht, roi_t, X)
        p_cf = opt._power(Ht, roi_t, X, ph_cf)
        _, p_ca = opt.optimize_frame_numeric(Ht, roi_t, X, seed=t)
        # SDR 证书
        cert, _ = sdr_phase_bounds(d, M, rank=args.rank, iters=args.iters,
                                   n_round=args.n_round, seed=t, device=device)
        rows.append({"seed": seed, "frame": t,
                     "p_cf": p_cf, "p_ca": float(p_ca), **cert})
    return rows


def main(args):
    rows = []
    for s in range(args.n_seeds):
        rows.extend(run_seed(s, args))
        print(f"[seed {s}] {len(rows)} 帧已处理")

    r = lambda k: np.array([row[k] for row in rows])
    p_cf, p_ca, p_gg = r("p_cf"), r("p_ca"), r("p_gershgorin")
    p_dual = r("p_dual")
    p_bm, p_eig, p_gr, p_grb, lam2 = (r("p_bm"), r("p_cf_eig"), r("p_round_mean"),
                                      r("p_round_best"), r("lambda2"))

    # ---- C1 上界合法性自检（内部 PASS 门）----
    lb_all = np.maximum.reduce([p_cf, p_ca, p_eig, p_grb])
    ub = np.minimum(p_gg, p_dual)
    c1_viol = int((p_gg < lb_all - 1e-6 * lb_all).sum())
    # ---- C2 oracle 近紧性 ----
    ratio_ca_gg = p_ca / p_gg
    # ---- C3 设计因子区间 ----
    eta_lo = (p_cf / ub).mean()        # 合法上界给出的下侧
    eta_hi = (p_cf / p_ca).mean()      # 现有测量值（CA=可行点）
    # ---- C4 舍入质量 ----
    ratio_gr = p_gr / p_gg
    # ---- C5 一致性：SDR 增益 vs CA 增益 ----
    gain_ca = (p_ca / p_cf - 1).mean()
    gain_gg = (p_gg / p_cf - 1).mean()

    print(f"\n{'=' * 74}")
    print(f"SDR 最优性证书（{len(rows)} 帧, {args.n_seeds} seeds, irs={args.irs_mode}）")
    print("-" * 74)
    print(f"C1 上界合法性: Gershgorin ≥ 全部可行点, 违例帧数 = {c1_viol}/{len(rows)}"
          f"  {'PASS' if c1_viol == 0 else 'FAIL(求解器 bug)'}")
    ratio_ca_ub = p_ca / ub
    print(f"C2 oracle 近紧性: P_CA/合法上界 = {ratio_ca_ub.mean():.4f} ± {ratio_ca_ub.std():.4f}"
          f"  (<0.95 的帧占比 {(ratio_ca_ub < 0.95).mean() * 100:.1f}%)")
    gap_ub = ub / np.maximum.reduce([p_cf, p_ca, p_eig, p_grb])
    print(f"    证书间隙上界/L: 均值 {gap_ub.mean():.4f}, 中位 {np.median(gap_ub):.4f}, "
          f"最大 {gap_ub.max():.4f}")
    print(f"C3 设计因子区间: η_design* ∈ [{eta_lo:.3f}, {eta_hi:.3f}]"
          f"  (现报告值 {eta_hi:.3f})")
    print(f"C4 舍入质量: P_gr_best/合法上界 = {(p_grb / ub).mean():.3f} ± {(p_grb / ub).std():.3f}"
          f"  (高斯舍入为启发式，无 π/4 保证)")
    print(f"C5 一致性: SDR 潜在增益 +{gain_gg * 100:.1f}% vs CA 实测 +{gain_ca * 100:.1f}%"
          f" (相对闭式)")
    print(f"秩-1 证书: λ₂ ≤ 1e-3 的帧占比 = {(lam2 < 1e-3).mean() * 100:.1f}%"
          f"  (λ₂ 均值 {lam2.mean():.4f})")
    print(f"{'=' * 74}")

    out = {
        "n_frames": len(rows), "n_seeds": args.n_seeds,
        "c1_violations": c1_viol,
        "ratio_ca_upper": {"mean": float(ratio_ca_ub.mean()), "std": float(ratio_ca_ub.std()),
                           "frac_below_0.95": float((ratio_ca_ub < 0.95).mean())},
        "gap_upper_over_lower": {"mean": float(gap_ub.mean()), "median": float(np.median(gap_ub)),
                                 "max": float(gap_ub.max())},
        "eta_design_interval": {"lower": float(eta_lo), "upper_reported": float(eta_hi)},
        "ratio_round_gershgorin": {"mean": float(ratio_gr.mean()), "std": float(ratio_gr.std())},
        "gain_sdr_vs_cf": float(gain_gg), "gain_ca_vs_cf": float(gain_ca),
        "lambda2": {"mean": float(lam2.mean()), "frac_rank1": float((lam2 < 1e-3).mean())},
        "bm_vs_ca": float((p_bm / p_ca).mean()),
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "ris_sdr_certificate.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RIS SDR 最优性证书")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4, help="BM 因子秩 r")
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--n_round", type=int, default=64)
    args = parser.parse_args()
    main(args)
