"""verify_tracking_rician.py — K-sweep 结论在莱斯衰落（L1 信道）下的稳健性验证

目的：TECH_REPORT v1.3 的自由空间 headline（K=1: +89.0% / K=2: +60.0% /
K=4: +36.6% / K=8: -41.5%）建立在理想自由空间信道上。本脚本向 isac_sat
场景信道注入**逐帧独立**的莱斯衰落（K 因子可调），重跑重构速率扫描，
检验"重构速率 vs 相干时间"结论对信道保真度的稳健性。

衰落模型（与 isac_sim/channels/rician.py 同构，逐元素功率对齐）：
    H' = sqrt(K/(K+1)) * H + sqrt(1/(K+1)) * |H| ⊙ z
    z ~ CN(0,1)（每实/虚分量方差 1/2），每帧每链路独立采样。
E|H'|^2 = |H|^2：莱斯化不改变平均链路预算，只引入时间选择性衰落。
K→∞ 退化为自由空间；K=0 dB（瑞利）为时间选择性最强的档位。

物理预期：
  - K=1（逐帧理想跟踪）提升仍显著为正（优化器只依赖当前帧 CSI）；
  - "boost 随 K 单调递减、K=8 增益消失甚至为负"的定性结论应跨信道档位保持。
    注意：绝对幅度是"相对随机相位"的比值，受随机基线功率共同影响，
    不同 K 因子下的幅度不宜直接横向比较（与 ROI 物体依赖性同理）。
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import SatScenarioChannels, _SIGNAL1
from phase_optimizer_sat import PhaseOptimizerSat, compare_tracking

OUT_DIR = "./sat_verify"
os.makedirs(OUT_DIR, exist_ok=True)

# TECH_REPORT v1.3 自由空间参考值（相对随机相位）
FS_REFERENCE = {1: 89.0, 2: 60.0, 4: 36.6, 8: -41.5}


class _RicianChannels:
    """轻量包装：只替换逐帧信道矩阵，暴露 PhaseOptimizerSat 所需字段。"""

    def __init__(self, base, channels_per_frame):
        self.frames = base.frames
        self.channels_per_frame = channels_per_frame


def perturb_channels(base, k_factor_db: float, seed: int):
    """对每帧每个 H_ 矩阵注入逐帧独立的莱斯衰落（固定种子可复现）。"""
    lin = 10.0 ** (k_factor_db / 10.0)
    a_los = math.sqrt(lin / (1.0 + lin))
    a_sc = math.sqrt(1.0 / (1.0 + lin))
    cpf = []
    for t, ht in enumerate(base.channels_per_frame):
        gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + t)
        new_ht = dict(ht)
        for key, M in ht.items():
            if not key.startswith("H_"):
                continue
            z = (torch.randn(M.shape, generator=gen)
                 + 1j * torch.randn(M.shape, generator=gen)) / math.sqrt(2.0)
            new_ht[key] = (a_los * M + a_sc * M.abs() * z).to(torch.complex64)
        cpf.append(new_ht)
    return _RicianChannels(base, cpf)


def run_boosts(channels, ROI, X, intervals, n_iter):
    """返回 {K: boost%}（相对随机相位基线）。"""
    torch.manual_seed(42)  # 随机相位基线跨设置可复现
    results = compare_tracking(channels, ROI, X, device="cpu",
                               n_iter=n_iter, intervals=intervals)
    rand_p = results["random"]["power"]
    return {k: (results[f"track_K={k}"]["power"] / rand_p - 1.0) * 100.0
            for k in intervals}, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--k_dbs", nargs="+", type=float, default=[10.0, 5.0, 0.0])
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--roi", choices=["legacy", "isac"], default="legacy",
                        help="legacy = 与 TECH_REPORT v1.3 同物体（可比）；isac = data_sat 生成器")
    args = parser.parse_args()

    device = "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    base = SatScenarioChannels(frames, irs_mode="sat", device=device)

    from verify_tracking import make_roi_voxel
    ROI = make_roi_voxel(prefer_legacy=(args.roi == "legacy"))
    X = base.tensor_a * torch.tensor(_SIGNAL1[:4], dtype=torch.complex64).view(4, 1)

    # ---- 0) 自由空间内部基线（与 TECH_REPORT 同设置的本次重跑）----
    print("\n[0/4] 自由空间基线（内部重跑）...")
    fs_boosts, _ = run_boosts(base, ROI, X, args.intervals, args.n_iter)
    settings = {"free-space (in-house)": {k: [v] for k, v in fs_boosts.items()}}

    # ---- 1-3) 莱斯注入，多种子 ----
    for k_db in args.k_dbs:
        print(f"\n[K={k_db:.0f} dB] 逐帧独立莱斯衰落，{args.n_seeds} 个种子...")
        per_seed = {k: [] for k in args.intervals}
        for s in range(args.n_seeds):
            ch = perturb_channels(base, k_db, seed=args.seed + s)
            boosts, _ = run_boosts(ch, ROI, X, args.intervals, args.n_iter)
            for k, b in boosts.items():
                per_seed[k].append(b)
            print("  seed {}: ".format(args.seed + s)
                  + "  ".join(f"K={k}:{boosts[k]:+.1f}%" for k in args.intervals))
        settings[f"Rician K={k_db:.0f} dB"] = per_seed

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print(f"{'信道设置':<22}" + "".join(f"{'K=' + str(k):>13}" for k in args.intervals)
          + "   （相对随机相位，多 seed 均值±std；括号内为 TECH_REPORT v1.3 自由空间参考）")
    print("-" * 78)
    summary = {}
    for name, per_seed in settings.items():
        row = f"{name:<22}"
        summary[name] = {}
        for k in args.intervals:
            arr = np.array(per_seed[k])
            m, sd = arr.mean(), (arr.std(ddof=1) if len(arr) > 1 else 0.0)
            summary[name][k] = (m, sd)
            ref = f" ({FS_REFERENCE[k]:+.1f})" if name.startswith("free-space") else ""
            row += f"{m:+7.1f}±{sd:4.1f}{ref:>10}"
        print(row)

    # ---- 稳健性判定 ----
    print("\n结论检查（跨信道档位的定性稳健性）:")
    ok = True
    for name, s in summary.items():
        if not name.startswith("Rician"):
            continue
        k1 = s[1][0]
        monotone = all(s[args.intervals[i]][0] > s[args.intervals[i + 1]][0]
                       for i in range(len(args.intervals) - 1))
        k8_worse = s[args.intervals[-1]][0] < s[args.intervals[1]][0] \
            if len(args.intervals) > 1 else True
        cond1 = k1 > 20.0
        print(f"  {name}: K=1 提升 {k1:+.1f}% (>+20% {'✓' if cond1 else '✗'}) | "
              f"随 K 单调递减 {'✓' if monotone else '✗'} | "
              f"K={args.intervals[-1]} 比 K={args.intervals[1]} 更差 {'✓' if k8_worse else '✗'}")
        ok = ok and cond1 and monotone and k8_worse

    fs_ok = all(abs(summary["free-space (in-house)"][k][0] - FS_REFERENCE[k]) < 5.0
                for k in args.intervals)
    print(f"  free-space 内部重跑 vs TECH_REPORT 参考（±5%容差）: {'✓ PASS' if fs_ok else '✗ DRIFT'}")
    print(f"\n总体: {'ROBUST — 结论在莱斯信道下保持' if ok else 'SENSITIVE — 结论依赖理想信道，需在报告中注明'}")

    # ---- 图：boost vs K，各信道档位 ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(settings)))
    for (name, s), c in zip(summary.items(), colors):
        ks = args.intervals
        ms = [s[k][0] for k in ks]
        sds = [s[k][1] for k in ks]
        ax.errorbar(ks, ms, yerr=sds, marker="o", ms=5, capsize=3, label=name, color=c)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("RIS reconfiguration interval K (frames)")
    ax.set_ylabel("Power boost vs random phase (%)")
    ax.set_title("RIS tracking trade-off under Rician fading (per-frame independent)")
    ax.set_xticks(args.intervals)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tracking_rician_summary.png"), dpi=150)
    print(f"\n[图] {OUT_DIR}/tracking_rician_summary.png 已保存")


if __name__ == "__main__":
    main()
