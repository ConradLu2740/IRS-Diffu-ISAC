"""verify_tracking.py — 动态 RIS 相位跟踪验证

对比策略：random（随机相位基线）/ track_K=1（理想逐帧）/ track_K=2/4/8（分段跟踪）
输出：平均接收功率对比 + 图（功率随帧变化 + 各策略汇总条形图）
"""
import os
import math
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


def make_roi_voxel():
    """生成一个简单 ROI 体素（中心物体）作为优化目标。"""
    from data import generate_ROI
    return torch.tensor(generate_ROI().astype("float32")).reshape(-1)


def main():
    device = "cpu"
    print(f"Device: {device}")
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)

    ROI = make_roi_voxel()
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:4], dtype=torch.complex64).view(4, 1)

    print("对比 RIS 相位跟踪策略（平均接收功率，越大越好）...")
    results = compare_tracking(channels, ROI, X, device=device,
                               n_iter=args.n_iter, intervals=args.intervals)

    print(f"\n{'策略':<14}{'平均功率':>14}{'相对随机提升':>14}")
    rand_p = results["random"]["power"]
    for name, r in results.items():
        boost = (r["power"] / rand_p - 1) * 100 if rand_p > 0 else 0
        print(f"{name:<14}{r['power']:14.6e}{boost:13.1f}%")
        if name == "track_K=1":
            ideal_p = r["power"]

    # ---- 图 1：各帧功率曲线 ----
    n_frames = len(frames)
    t = np.arange(n_frames)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in results.items():
        phases = r["phases"]
        pows = []
        for t_idx, Ht in enumerate(channels.channels_per_frame):
            if "H_ROI_IRS" not in Ht or t_idx >= len(phases):
                pows.append(np.nan)
            else:
                pows.append(PhaseOptimizerSat(channels, device=device)
                            ._power(Ht, ROI, X, phases[t_idx]))
        ax.plot(t, np.array(pows), marker="o", ms=4, label=name)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Received power")
    ax.set_title("RIS Phase Tracking: Received Power per Frame")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tracking_power.png"), dpi=150)
    print(f"\n[图] tracking_power.png 已保存")

    # ---- 图 2：策略汇总条形图 ----
    names = list(results.keys())
    powers = [results[n]["power"] for n in names]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.bar(names, powers, color=["gray", "steelblue", "mediumseagreen",
                                          "orange", "crimson"][:len(names)])
    ax2.set_ylabel("Mean received power")
    ax2.set_title("RIS Phase Tracking Strategy Comparison")
    ax2.set_xticklabels(names, rotation=15)
    ax2.grid(True, axis="y", ls="--", alpha=0.4)
    for b, p in zip(bars, powers):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                 f"{p:.2e}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "tracking_summary.png"), dpi=150)
    print("[图] tracking_summary.png 已保存")

    print("\n结论检查:")
    print(f"  理想跟踪 vs 随机: {(ideal_p / rand_p - 1) * 100:.1f}% 提升"
          f"{'  (PASS, 优化器有效)' if ideal_p > rand_p * 1.05 else '  (FAIL?)'}")
    for name, r in results.items():
        if name.startswith("track_K=") and name != "track_K=1":
            k = int(name.split("=")[1])
            if r["power"] < ideal_p:
                print(f"  {name} 功率低于理想跟踪 (符合预期: 重构速率受限)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()
    main()
