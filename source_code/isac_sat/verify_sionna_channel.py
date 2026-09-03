"""verify_sionna_channel.py — L2：Sionna CDL 标准信道对照（3GPP TR 38.901）

目的：P0/P1/P2 的全部信道结论建立在自建 L0/L1 信道（自由空间 +
逐帧独立莱斯）上。本脚本用 Sionna 2.x（PyTorch 后端）的 3GPP CDL-D
标准剖面做三项对照，检验自建模型的两个核心近似：

  (a) 平坦衰落近似：CDL-D 时延剖面在 30 GHz / 1 GHz 感知带宽下的
      频率相关函数 |ρ(Δf)|。K≈9 dB 的 LOS 主导剖面中，NLOS 引入的
      频率选择性波纹幅度 ~1/(K+1)；相干带宽 @0.9 与 1 GHz 带宽对比。
  (b) 逐帧独立近似：CDL-D CIR 时间序列在 LEO 相对速度（Sionna
      Jakes 多普勒）下的时间自相关。去相关时间 τ_c 与帧间隔
      FRAME_INTERVAL_S=1 s 对比；LOS 分量为确定性相位旋转
      （f_d = v_rel/λ ~ ±700 kHz），逐帧跟踪优化器只依赖当前帧 CSI，
      可吸收。
  (c) K 档位对齐：CDL-D 剖面 K 因子（由 38.901 表固定，~9 dB）与
      P1 K-sweep 档位 {10, 5, 0} dB 的关系——P1 的 K=10 dB 档
      近似覆盖标准信道的散射强度。

诚实边界：
  - Sionna 未内置 TR 38.811 NTN 剖面，CDL-D 是 38.901 地面剖面代理；
    LOS 结构与星-地链路一致，但时延扩展/角度弥散取地面典型值。
  - (b) 中 Sionna 的多普勒为 Jakes 各向同性散射模型，非真实星-地
    几何多普勒；真实场景帧间 f_d 随仰角演化（f_d_hz 逐帧输出）。

用法：
  cd source_code/isac_sat && python verify_sionna_channel.py [--skip-tracking]
输出：./sat_verify/sionna_channel_comparison.json + .png
依赖：pip install sionna（可选依赖，不影响仓库其余部分）
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import SatScenarioChannels

OUT_DIR = "./sat_verify"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="D", choices=["A", "B", "C", "D", "E"],
                        help="CDL 模型（D/E 含 LOS）")
    parser.add_argument("--delay-spread-ns", type=float, default=100.0,
                        help="RMS 时延扩展（ns）")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="跳过 (c) 的跟踪重跑（只做剖面级对照）")
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 仓库根目录加入 sys.path（脚本按 Makefile 约定从 source_code/isac_sat 运行）
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from isac_sim.channels.sionna_cdl import SionnaCdlChannel

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 场景：与 P1/P2 完全相同的 TLE 过境帧 ----
    print("[1/4] 构建 SGP4 过境场景（ISS TLE，30 GHz）...")
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    base = SatScenarioChannels(frames, irs_mode="sat", device="cpu")

    f_ds = [float(ht["f_d_bs_roi"]) for ht in base.channels_per_frame]
    v_rels = [abs(fd) * base.wavelength_m for fd in f_ds]
    elevs = [float(ht["elevation_deg"]) for ht in base.channels_per_frame]
    v_mid = v_rels[len(v_rels) // 2]
    print(f"  帧数: {len(frames)}，帧间隔 {ss.FRAME_INTERVAL_S} s")
    print(f"  仰角: {min(elevs):.1f}° - {max(elevs):.1f}°")
    print(f"  |v_rel|: {min(v_rels):.0f} - {max(v_rels):.0f} m/s"
          f"（中位 {v_mid:.0f}）→ f_d = ±{max(f_ds)/1e3:.0f} kHz @ 30 GHz")

    # ---- Sionna CDL-D ----
    print(f"\n[2/4] Sionna CDL-{args.model}（DS={args.delay_spread_ns} ns，"
          f"v={v_mid:.0f} m/s）...")
    ch = SionnaCdlChannel(carrier_hz=ss.FC_HZ, model=args.model,
                          delay_spread_s=args.delay_spread_ns * 1e-9,
                          ut_speed_mps=v_mid, seed=args.seed)
    k_db = ch.empirical_k_factor_db()
    print(f"  剖面 K 因子: {k_db:.2f} dB")

    # ---- (a) 频率选择性 ----
    print("\n[3/4] (a) 频率相关函数...")
    fc = ch.frequency_correlation(max_delay_hz=2e9, n_points=4096)
    B = 1e9  # isac_sim/waveforms/ofdm.py 默认感知带宽
    n_sc = 512
    scs = B / n_sc
    df_arr, rho = fc["freqs_hz"], fc["rho"]
    rho_at = lambda x: float(rho[np.searchsorted(df_arr, x)])
    result_freq = {
        "k_profile_db": k_db,
        "coherence_bw_09_hz": fc["coherence_bw_09_hz"],
        "coherence_bw_05_hz": fc["coherence_bw_05_hz"],  # 强 LOS 下可能 nan
        "rho_at_1GHz": rho_at(B),
        "rho_at_scs": rho_at(scs),
        "sensing_bw_hz": B,
        "scs_hz": scs,
    }
    print(f"  |ρ(Δf={scs/1e6:.2f} MHz)|（1 子载波）: {result_freq['rho_at_scs']:.4f}")
    print(f"  |ρ(Δf=1 GHz)|（全带宽）  : {result_freq['rho_at_1GHz']:.4f}")
    print(f"  相干带宽 @0.9: {result_freq['coherence_bw_09_hz']/1e6:.1f} MHz"
          f"（@0.5: {'nan（LOS 地板 %.3f)' % (ch.k_factor_linear/(1+ch.k_factor_linear)) if np.isnan(result_freq['coherence_bw_05_hz']) else '%.1f MHz' % (result_freq['coherence_bw_05_hz']/1e6)}）")

    # ---- (b) 时间选择性 ----
    print("\n(b) 时间自相关（Sionna Jakes 多普勒）...")
    ta = ch.temporal_autocorrelation(sampling_frequency=2e6,
                                     num_time_steps=2048, batch_size=8)
    jakes_zero_s = 2.405 / (2 * np.pi * v_mid / base.wavelength_m)
    result_time = {
        "decorr_time_05_s": ta["decorr_time_05_s"],
        "decorr_time_09_s": ta["decorr_time_09_s"],
        "jakes_first_zero_s": jakes_zero_s,
        "los_floor_theory": ta["los_floor"],
        "frame_interval_s": float(ss.FRAME_INTERVAL_S),
        "ratio_decorr_to_frame": ta["decorr_time_05_s"] / ss.FRAME_INTERVAL_S,
        "fd_max_hz": v_mid / base.wavelength_m,
    }
    print(f"  NLOS 散射去相关时间 @0.5: {result_time['decorr_time_05_s']*1e6:.1f} µs"
          f"（Jakes 理论首次过零 {jakes_zero_s*1e6:.1f} µs）"
          f"  vs 帧间隔 {ss.FRAME_INTERVAL_S} s"
          f"（比值 {result_time['ratio_decorr_to_frame']:.2e}）")
    print(f"  全信道 |ρ| 地板（LOS 主导）理论 K/(K+1) = {ta['los_floor']:.3f}；"
          f"LOS 相位旋转为确定性（场景逐帧 f_d 相位已显式建模）")

    # ---- (c) 跟踪结论在 K_cdl 档位重跑（复用 P1 机制）----
    tracking = None
    if not args.skip_tracking:
        print(f"\n[4/4] (c) K-sweep 跟踪重跑 @ K=CDL-D 剖面（{k_db:.2f} dB）...")
        from verify_tracking_rician import perturb_channels, run_boosts
        from verify_tracking import make_roi_voxel
        ROI = make_roi_voxel(prefer_legacy=True)
        X = base.tensor_a * torch.tensor(
            __import__("data_sat")._SIGNAL1[:4], dtype=torch.complex64).view(4, 1)
        intervals = [1, 2, 4, 8]
        per_seed = {k: [] for k in intervals}
        for s in range(args.n_seeds):
            ch_p = perturb_channels(base, k_db, seed=args.seed + s)
            boosts, _ = run_boosts(ch_p, ROI, X, intervals, n_iter=5)
            for k in intervals:
                per_seed[k].append(boosts[k])
            print("  seed {}:".format(args.seed + s)
                  + "  ".join(f"K={k}:{boosts[k]:+.1f}%" for k in intervals))
        tracking = {
            "k_db": k_db,
            "n_seeds": args.n_seeds,
            "boost_mean_pct": {k: float(np.mean(per_seed[k])) for k in intervals},
            "boost_std_pct": {k: float(np.std(per_seed[k], ddof=1))
                              if len(per_seed[k]) > 1 else 0.0 for k in intervals},
            "p1_reference_k10_db": {1: 89.0, 2: 60.0, 4: 36.6, 8: -41.5},
        }

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogx(df_arr[1:] / 1e6, rho[1:], lw=1.5)
    ax.axhline(ch.k_factor_linear / (1 + ch.k_factor_linear), ls="--", c="gray",
               lw=0.8, label=f"LOS floor K/(K+1)={ch.k_factor_linear/(1+ch.k_factor_linear):.3f}")
    ax.axvline(B / 1e6, c="r", ls=":", lw=1.0, label="sensing BW 1 GHz")
    ax.set_xlabel("Δf (MHz)"), ax.set_ylabel("|ρ(Δf)|")
    ax.set_title(f"(a) CDL-{args.model} 频率相关（30 GHz, DS={args.delay_spread_ns:.0f}ns, K={k_db:.1f}dB）")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax = axes[1]
    lags_us = ta["lags_s"] * 1e6
    ax.plot(lags_us, ta["rho_nlos"], lw=1.5, label="NLOS scatter")
    ax.plot(lags_us, ta["rho_full"], lw=0.9, alpha=0.6, label="full (LOS+NLOS)")
    ax.axhline(ta["los_floor"], ls="--", c="gray", lw=0.8,
               label=f"LOS floor K/(K+1)={ta['los_floor']:.3f}")
    ax.axvline(ss.FRAME_INTERVAL_S * 1e6, c="r", ls=":", lw=1.2,
               label=f"frame interval {ss.FRAME_INTERVAL_S:.0f}s")
    ax.set_xlabel("Δt (µs)"), ax.set_ylabel("|ρ(Δt)|")
    ax.set_xlim(0, 500)
    ax.set_title(f"(b) 时间自相关（v={v_mid:.0f} m/s → f_d≈{result_time['fd_max_hz']/1e3:.0f} kHz）")
    ax.legend(fontsize=7), ax.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "sionna_channel_comparison.png")
    plt.savefig(png, dpi=150)
    print(f"\n图已保存: {png}")

    # ---- JSON 汇总 ----
    out = {
        "tool": "Sionna 2.x CDL (TR 38.901) 对照，PyTorch 后端",
        "scenario": {"carrier_hz": ss.FC_HZ, "frames": len(frames),
                     "elev_deg": [min(elevs), max(elevs)],
                     "v_rel_mps": [min(v_rels), max(v_rels)],
                     "fd_hz": [min(f_ds), max(f_ds)]},
        "cdl": {"model": args.model,
                "delay_spread_ns": args.delay_spread_ns,
                "k_factor_db": k_db},
        "freq_selectivity": result_freq,
        "temporal_selectivity": result_time,
        "tracking_at_cdl_k": tracking,
    }
    js = os.path.join(OUT_DIR, "sionna_channel_comparison.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"数据已保存: {js}")


if __name__ == "__main__":
    main()
