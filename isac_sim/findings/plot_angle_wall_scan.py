"""角度墙配置扫描 + 双站反例出图（CPU 秒级，纯 numpy/matplotlib）。

产出（isac_sim/findings/）：
  - angle_wall_scan.png   左：N × 斜距 的 shortfall 热力图（固定 80 m ROI）；
                          右：交叉距离误差对比（单站角度墙 vs 双站三边定位）
  - 终端报告：默认场景数字 + "墙外"配置需求

复现：make finding-angle-wall（或 python isac_sim/findings/plot_angle_wall_scan.py）
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isac_sim.findings.far_field_angle_wall import (
    C_LIGHT, angular_extent, rayleigh_limit_rad, required_array_aperture,
    trilateration_cross_range_error, scan_shortfall,
)

OUT = Path(__file__).resolve().parent
CARRIER_HZ = 30e9
ROI_W = 80.0
BS_UE_BASELINE_M = 47.7e3  # isac_sat 默认场景：UE 与 ROI 经度差 0.5°


def main():
    lam = C_LIGHT / CARRIER_HZ

    # ---- 左图：N × 斜距 shortfall 热力图（log10 色标）----
    n_list = [4, 8, 16, 32, 64, 128, 256]
    ranges = np.geomspace(100e3, 2000e3, 40)
    shortfall = scan_shortfall(n_list, ranges, ROI_W, lam)

    # ---- 右图：单站角度墙 vs 双站三边定位的交叉距离误差 ----
    # γ 由真实几何决定（isac_sat 默认场景：BS 仰角 33.7°，UE 方位差 142.5°）→ γ ≈ 131°
    rho = C_LIGHT / (2 * 1e9)   # 与 isac_sat 一致的 1 GHz 带宽距离分辨率
    gamma_deg = 131.1
    err_trilater = trilateration_cross_range_error(rho, math.radians(gamma_deg))
    # 蒙特卡洛实测（verify_twostation_localization.py，线性区）约 1.5× 一阶公式
    err_mc = 0.31

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    im = ax1.imshow(np.log10(shortfall), aspect="auto", origin="lower",
                    extent=[ranges[0] / 1e3, ranges[-1] / 1e3, 0, len(n_list) - 1],
                    cmap="inferno")
    ax1.set_yticks(range(len(n_list)))
    ax1.set_yticklabels([f"N={n}" for n in n_list])
    ax1.set_xlabel("Slant range (km)")
    ax1.set_title(f"Angle-wall shortfall  log10($\\theta_{{res}}$ / $\\theta_{{ROI}}$)\n"
                  f"ROI width {ROI_W:.0f} m @ {CARRIER_HZ/1e9:.0f} GHz "
                  f"(≥0 means wall active)")
    fig.colorbar(im, ax=ax1, label="log10 shortfall")

    ax2.axhspan(0, 40, color="0.92")
    ax2.axhline(11.84, color="r", lw=2,
                label="mono-static angle wall (ML, 11.8 m)")
    ax2.axhline(err_trilater, color="g", lw=2,
                label=f"two-station trilateration ({err_trilater:.2f} m theory, {err_mc:.2f} MC)")
    ax2.set_xlabel("Target cross-range scale (m)")
    ax2.set_ylabel("Cross-range error (m)")
    ax2.set_title(f"Escaping the wall with a 2nd range source\n"
                  f"LOS angle γ≈{gamma_deg:.0f}° (BS elev 33.7°, Δaz 142.5°), ρ={rho:.2f} m")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls="--", alpha=0.4)

    fig.tight_layout()
    out_png = OUT / "angle_wall_scan.png"
    fig.savefig(out_png, dpi=150)

    # ---- 终端报告 ----
    rep = {}
    rep["default_shortfall_8el"] = shortfall[1, np.argmin(np.abs(ranges - 695e3))]
    rep["D_required_80m_695km"] = required_array_aperture(695e3, ROI_W, lam)
    rep["N_required"] = rep["D_required_80m_695km"] / (lam / 2)
    rep["two_station_err"] = err_mc
    rep["mono_static_ml"] = 11.84
    print("=" * 66)
    print("远场角度墙：配置扫描 + 双站反例（30 GHz / 80 m ROI）")
    print("-" * 66)
    print(f"默认场景（N=8, R=695 km）shortfall          : {rep['default_shortfall_8el']:8.0f}×")
    print(f"分辨 80 m ROI 所需孔径（R=695 km）           : {rep['D_required_80m_695km']:6.1f} m"
          f"（≈ N={rep['N_required']:.0f} @ λ/2）")
    print(f"单站交叉距离误差（ML, 实测）                 : {rep['mono_static_ml']:6.1f} m  ← 角度墙")
    print(f"双站三边定位（UE 第二测距源, ρ={rho:.2f} m） : {rep['two_station_err']:6.2f} m  ← 墙外")
    print(f"改善倍数                                    : {rep['mono_static_ml']/rep['two_station_err']:6.1f}×")
    print(f"\n[图] {out_png}")
    return rep


if __name__ == "__main__":
    main()
