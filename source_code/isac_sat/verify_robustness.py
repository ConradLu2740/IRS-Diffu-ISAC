"""verify_robustness.py — 多轨道 / Ka 频段鲁棒性验证

组合矩阵：{ISS, Starlink} × {30GHz, 28GHz Ka}
每组输出：轨道参数（高度/速度/周期）、过境窗口、多普勒范围、时延。
目的：验证星-地 ISAC 物理层对不同轨道与频段是否稳定合理。
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss

OUT_DIR = "./sat_verify"
os.makedirs(OUT_DIR, exist_ok=True)

ORBITS = [
    ("ISS", ss.ISS_TLE),
    ("Starlink", ss.STARLINK_TLE),
]
FREQS = [30e9, 28e9]


def orbit_stats(scenario, span_min=180):
    """粗略轨道参数：高度/速度（传播 span_min 分钟）。"""
    sat = scenario.sat
    jd0, fr0 = ss.jday(*scenario.start_utc)
    jd0_full = jd0 + fr0
    alts, speeds = [], []
    n = 300
    for i in range(n):
        jd_full = jd0_full + span_min * 60 * i / (n - 1) / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, v_eci = sat.sgp4(int(jd_i), fr_i)
        if err == 0:
            alts.append(np.linalg.norm(r_eci) - ss.EARTH_R_KM)
            speeds.append(np.linalg.norm(v_eci))
    return float(np.mean(alts)), float(np.mean(speeds))


def doppler_range(scenario, windows):
    """过境窗口内多普勒 min/max（Hz）。"""
    if not windows:
        return 0.0, 0.0
    win = windows[0]
    jd0, fr0 = ss.jday(*scenario.start_utc)
    jd0_full = jd0 + fr0
    fds = []
    for t in np.linspace(win[0], win[1], 100):
        jd_full = jd0_full + t / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, v_eci = scenario.sat.sgp4(int(jd_i), fr_i)
        if err != 0:
            continue
        r_ecef, v_ecef = ss.eci_to_ecef(r_eci, v_eci, jd_full)
        v_rel = ss.radial_velocity_mps(v_ecef, np.zeros(3), r_ecef, scenario.target_ecef)
        fds.append(v_rel / scenario.wavelength_m)
    return float(np.min(fds)), float(np.max(fds))


def main():
    rows = []
    for orbit_name, tle in ORBITS:
        for fc in FREQS:
            scenario = ss.SatISACScenario(tle_lines=tle, fc_hz=fc, sat_name=orbit_name)
            alt, spd = orbit_stats(scenario)
            windows = scenario.find_overpass()
            fmin, fmax = doppler_range(scenario, windows)
            frames = scenario.build_frames()
            delay = frames[len(frames) // 2]["delay_s"] * 1e3
            n_win = len(windows)
            rows.append({
                "orbit": orbit_name, "freq": f"{fc/1e9:.0f} GHz",
                "alt_km": alt, "speed": spd, "n_windows": n_win,
                "fmin_kHz": fmin / 1e3, "fmax_kHz": fmax / 1e3,
                "delay_ms": delay,
            })
            print(f"[{orbit_name:>8} | {fc/1e9:.0f}GHz] 高度={alt:6.1f}km 速度={spd:.3f}km/s "
                  f"窗口={n_win} 多普勒=[{fmin/1e3:7.1f},{fmax/1e3:7.1f}]kHz 时延={delay:.2f}ms")

    # 汇总表
    print("\n" + "=" * 92)
    print(f"{'轨道':<10}{'频段':<8}{'高度km':>9}{'速度km/s':>10}{'窗口数':>7}"
          f"{'多普勒kHz':>14}{'时延ms':>9}")
    print("-" * 92)
    for r in rows:
        print(f"{r['orbit']:<10}{r['freq']:<8}{r['alt_km']:>9.1f}{r['speed']:>10.3f}"
              f"{r['n_windows']:>7}[{r['fmin_kHz']:>6.1f},{r['fmax_kHz']:>6.1f}]"
              f"{r['delay_ms']:>9.2f}")
    print("=" * 92)

    # 图：多普勒范围对比（轨道 × 频段）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    labels = [f"{r['orbit']}\n{r['freq']}" for r in rows]
    mins = [r["fmin_kHz"] for r in rows]
    maxs = [r["fmax_kHz"] for r in rows]
    ax.bar(x - 0.2, mins, width=0.4, label="min f_d", color="steelblue")
    ax.bar(x + 0.2, maxs, width=0.4, label="max f_d", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Doppler shift (kHz)")
    ax.set_title("Doppler Range by Orbit and Frequency")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "robustness_doppler.png"), dpi=150)
    print(f"\n[图] robustness_doppler.png 已保存")

    print("\n鲁棒性检查:")
    for r in rows:
        ok_alt = 300 < r["alt_km"] < 600
        ok_speed = 7.3 < r["speed"] < 7.9
        ok_fd = abs(r["fmax_kHz"]) < 800
        print(f"  {r['orbit']} {r['freq']}: 高度{'PASS' if ok_alt else 'FAIL'}"
              f" 速度{'PASS' if ok_speed else 'FAIL'} 多普勒{'PASS' if ok_fd else 'FAIL'}")


if __name__ == "__main__":
    main()
