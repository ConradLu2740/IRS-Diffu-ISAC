"""verify_sat.py — 星-地 ISAC 场景物理正确性验证

验证项：
  1. 轨道参数：高度、速度、周期（对照 ISS 真实参数）
  2. 过境窗口：数量、时长、中心时刻
  3. 动态几何：整个过境窗口内 仰角 / 距离 / 多普勒 变化曲线（S 型多普勒）
  4. 信道与相位：get_channel_far 输出远场信道 + 多普勒相位因子

输出：数值检查 + 保存 sat_verify/*.png 可视化
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss

OUT_DIR = "./sat_verify"
os.makedirs(OUT_DIR, exist_ok=True)


def verify_orbit(scenario):
    """传播 2 个轨道周期，验证高度/速度/周期。"""
    print("=" * 70)
    print("[1] 轨道参数验证 (对照 ISS: 高度~420km, 速度~7.66 km/s, 周期~92.9 min)")
    sat = scenario.sat
    jd0, fr0 = ss.jday(*scenario.start_utc)
    jd0_full = jd0 + fr0

    n_pts = 4000
    span_s = 2 * 93 * 60  # 2 个周期
    alts, speeds = [], []
    for i in range(n_pts):
        jd_full = jd0_full + span_s * i / (n_pts - 1) / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, v_eci = sat.sgp4(int(jd_i), fr_i)
        if err != 0:
            continue
        alt = np.linalg.norm(r_eci) - ss.EARTH_R_KM
        spd = np.linalg.norm(v_eci)
        alts.append(alt)
        speeds.append(spd)

    alts, speeds = np.array(alts), np.array(speeds)
    print(f"  高度: mean={alts.mean():.1f} km, min={alts.min():.1f}, max={alts.max():.1f}  (ISS 期望 ~400-450)")
    print(f"  速度: mean={speeds.mean():.3f} km/s  (ISS 期望 ~7.66)")

    # 周期：找高度峰值间距（近地点间隔）
    from scipy.signal import argrelextrema
    idx = argrelextrema(alts, np.less_equal, order=20)[0]
    if len(idx) >= 2:
        period = (idx[-1] - idx[-2]) * span_s / (n_pts - 1) / 60.0
        print(f"  周期: {period:.2f} min (由近地点间隔估计, ISS 期望 ~92.9)")

    # 轨道图（ECI 3D）
    r_eci_list = []
    for i in range(0, n_pts, 40):
        jd_full = jd0_full + span_s * i / (n_pts - 1) / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, _ = sat.sgp4(int(jd_i), fr_i)
        if err == 0:
            r_eci_list.append(r_eci)
    r = np.array(r_eci_list)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    xs = ss.EARTH_R_KM * np.sin(v) * np.cos(u)
    ys = ss.EARTH_R_KM * np.sin(v) * np.sin(u)
    zs = ss.EARTH_R_KM * np.cos(v)
    ax.plot_surface(xs, ys, zs, color="lightblue", alpha=0.35, rstride=1, cstride=1)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color="crimson", lw=1.8, label="Satellite orbit (ECI)")
    t = scenario.target_ecef
    ax.scatter(*t, color="green", s=60, label="Target (ground)")
    ax.scatter(*scenario.ground_ecef, color="darkorange", s=60, label="Ground station")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    ax.set_title("LEO Orbit (ECI) + Ground Geometry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "orbit_3d.png"), dpi=150)
    print(f"  [图] orbit_3d.png 已保存")


def verify_overpass(scenario):
    """过境窗口 + 完整窗口内 仰角/距离/多普勒曲线。"""
    print("=" * 70)
    print("[2] 过境窗口与多普勒曲线")
    windows = scenario.find_overpass()
    print(f"  过境窗口数: {len(windows)}")
    for i, (t0, t1) in enumerate(windows):
        print(f"    win{i}: [{t0/60:.1f}, {t1/60:.1f}] min, 时长 {(t1-t0)/60:.1f} min, 中心 {((t0+t1)/2)/60:.1f} min")

    if not windows:
        print("  无过境窗口，跳过曲线绘制。")
        return

    win = windows[0]
    jd0, fr0 = ss.jday(*scenario.start_utc)
    jd0_full = jd0 + fr0

    # 整个窗口 + 前后各 1 分钟，密采样
    t_span = np.linspace(win[0] - 60, win[1] + 60, 500)
    elevs, dists, fds = [], [], []
    for t in t_span:
        jd_full = jd0_full + t / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, v_eci = scenario.sat.sgp4(int(jd_i), fr_i)
        if err != 0:
            elevs.append(np.nan); dists.append(np.nan); fds.append(np.nan)
            continue
        r_ecef, v_ecef = ss.eci_to_ecef(r_eci, v_eci, jd_full)
        elevs.append(ss.elevation_deg(scenario.target_ecef, r_ecef))
        dists.append(np.linalg.norm(r_ecef - scenario.target_ecef))
        v_rel = ss.radial_velocity_mps(v_ecef, np.zeros(3), r_ecef, scenario.target_ecef)
        fds.append(v_rel / scenario.wavelength_m)

    t_min = (t_span - win[0]) / 60.0
    elevs, dists, fds = np.array(elevs), np.array(dists), np.array(fds)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t_min, elevs, color="green")
    axes[0].set_ylabel("Elevation (deg)"); axes[0].set_title("Satellite Overpass (window from TLE)")
    axes[0].axhline(ss.MIN_ELEVATION_DEG, ls="--", color="gray", alpha=0.7, label="min elev")
    axes[0].legend()
    axes[1].plot(t_min, dists / 1000.0, color="blue")
    axes[1].set_ylabel("Range (1000 km)")
    axes[2].plot(t_min, fds / 1e3, color="crimson")
    axes[2].set_ylabel("Doppler (kHz)")
    axes[2].set_xlabel("Time relative to window start (min)")
    axes[2].axhline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "overpass_dynamics.png"), dpi=150)
    print(f"  [图] overpass_dynamics.png 已保存")
    print(f"  窗口内多普勒范围: [{np.nanmin(fds)/1e3:.1f}, {np.nanmax(fds)/1e3:.1f}] kHz")

    return win


def verify_channel(scenario, frames):
    """信道与多普勒相位检查（公式级验证）。"""
    print("=" * 70)
    print("[3] 远场信道检查 (BS→ROI→UE)")
    f0 = frames[len(frames) // 2]
    # BS→ROI
    H1, d1, fd1, dl1 = scenario.get_channel_far(
        f0["sat_pos"], f0["target_pos"], f0["t_sec"],
        v_a_km_s=f0["sat_vel"], v_b_km_s=np.zeros(3))
    # ROI→UE
    H2, d2, fd2, dl2 = scenario.get_channel_far(
        f0["target_pos"], f0["ground_pos"], f0["t_sec"],
        v_a_km_s=np.zeros(3), v_b_km_s=np.zeros(3))
    print(f"  BS→ROI: d={d1:.2f} km, |H|={abs(H1):.3e}, f_d={fd1/1e3:.2f} kHz, delay={dl1*1e3:.2f} ms")
    print(f"  ROI→UE: d={d2:.2f} km, |H|={abs(H2):.3e}, f_d={fd2/1e3:.2f} kHz, delay={dl2*1e3:.2f} ms")

    # (a) 幅度：|H| == sqrt(0.1)/d
    amp_ok = abs(abs(H1) - np.sqrt(0.1) / (d1 * 1000)) < 1e-12
    amp_ok &= abs(abs(H2) - np.sqrt(0.1) / (d2 * 1000)) < 1e-12
    print(f"  幅度公式 sqrt(0.1)/d: {'PASS' if amp_ok else 'FAIL'}")

    # (b) 无多普勒时相位：2π d/λ (mod 2π)
    H_nd, *_ = scenario.get_channel_far(f0["sat_pos"], f0["target_pos"], 0.0)
    phase_expect = (2 * np.pi * d1 * 1000 / scenario.wavelength_m) % (2 * np.pi)
    phase_ok = abs((np.angle(H_nd) % (2 * np.pi)) - phase_expect) < 1e-6
    print(f"  距离相位 2πd/λ: {'PASS' if phase_ok else 'FAIL'}")

    # (c) 多普勒：微小 Δt（无相位混叠）下相位差 == 2π f_d Δt
    dt = 1e-7  # s，f_d*dt << 1
    H_a, _, fda, _ = scenario.get_channel_far(
        f0["sat_pos"], f0["target_pos"], 0.0,
        v_a_km_s=f0["sat_vel"], v_b_km_s=np.zeros(3))
    H_b, _, fdb, _ = scenario.get_channel_far(
        f0["sat_pos"], f0["target_pos"], dt,
        v_a_km_s=f0["sat_vel"], v_b_km_s=np.zeros(3))
    dphi = np.angle(H_b / H_a)  # 同一几何，距离相位抵消
    fd_est = dphi / (2 * np.pi * dt)
    doppler_ok = abs(fd_est - fda) / (abs(fda) + 1e-9) < 0.01 and abs(fda - fdb) < 1e-6
    print(f"  多普勒相位注入: est={fd_est/1e3:.2f} kHz vs 解析 {fda/1e3:.2f} kHz -> {'PASS' if doppler_ok else 'FAIL'}")

    ok = amp_ok and phase_ok and doppler_ok
    return ok


if __name__ == "__main__":
    scenario, frames = ss.build_default_scenario()
    print(f"载频 {scenario.fc_hz/1e9:.1f} GHz, λ={scenario.wavelength_m*100:.1f} cm")
    print(f"目标 ({scenario.target_lat}N,{scenario.target_lon}E)  地面站 ({scenario.ground_lat}N,{scenario.ground_lon}E)")
    verify_orbit(scenario)
    win = verify_overpass(scenario)
    ok_channel = verify_channel(scenario, frames)
    print("=" * 70)
    print("物理验证完成:", "ALL PASS" if ok_channel else "有 FAIL，请检查")
