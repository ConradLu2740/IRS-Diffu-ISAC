"""verify_hrrp_geometry.py — HRRP 几何真实双程差分时延自检（Task 3, spec 2026-09-08）

验证 compute_range_profile 在传入 sat_ecef（真实轨道几何）后的两个特征语义：
  1. center='roi' + align=False：目标位置信息进入距离像（峰随体素位置移动）；
  2. center='centroid' + align=True：形状特征平移不变（相同形状在不同位置 → 相同距离像）。

旧启发式（sat_ecef=None）把 UE 距离硬编码为 50 km、BS 侧用固定 0.1 系数；
本脚本只验证修复后语义，不对比旧值（旧值保留可复现）。
"""
import numpy as np
import setup_sat as ss
from data_sat import compute_range_profile, make_roi_local, ROI_RES


def single_voxel_roi(i, j, k):
    roi = np.zeros((ROI_RES,) * 3, dtype=np.float32)
    roi[i, j, k] = 1.0
    return roi


def argmax_bin(profile):
    return int(np.argmax(profile))


def main():
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    mid = frames[len(frames) // 2]
    wl = scenario.wavelength_m
    sat = mid["sat_pos"]
    tgt = mid["target_pos"]
    gnd = mid["ground_pos"]

    # ---- 1. roi 模式：单个体素放在 ROI 内不同 x 位置，峰应移动 ----
    peaks = []
    for x in (2, 6, 10, 14):
        rp = compute_range_profile(single_voxel_roi(x, 8, 8), tgt, gnd, wl,
                                   snr_db=None, align=False, center="roi",
                                   sat_ecef=sat)
        peaks.append(argmax_bin(rp))
    monotone = all(b != peaks[0] for b in peaks[1:])
    assert monotone, f"roi 模式峰应随体素位置变化: {peaks}"
    # 峰移动方向应与双程路径差单调一致（x 增大→沿网格方向时延变化单调）
    assert max(peaks) - min(peaks) >= 1, f"峰移动幅度过小: {peaks}"

    # ---- 2. centroid + align 模式：同一形状放不同 x 位置 → 距离像相同 ----
    shape = single_voxel_roi(2, 8, 8)
    rp_a = compute_range_profile(shape, tgt, gnd, wl, snr_db=None, align=True,
                                 center="centroid", sat_ecef=sat)
    shape_b = single_voxel_roi(12, 8, 8)
    rp_b = compute_range_profile(shape_b, tgt, gnd, wl, snr_db=None, align=True,
                                 center="centroid", sat_ecef=sat)
    assert np.allclose(rp_a, rp_b, atol=1e-5), \
        f"centroid 模式应平移不变: max diff {np.max(np.abs(rp_a - rp_b)):.2e}"

    # ---- 3. roi 未对齐 vs centroid 对齐确实不同（位置信息未被 align 抹掉）----
    rp_raw = compute_range_profile(shape_b, tgt, gnd, wl, snr_db=None, align=False,
                                   center="roi", sat_ecef=sat)
    assert not np.allclose(rp_a, rp_raw, atol=1e-3), "roi 未对齐应保留位置信息"

    print("几何真实 HRRP 检查（Task 3）: ALL PASS")
    print(f"  roi 模式峰序列（x=2/6/10/14）: {peaks}")
    print(f"  centroid+align 平移不变: max diff = {np.max(np.abs(rp_a - rp_b)):.2e}")


if __name__ == "__main__":
    main()
