"""isac_sim 分层骨架冒烟测试。

运行：python tests/test_smoke_isac_sim.py  （或 pytest tests/）
原则：小样本、秒级、numpy-only；每层一个物理 sanity 断言。
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isac_sim.channels import FreeSpaceChannel, RicianChannel
from isac_sim.waveforms import OfdmWaveform
from isac_sim.ris import (BinaryPhaseRis, ContinuousPhaseRis, RisModel,
                          SegmentedTracking, coherent_power)
from isac_sim.comm import QpskAwgnLink
from isac_sim.sensing import CA_Cfar1D
from isac_sim.tracking import NearestNeighborTracker
from isac_sim.findings.far_field_angle_wall import (default_scenario_report,
    trilateration_cross_range_error, scan_shortfall)


def test_channel_free_space():
    """L0：1 km @30 GHz 幅度 = sqrt(0.1)/1000，相位 = 2πd/λ。"""
    ch = FreeSpaceChannel(carrier_hz=30e9)
    g = ch.link_gain(np.array([1000.0]))[0]
    assert np.isclose(np.abs(g), np.sqrt(0.1) / 1000.0, rtol=1e-12)
    assert np.isclose(np.angle(g) % (2 * np.pi), (2 * np.pi * 1000.0 / ch.wavelength_m) % (2 * np.pi), atol=1e-9)


def test_channel_rician_power_alignment():
    """L1：K=10 dB 莱斯的平均功率与自由空间对齐（蒙特卡洛 ~1% 精度）。"""
    fs = FreeSpaceChannel(carrier_hz=30e9)
    d = np.array([700e3])
    p_fs = np.abs(fs.link_gain(d)) ** 2
    powers = [abs(RicianChannel(30e9, k_factor_db=10.0, n_scatter=256, seed=s).link_gain(d)) ** 2
              for s in range(64)]
    p_mean = float(np.mean(powers))
    assert abs(p_mean / p_fs - 1.0) < 0.05, f"Rician mean power off: {p_mean / p_fs}"


def test_waveform_ofdm_range_profile():
    """波形：双散射体（双程时延 2d/c）→ 距离像峰出现在对应距离处。"""
    wf = OfdmWaveform(n_subcarriers=512, bandwidth_hz=1e9)
    c = 299_792_458.0
    d1, d2 = 10.0, 30.0  # 单程距离（米）；无模糊窗 = N*c/(2B) = 76.8 m，安全
    h = wf.channel_response(2 * np.array([d1, d2]) / c, np.array([1.0, 0.8]))
    rp = wf.range_profile(h)
    axis_m = np.fft.fftshift(np.fft.fftfreq(512, d=1e9 / 512)) * c / 2.0
    top2 = axis_m[np.argsort(rp)[-2:]]
    for d_true in (d1, d2):
        assert np.min(np.abs(top2 - d_true)) < 2.0 * wf.range_resolution_m, \
            (d_true, top2, wf.range_resolution_m)
    # 频率轴一致性：fftfreq 的采样间隔必须是子载波间隔 Δf = B/N
    assert np.isclose(wf.range_resolution_m, 299_792_458.0 / (2 * 1e9))


def test_ris_phase_alignment():
    """RIS：连续对齐 > 1-bit 量化 > 随机相位（相干合成功率单调）。"""
    rng = np.random.default_rng(42)
    g = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    p_cont = coherent_power(g, ContinuousPhaseRis().configure(g))
    p_bin = coherent_power(g, BinaryPhaseRis().configure(g))
    p_rand = coherent_power(g, rng.uniform(0, 2 * np.pi, 32))
    assert p_cont >= 1.5 * p_bin, (p_cont, p_bin)   # 对齐应优于 1-bit 量化
    assert p_bin >= p_rand, (p_bin, p_rand)
    # 分段重构：逐元素相位漂移下，K=1 逐帧最优；K=4 沿用旧相位应更差
    drift = np.exp(1j * 0.05 * np.arange(32))  # 每帧逐元素固定漂移向量
    gains = [g * drift**t for t in range(8)]
    r_k1 = SegmentedTracking(ContinuousPhaseRis(), 1).run(gains)
    r_k4 = SegmentedTracking(ContinuousPhaseRis(), 4).run(gains)
    assert r_k1["powers"].sum() > r_k4["powers"].sum(), \
        (r_k1["powers"].sum(), r_k4["powers"].sum())


def test_comm_qpsk_ber():
    """通信：QPSK over AWGN，实测 BER 与理论一致（±40% 相对误差）。"""
    link = QpskAwgnLink(seed=42)
    res = link.run_awgn(n_bits=40_000, snr_db=8.0)
    assert abs(res["ber"] - res["ber_theory"]) < 0.4 * res["ber_theory"], res


def test_sensing_cfar():
    """感知：强噪底 + 两个峰 → CA-CFAR 恰检出 2 个峰。"""
    rng = np.random.default_rng(42)
    noise = rng.exponential(1.0, 512)  # 单元功率 ~ 指数分布（复高斯功率）
    x = noise.copy()
    x[100] += 50.0
    x[300] += 30.0
    det = CA_Cfar1D(n_train=16, n_guard=4, p_fa=1e-4).detect_peaks(x)
    assert set(det) == {100, 300}, det


def test_tracking_nn():
    """跟踪：单目标 1 m/帧匀速运动 → 单轨迹锁定，终点位置误差 < 0.5 m。"""
    tr = NearestNeighborTracker(gate_m=5.0)
    true_pos = np.array([0.0, 0.0, 0.0])
    for t in range(10):
        true_pos = true_pos + np.array([1.0, 0.0, 0.0])
        out = tr.step(true_pos[None, :] + 0.1 * np.random.default_rng(t).standard_normal(3))
    assert len(tr.tracks) == 1
    est = tr.tracks[list(tr.tracks)[0]]["pos"]
    assert np.linalg.norm(est - true_pos) < 0.5, (est, true_pos)


def test_finding_angle_wall():
    """发现2：默认星-地场景下，ULA 分辨率比 ROI 张角差 > 3 个量级。"""
    rep = default_scenario_report()
    assert rep["shortfall_ratio"] > 1000.0, rep
    assert rep["required_aperture_m"] > 10.0, rep  # 需要数十米级孔径


def test_finding_two_station_escape():
    """发现2反例：γ 基于真实星-地几何（131°）时三边定位远破单站墙；
    误差随 γ 减小而增大；Δaz=0（秩亏）时应返回病态大误差提示。"""
    rho = 299_792_458.0 / (2 * 1e9)          # 1 GHz 带宽 → 0.15 m
    err = trilateration_cross_range_error(rho, math.radians(131.1))
    assert 0.1 < err < 0.5, err              # 默认几何一阶理论 ≈ 0.20 m
    assert err < trilateration_cross_range_error(rho, math.radians(34.0))  # γ 越小越差
    import math as _m
    assert trilateration_cross_range_error(rho, 1e-12) == float("inf")  # 秩亏 → inf
    # 扫描形状与单调性：N 越大 shortfall 越小
    sc = scan_shortfall([8, 64], [695e3], 80.0, 299_792_458.0 / 30e9)
    assert sc[0, 0] > sc[1, 0] > 1.0, sc


TESTS = [test_channel_free_space, test_channel_rician_power_alignment,
         test_waveform_ofdm_range_profile, test_ris_phase_alignment,
         test_comm_qpsk_ber, test_sensing_cfar, test_tracking_nn,
         test_finding_angle_wall, test_finding_two_station_escape]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        sys.exit(f"{failed} test(s) failed")
    print(f"ALL PASS ({len(TESTS)} layer smoke tests)")
