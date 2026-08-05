"""
sdr_ingest.py — SDR IQ → ISAC 感知输入转换

把 SDR 采集的基带 IQ（时域）转成感知模型输入：
  - 宽带路径：时域 IQ → FFT（频域信道响应）→ 距离像（HRRP）
  - 窄带路径：时域 IQ → 相关解调 → 接收信号特征 → cond

模拟 IQ 由 sdr_io.simulate_isac_iq_wideband 生成（无硬件验证管线）；
真实硬件采集的 IQ 走完全相同路径。

管线保真验证：仿真直接算距离像 vs SDR 管线恢复距离像，应一致。
"""

import numpy as np
import torch

import setup_sat as ss
from data_sat import compute_range_profile, make_roi_local, WIDEBAND_K, WIDEBAND_BW_HZ


def iq_to_range_profile(iq, k=WIDEBAND_K, align=False, snr_db=None, seed=0):
    """时域 IQ → 距离像（HRRP）。

    物理：SDR 采集时域回波 → FFT 得频域响应 → IFFT 得距离像。
    iq: complex 数组（长度 = k）
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < k:
        iq = np.pad(iq, (0, k - len(iq)))
    H = np.fft.fft(iq[:k])                      # 频域响应
    if snr_db is not None:
        rng = np.random.RandomState(seed)
        sig_pow = np.mean(np.abs(H) ** 2)
        n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
    rp = np.abs(np.fft.ifft(H))                 # 距离像
    if align:
        idx = np.arange(k)
        total = rp.sum()
        if total > 1e-12:
            centroid = int(np.clip(np.sum(rp * idx) / total, 0, k - 1))
            rp = np.roll(rp, k // 2 - centroid)
    norm = np.linalg.norm(rp)
    return (rp / (norm + 1e-12)).astype(np.float32)


def iq_to_sensing_feature(iq, mode="wideband", align=False):
    """IQ → 感知模型输入特征（统一入口）。"""
    if mode == "wideband":
        return iq_to_range_profile(iq, align=align)
    raise ValueError(f"未支持的模式: {mode}")


def verify_pipeline(roi_np, target_ecef, ground_ecef, wavelength_m,
                    k=WIDEBAND_K, snr_db=20.0, seed=0):
    """保真验证：仿真距离像 vs SDR 管线恢复距离像。

    返回 (corr, sim_rp, sdr_rp)
    """
    # 仿真直接计算（参考真值）
    sim_rp = compute_range_profile(roi_np, target_ecef, ground_ecef, wavelength_m,
                                   k=k, snr_db=None, seed=seed, align=False)
    # 生成时域 IQ（无噪声，作为 SDR 采集）
    from sdr_io import simulate_isac_iq_wideband
    iq = simulate_isac_iq_wideband(roi_np, target_ecef, ground_ecef, wavelength_m,
                                   k=k, snr_db=None, seed=seed)
    # SDR 管线恢复
    sdr_rp = iq_to_range_profile(iq, k=k, align=False, snr_db=snr_db, seed=seed + 1)
    corr = float(np.corrcoef(sim_rp, sdr_rp)[0, 1])
    return corr, sim_rp, sdr_rp
