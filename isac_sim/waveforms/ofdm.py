"""OFDM 感知波形（默认实现，smoke 级）。

K 个子载波、带宽 B、子载波间隔 B/K；频率栅格取基带 [0, B)（与 isac_sat
compute_range_profile 的差分时延约定兼容：时延 tau 相对参考点）。
"""

import numpy as np

from .base import Waveform


class OfdmWaveform(Waveform):
    def __init__(self, n_subcarriers: int = 512, bandwidth_hz: float = 1e9,
                 carrier_hz: float = 30e9):
        self.n_sub = int(n_subcarriers)
        self.bandwidth_hz = float(bandwidth_hz)
        self.carrier_hz = float(carrier_hz)

    def channel_response(self, delays_s: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        k = np.arange(self.n_sub)
        f = k * (self.bandwidth_hz / self.n_sub)
        delays_s = np.atleast_1d(delays_s)
        amplitudes = np.atleast_1d(amplitudes)
        # [n_scatter, n_sub] 求和 → [n_sub]
        phase = np.exp(-1j * 2.0 * np.pi * f[None, :] * delays_s[:, None])
        return (amplitudes[:, None] * phase).sum(axis=0)

    def range_profile(self, h_freq: np.ndarray) -> np.ndarray:
        rp = np.fft.ifft(h_freq)
        return np.fft.fftshift(np.abs(rp))

    @property
    def range_resolution_m(self) -> float:
        return 299_792_458.0 / (2.0 * self.bandwidth_hz)
