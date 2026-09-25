"""AFDM 波形（chirp 多载波；DAFT/IDAFT 链，高动态 LEO 候选）。

数学约定（对齐 Bemani–Ksairi–Kountouris, IEEE TWC 2022）：
  - IDAFT（发送）：s = Λ_{c1}^H F^H Λ_{c2}^H x，即
        s[n] = e^{j2πc1 n²} · (1/√N) Σ_m (x[m] e^{j2πc2 m²}) e^{j2π mn/N}
  - DAFT（接收）：y = Λ_{c2} F Λ_{c1} r（IDAFT 的精确逆，单位范数）。
  - 啁啾参数：c1 = (2α_max+1)/(2N)（使 2Nc1 为奇整数；N 偶时 CPP 退化为
    普通 CP，长度 ≥ 最大时延样本数即可把多径变成圆周移位），
    c2 = (2N+1)·c1（满足 c1 = c2/(2N+1) 型关系；c2 在 DAFT 域信道中
    只贡献逐子载波相位，不影响路径定位结构）。
  - DAFT 域信道（on-grid 整数归一化多普勒 ν_i = f_d/Δf）：每行单非零元
    （置换）q = (p + loc_i) mod N，loc_i = 2Nc1·l_i − ν_i（本实现多普勒
    相位取 e^{+j2πν} 约定）；分数 ν_i 按 Dirichlet 核扩散（banded）。
  - 多块帧：块间信道仅差已知的对角相位 e^{j2πν_i n_b}（CSI 已知可吸收）。

参考：A. Bemani, N. Ksairi, M. Kountouris, "AFDM: A Full Diversity Next
Generation Waveform for High Mobility Communications", IEEE TWC 2022。
"""

import numpy as np

from .base import Waveform
from .otfs import C_LIGHT, dirichlet_kernel


class AfdmWaveform(Waveform):
    """AFDM 波形：帧 = n_blocks 个 DAFT 块 × n_subcarriers 个 chirp 子载波。

    paths 统一表示为 (gain, delay_s, doppler_hz) 三元组列表；
    时延按 T_s = 1/B 量化到样本栅格（0 ≤ l_i < n_subcarriers）。
    """

    def __init__(self, n_blocks: int = 16, n_subcarriers: int = 512,
                 bandwidth_hz: float = 1e9, carrier_hz: float = 30e9,
                 alpha_max: int = 1):
        self.n_blocks = int(n_blocks)
        self.n_subcarriers = int(n_subcarriers)
        self.bandwidth_hz = float(bandwidth_hz)
        self.carrier_hz = float(carrier_hz)
        self.wavelength_m = C_LIGHT / float(carrier_hz)
        self.alpha_max = int(alpha_max)
        # 啁啾参数：c1 = (2α_max+1)/(2N)；c2 = (2N+1)c1 ⇒ c1 = c2/(2N+1)
        self.c1 = (2 * self.alpha_max + 1) / (2.0 * self.n_subcarriers)
        self.c2 = (2 * self.n_subcarriers + 1) * self.c1
        n = np.arange(self.n_subcarriers)
        self._pre = np.exp(2j * np.pi * self.c2 * n ** 2)    # Λ_{c2}^H
        self._post = np.exp(2j * np.pi * self.c1 * n ** 2)   # Λ_{c1}^H

    # ------------------------------------------------------------------
    # 网格物理量
    # ------------------------------------------------------------------
    @property
    def n_sub(self) -> int:
        return self.n_subcarriers

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.bandwidth_hz / self.n_subcarriers

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def block_duration_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def frame_duration_s(self) -> float:
        return self.n_blocks * self.block_duration_s

    @property
    def delay_resolution_s(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def two_n_c1(self) -> float:
        """2Nc1（应为奇整数 ⇒ CPP 退化为普通 CP）。"""
        return 2 * self.n_subcarriers * self.c1

    # ------------------------------------------------------------------
    # 变换链（单位范数）
    # ------------------------------------------------------------------
    def idaft(self, x_daft: np.ndarray) -> np.ndarray:
        """DAFT 域符号 [n_blocks, N] → 时间块 [n_blocks, N]。"""
        return (np.fft.ifft(np.asarray(x_daft) * self._pre, axis=1)
                * np.sqrt(self.n_subcarriers)) * self._post

    def daft(self, s_time: np.ndarray) -> np.ndarray:
        """时间块 [n_blocks, N] → DAFT 域 [n_blocks, N]。"""
        return (np.fft.fft(np.asarray(s_time) / self._post, axis=1)
                / np.sqrt(self.n_subcarriers)) * np.conj(self._pre)

    def modulate(self, x_daft: np.ndarray) -> np.ndarray:
        return self.idaft(x_daft).reshape(-1)

    def demodulate(self, r_time: np.ndarray) -> np.ndarray:
        r = np.asarray(r_time).reshape(self.n_blocks, self.n_subcarriers)
        return self.daft(r)

    # ------------------------------------------------------------------
    # 信道（CP 等价：每块循环时移 + 连续多普勒相位）
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_paths(paths):
        return [(complex(g), d, float(f)) for g, d, f in paths]

    def delay_samples(self, delay_s: float) -> int:
        return int(round(float(delay_s) * self.bandwidth_hz))

    def apply_channel(self, s_time: np.ndarray, paths) -> np.ndarray:
        s = np.asarray(s_time).reshape(self.n_blocks, self.n_subcarriers)
        n_idx = np.arange(self.n_blocks)[:, None]
        p_idx = np.arange(self.n_subcarriers)[None, :]
        t = n_idx * self.block_duration_s + p_idx * self.sample_period_s
        r = np.zeros_like(s)
        for gain, delay_s, fd_hz in self.normalize_paths(paths):
            l_i = self.delay_samples(delay_s)
            r += gain * np.roll(s, l_i, axis=1) * np.exp(2j * np.pi * fd_hz * t)
        return r.reshape(-1)

    def transmit_through(self, x_daft: np.ndarray, paths) -> np.ndarray:
        return self.demodulate(self.apply_channel(self.modulate(x_daft), paths))

    # ------------------------------------------------------------------
    # DAFT 域（仿射 DD）响应与均衡
    # ------------------------------------------------------------------
    def _nu(self, fd_hz: float) -> float:
        """归一化多普勒 ν = f_d/Δf。"""
        return float(fd_hz) / self.subcarrier_spacing_hz

    def loc_i(self, delay_s: float, fd_hz: float) -> float:
        """DAFT 域置换位置 loc_i = 2Nc1·l_i − ν_i（本实现的 e^{+j2πν} 约定）。"""
        return self.two_n_c1 * self.delay_samples(delay_s) - self._nu(fd_hz)

    def dd_response(self, delays_s, dopplers_hz, amplitudes,
                    block: int = 0) -> np.ndarray:
        """DAFT 域仿射 DD 冲激响应 [N]（块 block，默认 0）。

        对路径 i：R[p] = a_i·e^{j2πc1 l_i²}·D_N(ν_i − 2Nc1 l_i − p)/N
        （冲激输入 q=0；块相位 e^{j2πν_i·block} 单独乘上）。
        """
        delays_s = np.atleast_1d(delays_s)
        dopplers_hz = np.atleast_1d(dopplers_hz)
        amplitudes = np.atleast_1d(amplitudes)
        N = self.n_subcarriers
        p_ax = np.arange(N)
        rx_chirp = np.exp(-2j * np.pi * self.c2 * p_ax ** 2)  # 接收侧 Λ_{c2}
        R = np.zeros(N, dtype=complex)
        for tau, fd, a in zip(delays_s, dopplers_hz, amplitudes):
            l_i = self.delay_samples(tau)
            nu = self._nu(fd)
            R += (a * np.exp(2j * np.pi * self.c1 * l_i ** 2)
                  * np.exp(2j * np.pi * nu * block)
                  * rx_chirp
                  * dirichlet_kernel(nu - self.two_n_c1 * l_i - p_ax, N) / N)
        return R

    def channel_matrix(self, paths, block: int) -> np.ndarray:
        """块 block 的 DAFT 域信道矩阵 H_b [N, N]（稠密精确，含分数多普勒扩散）。"""
        N = self.n_subcarriers
        p_ax = np.arange(N)[:, None]
        q_ax = np.arange(N)[None, :]
        # 收发两侧 chirp：e^{j2πc2 q²}（发送 Λ_{c2}^H）× e^{-j2πc2 p²}（接收 Λ_{c2}）
        c2_phase = np.exp(2j * np.pi * self.c2 * (q_ax ** 2 - p_ax ** 2))
        H = np.zeros((N, N), dtype=complex)
        for gain, delay_s, fd_hz in self.normalize_paths(paths):
            l_i = self.delay_samples(delay_s)
            nu = self._nu(fd_hz)
            kern = (dirichlet_kernel(q_ax - p_ax + nu - self.two_n_c1 * l_i, N)
                    / N)
            H += (gain * np.exp(2j * np.pi * nu * block)
                  * np.exp(2j * np.pi * self.c1 * l_i ** 2)
                  * np.exp(-2j * np.pi * q_ax * l_i / N) * kern)
        return H * c2_phase

    def mmse_filters(self, paths, noise_var: float) -> list:
        """预计算各块 MMSE 滤波器 W_b = (H_b^H H_b + σ²I)^{-1} H_b^H。"""
        N = self.n_subcarriers
        eye = np.eye(N)
        filters = []
        for b in range(self.n_blocks):
            H = self.channel_matrix(paths, b)
            W = np.linalg.solve(H.conj().T @ H + noise_var * eye, H.conj().T)
            filters.append(W)
        return filters

    def mmse_equalize(self, y_daft: np.ndarray, paths, noise_var: float,
                      filters=None) -> np.ndarray:
        """DAFT 域逐块 MMSE 均衡（CSI 已知）。"""
        y = np.asarray(y_daft)
        if filters is None:
            filters = self.mmse_filters(paths, noise_var)
        out = np.empty_like(y)
        for b in range(self.n_blocks):
            out[b] = filters[b] @ y[b]
        return out

    # ------------------------------------------------------------------
    # Waveform 基类接口（频域距离像与 OFDM 同构）
    # ------------------------------------------------------------------
    def channel_response(self, delays_s, amplitudes) -> np.ndarray:
        k = np.arange(self.n_subcarriers)
        f = k * self.subcarrier_spacing_hz
        delays_s = np.atleast_1d(delays_s)
        amplitudes = np.atleast_1d(amplitudes)
        phase = np.exp(-1j * 2.0 * np.pi * f[None, :] * delays_s[:, None])
        return (amplitudes[:, None] * phase).sum(axis=0)

    def range_profile(self, h_freq: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.abs(np.fft.ifft(h_freq)))

    @property
    def range_resolution_m(self) -> float:
        return C_LIGHT / (2.0 * self.bandwidth_hz)
