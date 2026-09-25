"""OTFS 感知/通信波形（高动态 NTN 候选；Zak 变换视角）。

数学约定（对齐 Raviteja–Phan–Hong / Surabhi–Augustine–Chockalingam）：
  - OTFS 帧 = N 个 OFDM 符号 × M 个子载波；DD 网格 x[k, l]（形状 [N, M]）：
      k = 0..N-1 多普勒 bin（间隔 1/T_frame = Δf/N），
      l = 0..M-1 时延 bin（间隔 1/B = 1/(MΔf) = T_s）。
  - ISFFT（DD→TF）：X = fft(ifft(x, axis=0)·√N, axis=1)/√M
    （沿多普勒轴 IDFT + 沿时延轴 DFT，单位范数）。
  - Heisenberg（TF→时间）：每符号沿子载波轴 IDFT（矩形脉冲 ⇒ 与 OFDM 同构）。
  - Wigner（时间→TF）：每符号沿子载波轴 FFT。
  - SFFT（TF→DD）：y = fft(ifft(Y, axis=1)·√M, axis=0)/√N。
  - 信道（CP 等价模型）：每符号循环时移 l_i（等价于长度 ≥ 最大时延的 CP）
    + 连续多普勒相位 e^{j2π f_d t}，t = n·T_sym + p·T_s 为帧内时间。
  - DD 域输入输出（单径，on-grid 整数 (k_i, l_i)）：
      y[k,l] = h·e^{j2π l k_i/(MN)}·x[(k-k_i) mod N, (l-l_i) mod M]
    即二维扭曲卷积（Zak 变换视角；Hadani Prop.1 的离散版）。
    分数多普勒 k_i 时沿多普勒轴按 Dirichlet 核扩散，泄漏能谱
    ≈ 1 − sinc²(δ)（δ 为分数 bin 偏移），与 OFDM 单音 ICI 恒等式同源。

参考：R. Hadani et al., WCNC 2017；Raviteja et al., IEEE TCOM 2019。
"""

import numpy as np

from .base import Waveform

C_LIGHT = 299_792_458.0


def dirichlet_kernel(delta: np.ndarray, n: int) -> np.ndarray:
    """D_n(Δ) = Σ_{t=0}^{n-1} e^{j2πtΔ/n}（Dirichlet 核，OFDM ICI 的精确核）。

    整数 Δ≠0 时精确为 0，Δ=0 时为 n；非整数用闭式
    e^{jπΔ(n-1)/n}·sin(πΔ)/sin(πΔ/n)。
    """
    delta = np.asarray(delta, dtype=float)
    out = np.empty(delta.shape, dtype=complex)
    near_zero = np.abs(delta - np.round(delta)) < 1e-12
    if np.any(near_zero):
        r = np.round(delta[near_zero])
        # 整数 Δ：D_n(Δ) = n 当 Δ ≡ 0 (mod n)，否则 0（Dirichlet 核以 n 为周期）
        out[near_zero] = np.where(np.mod(r, n) == 0, float(n), 0.0)
    rest = ~near_zero
    if np.any(rest):
        d = delta[rest]
        out[rest] = (np.exp(1j * np.pi * d * (n - 1) / n)
                     * np.sin(np.pi * d) / np.sin(np.pi * d / n))
    return out


class OtfsWaveform(Waveform):
    """OTFS 波形：DD 网格 [n_symbols, n_subcarriers] = [多普勒, 时延]。

    paths 统一表示为 (gain, delay_s, doppler_hz) 三元组列表；
    时延按 T_s = 1/B 量化到样本栅格（要求 0 ≤ l_i < n_subcarriers，
    由 CP 等价模型保证无帧间 ISI）。
    """

    def __init__(self, n_symbols: int = 16, n_subcarriers: int = 512,
                 bandwidth_hz: float = 1e9, carrier_hz: float = 30e9):
        self.n_symbols = int(n_symbols)          # N：多普勒 bin 数（= OFDM 符号数）
        self.n_subcarriers = int(n_subcarriers)  # M：时延 bin 数（= 子载波数）
        self.bandwidth_hz = float(bandwidth_hz)
        self.carrier_hz = float(carrier_hz)
        self.wavelength_m = C_LIGHT / float(carrier_hz)

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
    def symbol_duration_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def frame_duration_s(self) -> float:
        return self.n_symbols * self.symbol_duration_s

    @property
    def doppler_resolution_hz(self) -> float:
        return 1.0 / self.frame_duration_s

    @property
    def delay_resolution_s(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def doppler_axis_hz(self) -> np.ndarray:
        """DD 多普勒轴（Hz，k=0 为 0，k≥N/2 为负频率，圆周约定）。"""
        k = np.arange(self.n_symbols)
        return np.where(k < self.n_symbols / 2, k,
                        k - self.n_symbols) * self.doppler_resolution_hz

    @property
    def delay_axis_s(self) -> np.ndarray:
        return np.arange(self.n_subcarriers) * self.delay_resolution_s

    # ------------------------------------------------------------------
    # 变换链（单位范数）
    # ------------------------------------------------------------------
    def isfft(self, x_dd: np.ndarray) -> np.ndarray:
        """DD → TF。x_dd: [N, M]（多普勒, 时延）→ X: [N, M]（符号, 子载波）。"""
        x_dd = np.asarray(x_dd)
        return np.fft.fft(np.fft.ifft(x_dd, axis=0) * np.sqrt(self.n_symbols),
                          axis=1) / np.sqrt(self.n_subcarriers)

    def heisenberg(self, x_tf: np.ndarray) -> np.ndarray:
        """TF → 时间（每符号子载波 IDFT）。返回 [N, M]（符号, 样本）。"""
        return np.fft.ifft(np.asarray(x_tf), axis=1) * np.sqrt(self.n_subcarriers)

    def wigner(self, s_time: np.ndarray) -> np.ndarray:
        """时间 → TF（每符号子载波 FFT）。"""
        return np.fft.fft(np.asarray(s_time), axis=1) / np.sqrt(self.n_subcarriers)

    def sfft(self, y_tf: np.ndarray) -> np.ndarray:
        """TF → DD。"""
        return np.fft.fft(
            np.fft.ifft(np.asarray(y_tf), axis=1) * np.sqrt(self.n_subcarriers),
            axis=0) / np.sqrt(self.n_symbols)

    # ------------------------------------------------------------------
    # 发送/接收
    # ------------------------------------------------------------------
    def modulate(self, x_dd: np.ndarray) -> np.ndarray:
        """DD 符号 [N, M] → 时间样本 [N*M]（符号优先串行）。"""
        s = self.heisenberg(self.isfft(x_dd))
        return s.reshape(-1)

    def demodulate(self, r_time: np.ndarray) -> np.ndarray:
        """时间样本 [N*M] → DD 接收 [N, M]。"""
        r = np.asarray(r_time).reshape(self.n_symbols, self.n_subcarriers)
        return self.sfft(self.wigner(r))

    # ------------------------------------------------------------------
    # 信道
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_paths(paths):
        """paths → [(gain, delay_samples, doppler_hz), ...]（时延量化到样本）。"""
        out = []
        for p in paths:
            gain, delay_s, fd_hz = p
            out.append((complex(gain), delay_s, float(fd_hz)))
        return out

    def delay_samples(self, delay_s: float) -> int:
        return int(round(float(delay_s) * self.bandwidth_hz))

    def doppler_bin(self, fd_hz: float) -> float:
        """多普勒的 DD bin 坐标（可分数）。"""
        return float(fd_hz) * self.frame_duration_s

    def apply_channel(self, s_time: np.ndarray, paths) -> np.ndarray:
        """CP 等价信道：每符号循环时移 + 连续多普勒相位。"""
        s = np.asarray(s_time).reshape(self.n_symbols, self.n_subcarriers)
        n_idx = np.arange(self.n_symbols)[:, None]
        p_idx = np.arange(self.n_subcarriers)[None, :]
        t = n_idx * self.symbol_duration_s + p_idx * self.sample_period_s
        r = np.zeros_like(s)
        for gain, delay_s, fd_hz in self.normalize_paths(paths):
            l_i = self.delay_samples(delay_s)
            r += gain * np.roll(s, l_i, axis=1) * np.exp(2j * np.pi * fd_hz * t)
        return r.reshape(-1)

    def transmit_through(self, x_dd: np.ndarray, paths) -> np.ndarray:
        """DD 输入 → 信道 → DD 输出（完整链，用于协议验证）。"""
        return self.demodulate(self.apply_channel(self.modulate(x_dd), paths))

    # ------------------------------------------------------------------
    # DD 域响应（感知）
    # ------------------------------------------------------------------
    def dd_response(self, delays_s, dopplers_hz, amplitudes) -> np.ndarray:
        """DD 域冲激响应（Zak 视角）：[N, M] = [多普勒, 时延]。

        对位于 (k_i, l_i) 的路径 i（l_i 量化到样本栅格，k_i = f_d·T_frame 可分数）：
            R[k, l] = a_i·e^{j2π l k_i/(MN)}·D_N(k_i - k)/N·δ[l = l_i]
        分数 k_i 的 Dirichlet 扩散泄漏能谱 ≈ 1 − sinc²(δ)。
        """
        delays_s = np.atleast_1d(delays_s)
        dopplers_hz = np.atleast_1d(dopplers_hz)
        amplitudes = np.atleast_1d(amplitudes)
        N, M = self.n_symbols, self.n_subcarriers
        k_ax = np.arange(N)
        l_ax = np.arange(M)
        R = np.zeros((N, M), dtype=complex)
        for tau, fd, a in zip(delays_s, dopplers_hz, amplitudes):
            l_i = self.delay_samples(tau)
            if not (0 <= l_i < M):
                raise ValueError(f"时延 {tau*1e9:.1f} ns 超出 DD 时延轴 "
                                 f"[0, {M/self.bandwidth_hz*1e9:.0f}) ns")
            k_i = self.doppler_bin(fd)
            phase = np.exp(2j * np.pi * l_i * k_i / (M * N))
            kern = dirichlet_kernel(k_i - k_ax, N) / N
            R[:, l_i] += a * phase * kern
        return R

    def dd_channel_matvec(self, v: np.ndarray, paths) -> np.ndarray:
        """H @ v（DD 域信道矩阵作用于向量/网格，免建稠密矩阵）。"""
        v = np.asarray(v).reshape(self.n_symbols, self.n_subcarriers)
        N, M = self.n_symbols, self.n_subcarriers
        k_ax = np.arange(N)
        l_ax = np.arange(M)
        out = np.zeros_like(v)
        for gain, delay_s, fd_hz in self.normalize_paths(paths):
            l_i = self.delay_samples(delay_s)
            k_i = self.doppler_bin(fd_hz)
            u = np.roll(v, l_i, axis=1)
            u = u * np.exp(2j * np.pi * l_ax[None, :] * k_i / (M * N))
            K = dirichlet_kernel(k_i + k_ax[None, :] - k_ax[:, None], N) / N
            out += gain * (K @ u)
        return out

    def dd_channel_rmatvec(self, v: np.ndarray, paths) -> np.ndarray:
        """H^H @ v（H 的精确伴随：<Hv,w> = <v,H^Hw>）。"""
        v = np.asarray(v).reshape(self.n_symbols, self.n_subcarriers)
        N, M = self.n_symbols, self.n_subcarriers
        k_ax = np.arange(N)
        l_ax = np.arange(M)
        out = np.zeros_like(v)
        for gain, delay_s, fd_hz in self.normalize_paths(paths):
            l_i = self.delay_samples(delay_s)
            k_i = self.doppler_bin(fd_hz)
            u = np.roll(v, -l_i, axis=1)
            l_ph = (l_ax[None, :] + l_i) % M
            u = u * np.exp(-2j * np.pi * l_ph * k_i / (M * N))
            K = dirichlet_kernel(k_i + k_ax[None, :] - k_ax[:, None], N) / N
            out += np.conj(gain) * (K.conj().T @ u)
        return out

    def mmse_equalize(self, y_dd: np.ndarray, paths, noise_var: float,
                      n_iter: int = 80, tol: float = 1e-10) -> np.ndarray:
        """DD 域 MMSE 均衡（CSI 已知，共轭梯度解 (H^HH+σ²I)x = H^Hy）。"""
        y = np.asarray(y_dd)
        b = self.dd_channel_rmatvec(y, paths)
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rs = float(np.vdot(r, r).real)
        if rs == 0.0:
            return x
        for _ in range(n_iter):
            Hp = self.dd_channel_matvec(p, paths)
            Ap = self.dd_channel_rmatvec(Hp, paths) + noise_var * p
            denom = float(np.vdot(p, Ap).real)
            if denom <= 0:
                break
            alpha = rs / denom
            x += alpha * p
            r -= alpha * Ap
            rs_new = float(np.vdot(r, r).real)
            if rs_new < tol * rs:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

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
