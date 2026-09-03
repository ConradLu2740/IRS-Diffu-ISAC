"""L2：Sionna 2.x（PyTorch 后端）CDL 标准信道对照（3GPP TR 38.901）。

目的：为 L0/L1 自建信道（FreeSpace / Rician，平坦衰落单抽头）提供
3GPP 标准簇时延线（CDL）对照，量化"平坦衰落 + 逐帧独立"两个近似
在 30 GHz 星-地 ISAC 场景下的适用边界。

建模代理说明（诚实边界）：
  - Sionna 内置 TR 38.901 CDL-A..E 与 TDL，未内置 TR 38.811 NTN 专用
    剖面。本层用 CDL-D（含显式 LOS 路径）作标准信道代理：
    LOS 结构与星-地链路一致，时延扩展/多普勒由参数显式给定。
  - 与 L0/L1 的关系：CDL-D 的 K 因子由剖面固定（约 9 dB 量级，见
    empirical_k_factor），恰好落在 P1 K-sweep 覆盖的档位附近。

依赖：sionna>=2.0（PyTorch 后端）。懒加载——未安装 sionna 时仅在
实例化时报错，不影响 isac_sim 其余部分。安装：pip install sionna。

注意：Sionna 在有 GPU 的机器上默认 device='cuda:0'；本类默认 'cpu'
以保持与 isac_sim 其余部分一致的确定性/轻量行为。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SionnaCdlChannel"]


class SionnaCdlChannel:
    """3GPP CDL 标准信道对照（Sionna 2.x PyTorch 后端）。

    参数
    ----
    carrier_hz : 载频（Hz），默认 30 GHz（与 setup_sat.FC_HZ 一致）。
    model : "A".."E"，默认 "D"（CDL-D，含 LOS 路径）。
    delay_spread_s : RMS 时延扩展（s），默认 100 ns（NTN 陆地移动档量级）。
    ut_speed_mps : UT 速度（m/s）。设为 LEO 相对速度（如 ~7000）可复现
        星-地多普勒；None 则用 Sionna 默认（静止）。
    seed : 随机种子（CDL 内部射线耦合/相位）。
    device : "cpu"（默认）或 "cuda:0"。
    """

    def __init__(self, carrier_hz: float = 30e9, model: str = "D",
                 delay_spread_s: float = 100e-9, ut_speed_mps: float | None = None,
                 seed: int = 42, device: str = "cpu"):
        try:
            from sionna.phy.channel.tr38901 import CDL, AntennaArray
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SionnaCdlChannel 需要 sionna>=2.0（pip install sionna）") from e

        self.carrier_hz = float(carrier_hz)
        self.model = str(model)
        self.delay_spread_s = float(delay_spread_s)
        self.ut_speed_mps = ut_speed_mps
        self.seed = int(seed)
        self.device = str(device)

        self._ant = AntennaArray(num_rows=1, num_cols=1, polarization="single",
                                 polarization_type="V", antenna_pattern="38.901",
                                 carrier_frequency=self.carrier_hz,
                                 device=self.device)
        self._cdl = CDL(model=self.model, delay_spread=self.delay_spread_s,
                        carrier_frequency=self.carrier_hz,
                        ut_array=self._ant, bs_array=self._ant,
                        device=self.device)
        if ut_speed_mps is not None:
            self._cdl._min_speed = float(ut_speed_mps)
            self._cdl._max_speed = float(ut_speed_mps)
        torch = __import__("torch")
        torch.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # 剖面属性（来自 3GPP 标准表）
    # ------------------------------------------------------------------
    @property
    def profile_delays_s(self) -> np.ndarray:
        """NLOS 簇时延（s，LOS 单独抽取，不含 LOS）。"""
        return self._cdl._delays.numpy().ravel().copy()

    @property
    def profile_powers_linear(self) -> np.ndarray:
        """NLOS 簇线性功率（已归一化，和为 1；LOS 由 _k_factor 单独给出）。"""
        p = self._cdl._powers
        arr = np.asarray(p.detach().cpu().numpy()).ravel()
        return arr / arr.sum()

    @property
    def k_factor_linear(self) -> float:
        """标准剖面 K 因子（线性）：LOS 功率 / NLOS 总功率。"""
        kf = self._cdl._k_factor
        return float(kf.detach().cpu().numpy().ravel()[0])

    @property
    def has_los(self) -> bool:
        return bool(self._cdl._has_los)

    def empirical_k_factor_db(self) -> float:
        """CDL 剖面的 K 因子（dB）。NLOS 剖面返回 -inf。"""
        if not self.has_los:
            return float("-inf")
        return 10.0 * np.log10(self.k_factor_linear)

    # ------------------------------------------------------------------
    # CIR 生成
    # ------------------------------------------------------------------
    def cir(self, batch_size: int = 1, num_time_steps: int = 1,
            sampling_frequency: float = 1.0):
        """生成标准 CDL CIR 系数 (a, tau)。

        返回
        ----
        a   : [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time] 复张量
        tau : [batch, num_rx, num_tx, num_paths] 时延（s）
        """
        a, tau = self._cdl(batch_size=batch_size,
                           num_time_steps=num_time_steps,
                           sampling_frequency=float(sampling_frequency))
        return a, tau

    def flat_gain_series(self, batch_size: int = 1, num_time_steps: int = 1,
                         sampling_frequency: float = 1.0,
                         include_los: bool = True) -> np.ndarray:
        """载频处的平坦复增益时间序列 h(t) = Σ_l a_l(t)。

        include_los=False 时只累加 NLOS 散射路径（LOS 单独抽取后
        路径索引 0 不参与）。

        返回 [batch, num_time] 复数数组（单收单发）。
        """
        a, _ = self.cir(batch_size, num_time_steps, sampling_frequency)
        paths = a[:, 0, 0, 0, 0, :, :]
        if not include_los and self.has_los:
            paths = paths[..., 1:, :]
        h = paths.sum(axis=-2)  # [batch, num_time]
        return h.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 对照指标 1：频率选择性（平坦衰落假设检验）
    # ------------------------------------------------------------------
    def frequency_correlation(self, max_delay_hz: float = 2e9,
                              n_points: int = 4096) -> dict:
        """功率时延谱驱动的频率相关函数 |ρ(Δf)|。

        ρ(Δf) = |Σ_l P_l exp(-j 2π Δf τ_l)| / Σ_l P_l

        返回 dict(freqs_hz, rho, coherence_bw_05_hz, coherence_bw_09_hz)。
        相干带宽按 |ρ| 降到 0.5 / 0.9 的最低 Δf 计。

        LOS 路径时延为 0（normalize_delays），贡献常数项 K/(K+1)。
        """
        tau = self.profile_delays_s
        p = self.profile_powers_linear
        df = np.linspace(0.0, float(max_delay_hz), int(n_points))
        if self.has_los:
            k = self.k_factor_linear
            ph = np.exp(-1j * 2.0 * np.pi * tau[:, None] * df[None, :])  # [簇, 频点]
            nlos = np.abs((ph * p[:, None]).sum(axis=0))  # [n_points]
            rho = (k / (1.0 + k) + nlos / (1.0 + k))
        else:
            ph = np.exp(-1j * 2.0 * np.pi * tau[:, None] * df[None, :])  # [簇, 频点]
            rho = np.abs((ph * p[:, None]).sum(axis=0)) / p.sum()
        rho = np.abs(rho)
        def _bw(level: float) -> float:
            idx = np.argmax(rho < level)
            return float(df[idx]) if rho[idx] < level else float("nan")
        return {"freqs_hz": df, "rho": rho,
                "coherence_bw_05_hz": _bw(0.5),
                "coherence_bw_09_hz": _bw(0.9)}

    # ------------------------------------------------------------------
    # 对照指标 2：时间选择性（逐帧独立近似检验）
    # ------------------------------------------------------------------
    def temporal_autocorrelation(self, sampling_frequency: float = 2e6,
                                 num_time_steps: int = 4096,
                                 batch_size: int = 8) -> dict:
        """从 Sionna CIR 时间序列估计归一化自相关 |ρ(Δt)|。

        物理分解（TR 38.901 / Jakes）：
          ρ(Δt) = [K·ρ_LOS + ρ_sc(Δt)] / (K+1)
          |ρ_LOS| = 1（确定性相位旋转，速率为 LOS 多普勒），
          ρ_sc(Δt) ≈ J0(2π f_d Δt)（各向同性散射）。
        因此全信道 |ρ(Δt)| 存在地板 K/(K+1)；去相关时间只在
        NLOS 散射分量上定义（nlos 曲线）。

        返回 dict(lags_s, rho_full, rho_nlos, los_floor,
                  decorr_time_05_s, decorr_time_09_s)。
        """
        def _ac(h: np.ndarray) -> tuple:
            n = h.shape[1]
            spec = np.fft.fft(h, axis=1)
            ac = np.fft.ifft(np.abs(spec) ** 2, axis=1)[:, :n].real
            lags = np.arange(n) / float(sampling_frequency)
            return lags, np.abs(ac / ac[:, :1]).mean(axis=0)

        kw = dict(batch_size=batch_size, num_time_steps=num_time_steps,
                  sampling_frequency=sampling_frequency)
        lags, rho_full = _ac(self.flat_gain_series(include_los=True, **kw))
        if self.has_los:
            _, rho_nlos = _ac(self.flat_gain_series(include_los=False, **kw))
        else:
            rho_nlos = rho_full

        def _dt(rho: np.ndarray, level: float) -> float:
            idx = np.argmax(rho[1:] < level) + 1
            return float(lags[idx]) if rho[idx] < level else float("nan")

        return {"lags_s": lags, "rho_full": rho_full, "rho_nlos": rho_nlos,
                "los_floor": self.k_factor_linear / (1.0 + self.k_factor_linear)
                if self.has_los else 0.0,
                "decorr_time_05_s": _dt(rho_nlos, 0.5),
                "decorr_time_09_s": _dt(rho_nlos, 0.9)}
