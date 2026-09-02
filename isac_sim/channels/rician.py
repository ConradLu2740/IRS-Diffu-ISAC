"""L1：莱斯衰落信道（LOS + 散射簇）。

模型：
    g(d) = sqrt(P_los) * exp(j*2*pi*d/lambda) + sqrt(P_sc / L) * Σ_{l=1..L} h_l
    h_l ~ CN(0, 1)，K = P_los / P_sc。

功率归一化：E|g|^2 = P_los + P_sc = |g_fs(d)|^2（与 FreeSpaceChannel 平均功率一致），
保证不同信道档位下的链路预算可比。P_los 由自由空间增益幅度平方给出。

状态：smoke 级 —— 无角度依赖/极化/大气损耗；L2（3GPP TR 38.811 对齐）见 README 路线图。
"""

import numpy as np

from .base import ChannelModel
from .free_space import FreeSpaceChannel, IMPL_LOSS


class RicianChannel(ChannelModel):
    """莱斯衰落：K 因子 + 散射射线数 L，固定种子可复现。"""

    def __init__(self, carrier_hz: float, k_factor_db: float = 10.0,
                 n_scatter: int = 8, a_gain: float = 1.0, b_gain: float = 1.0,
                 seed: int = 42):
        super().__init__(carrier_hz, seed)
        self.k_linear = 10.0 ** (k_factor_db / 10.0)
        self.n_scatter = int(n_scatter)
        self.a_gain = float(a_gain)
        self.b_gain = float(b_gain)

    def _scatter_gain(self, d_m: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        # CN(0,1)：每个实/虚分量为方差 1/2 的高斯（E|h|^2 = 1）
        h = (rng.standard_normal((self.n_scatter,) + d_m.shape)
             + 1j * rng.standard_normal((self.n_scatter,) + d_m.shape)) / np.sqrt(2.0)
        return h.sum(axis=0)

    def link_gain(self, d_m: np.ndarray) -> np.ndarray:
        d_m = np.maximum(np.asarray(d_m, dtype=float), 1e-6)
        p_fs = IMPL_LOSS * self.a_gain * self.b_gain / d_m**2  # 自由空间平均功率
        p_los = p_fs * self.k_linear / (1.0 + self.k_linear)
        p_sc = p_fs / (1.0 + self.k_linear)
        los = np.sqrt(p_los) * np.exp(1j * 2.0 * np.pi * d_m / self.wavelength_m)
        scat = np.sqrt(p_sc / self.n_scatter) * self._scatter_gain(d_m)
        return los + scat
