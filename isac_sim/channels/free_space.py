"""L0：自由空间远场信道（默认实现）。

与 source_code/isac_sat/data_sat.py::get_channel_mat 同构：
    H[i,j] = sqrt(b_gain * a_gain * 0.1) * exp(j*2*pi*d/lambda) / max(d, eps)
其中 0.1 为工程实现的口径/实现损耗因子（保留以便数值对齐交叉验证）。
"""

import numpy as np

from .base import ChannelModel

IMPL_LOSS = 0.1  # 与 isac_sat 一致的实现损耗因子


class FreeSpaceChannel(ChannelModel):
    """自由空间远场：确定性相位 + 1/d 幅度衰减。"""

    def __init__(self, carrier_hz: float, a_gain: float = 1.0, b_gain: float = 1.0):
        super().__init__(carrier_hz)
        self.a_gain = float(a_gain)
        self.b_gain = float(b_gain)

    def link_gain(self, d_m: np.ndarray) -> np.ndarray:
        d_m = np.maximum(np.asarray(d_m, dtype=float), 1e-6)
        amp = np.sqrt(IMPL_LOSS * self.a_gain * self.b_gain) / d_m
        return amp * np.exp(1j * 2.0 * np.pi * d_m / self.wavelength_m)
