"""1-bit 离散相位 RIS（smoke 级）。

相位只能取 {0, pi}：对理想连续相位解做最近邻量化
（等价于取 sign(cos(theta*)) 决定 0 或 pi）。
这是 RIS 硬件离散化的最粗档位，也是 1-bit/2-bit/连续系列的第一个台阶。
"""

import numpy as np

from .base import RisModel


class BinaryPhaseRis(RisModel):
    def configure(self, effective_gain: np.ndarray) -> np.ndarray:
        theta = np.angle(np.asarray(effective_gain))
        return np.where(np.cos(theta) >= 0.0, 0.0, np.pi)
