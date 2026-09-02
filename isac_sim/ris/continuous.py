"""连续相位解析对齐（默认实现）。

与 isac_sat/phase_optimizer_sat.py::optimize_frame 同构：
phi_i = -angle(g_i)，使各元素相干同相叠加。
"""

import numpy as np

from .base import RisModel


class ContinuousPhaseRis(RisModel):
    def configure(self, effective_gain: np.ndarray) -> np.ndarray:
        return -np.angle(np.asarray(effective_gain))
