"""RIS 模型基类与功率评估工具。"""

import numpy as np
from abc import ABC, abstractmethod


class RisModel(ABC):
    """RIS 相位配置接口。

    约定：`configure` 接收每个 RIS 元素的有效相干复增益
    g_i = (BS→RIS_i 链路和) * (RIS_i→ROI→UE 级联和)，返回单位模相位 [N]（弧度）。
    """

    @abstractmethod
    def configure(self, effective_gain: np.ndarray) -> np.ndarray:
        """给定有效复增益 [N]，返回相位配置 [N]（弧度）。"""


def coherent_power(effective_gain: np.ndarray, phases: np.ndarray) -> float:
    """给定有效增益与相位，返回相干合成功率 |Σ g_i * exp(j*phi_i)|^2。

    这是 isac_sat 闭环功率模型在最粗粒度上的等效（smoke 评估用；
    场景级评估仍以 isac_sat 的完整 5 径模型为准）。
    """
    effective_gain = np.asarray(effective_gain)
    phases = np.asarray(phases)
    y = np.sum(effective_gain * np.exp(1j * phases))
    return float(np.abs(y) ** 2)
