"""信道模型基类：所有信道实现的最小插拔接口。"""

import numpy as np
from abc import ABC, abstractmethod


class ChannelModel(ABC):
    """信道模型接口。

    子类必须实现 `link_gain`：给定传播距离（米），返回复数链路增益。
    矩阵级辅助方法 `matrix` 默认基于 link_gain 构造，可按需覆盖。

    物理约定（与 isac_sat 保持同构，便于交叉验证）：
      H[i,j] = g(d_ij)，g 为无量纲复增益；自由空间下
      g(d) = sqrt(0.1) * exp(j*2*pi*d/lambda) / d。
    """

    def __init__(self, carrier_hz: float, seed: int = 42):
        self.carrier_hz = float(carrier_hz)
        self.seed = int(seed)
        self.wavelength_m = 299_792_458.0 / self.carrier_hz

    @abstractmethod
    def link_gain(self, d_m: np.ndarray) -> np.ndarray:
        """给定传播距离 d_m（米，任意 shape），返回同 shape 复数链路增益。"""

    def matrix(self, a_pos: np.ndarray, b_pos: np.ndarray) -> np.ndarray:
        """a→b 链路矩阵 [len(a), len(b)]。

        a_pos, b_pos: [N, 3] / [M, 3] 位置（米）。
        """
        a_pos = np.atleast_2d(a_pos)
        b_pos = np.atleast_2d(b_pos)
        d = np.linalg.norm(a_pos[:, None, :] - b_pos[None, :, :], axis=-1)
        return self.link_gain(d)
