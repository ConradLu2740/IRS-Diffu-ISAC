"""波形基类。"""

from abc import ABC, abstractmethod

import numpy as np


class Waveform(ABC):
    """感知/通信一体波形接口。

    子类实现 `channel_response`（给定散射中心时延/幅度，返回频域响应）
    与 `range_profile`（频域响应 → 距离像）。
    """

    @abstractmethod
    def channel_response(self, delays_s: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        """H(f_k) = Σ_i a_i * exp(-j*2*pi*f_k*tau_i)，返回 [n_sub] 复数频域响应。"""

    @abstractmethod
    def range_profile(self, h_freq: np.ndarray) -> np.ndarray:
        """频域响应 → 距离像（长度 n_sub 的实幅度序列）。"""

    @property
    @abstractmethod
    def range_resolution_m(self) -> float:
        """理论距离分辨率 c / (2B)（单站收发双程）。"""
