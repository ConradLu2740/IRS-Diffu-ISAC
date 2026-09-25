"""波形基类。"""

from abc import ABC, abstractmethod

import numpy as np


class Waveform(ABC):
    """感知/通信一体波形接口。

    子类实现 `channel_response`（给定散射中心时延/幅度，返回频域响应）
    与 `range_profile`（频域响应 → 距离像）。

    `dd_response` 为 DD（时延-多普勒）域抽象：给定散射中心的时延/多普勒/
    幅度，返回波形本生素形下的 DD 域响应（OTFS 的 DD 网格响应、AFDM 的
    DAFT 域仿射 DD 响应）。默认实现抛 NotImplementedError——保持既有
    OFDM 等实现无需改动即可实例化；新的 2D 波形（OTFS/AFDM）应覆盖它。
    """

    @abstractmethod
    def channel_response(self, delays_s: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        """H(f_k) = Σ_i a_i * exp(-j*2*pi*f_k*tau_i)，返回 [n_sub] 复数频域响应。"""

    @abstractmethod
    def range_profile(self, h_freq: np.ndarray) -> np.ndarray:
        """频域响应 → 距离像（长度 n_sub 的实幅度序列）。"""

    def dd_response(self, delays_s: np.ndarray, dopplers_hz: np.ndarray,
                    amplitudes: np.ndarray) -> np.ndarray:
        """DD 域（时延-多普勒）信道响应（默认未实现）。

        delays_s / dopplers_hz / amplitudes: 散射中心时延(s)、多普勒(Hz)、
        复幅度。返回形状由子类定义（OTFS: [n_symbols, n_subcarriers] =
        [Doppler, delay]；AFDM: [n_subcarriers] DAFT 域仿射 DD 响应）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 dd_response（DD 域感知抽象）")

    @property
    @abstractmethod
    def range_resolution_m(self) -> float:
        """理论距离分辨率 c / (2B)（单站收发双程）。"""
