"""数字调制基类。"""

from abc import ABC, abstractmethod

import numpy as np


class DigitalModulation(ABC):
    """bits → 复基带符号 → bits 的最小接口（能量归一化 Es=1）。"""

    bits_per_symbol: int

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """[n_bits] 0/1 比特 → [n_symbols] 复基带符号（Es=1）。"""

    @abstractmethod
    def demodulate_hard(self, symbols: np.ndarray) -> np.ndarray:
        """复基带符号 → 硬判决比特。"""

    @abstractmethod
    def theoretical_ber(self, snr_db: float) -> float:
        """AWGN 下的理论 BER（灰映射格雷编码假设）。"""
