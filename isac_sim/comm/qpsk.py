"""QPSK over AWGN 最小链路（smoke 级）。

格雷映射 QPSK：每比特独立看等价于 BPSK over AWGN，
理论 BER = Q(sqrt(2*EbN0))，EbN0 = SNR / bits_per_symbol（Es=1, 2 比特/符号）。
"""

import numpy as np

from .base import DigitalModulation


def _qfunc(x: float) -> float:
    import math
    return 0.5 * math.erfc(x / np.sqrt(2.0))


class QpskAwgnLink(DigitalModulation):
    bits_per_symbol = 2

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=int).reshape(-1, self.bits_per_symbol)
        # 格雷映射：b0 → I 符号，b1 → Q 符号
        symbols = (1.0 - 2.0 * bits[:, 0]) + 1j * (1.0 - 2.0 * bits[:, 1])
        return symbols / np.sqrt(self.bits_per_symbol)  # Es = 1

    def demodulate_hard(self, symbols: np.ndarray) -> np.ndarray:
        symbols = np.asarray(symbols)
        b0 = (symbols.real < 0).astype(int)
        b1 = (symbols.imag < 0).astype(int)
        return np.stack([b0, b1], axis=1).reshape(-1)

    def theoretical_ber(self, snr_db: float) -> float:
        ebn0 = 10.0 ** (snr_db / 10.0) / self.bits_per_symbol
        return _qfunc(np.sqrt(2.0 * ebn0))

    def run_awgn(self, n_bits: int = 20_000, snr_db: float = 8.0) -> dict:
        """端到端一次：随机比特 → QPSK → AWGN → 硬判决 → BER。"""
        bits = self.rng.integers(0, 2, n_bits)
        tx = self.modulate(bits)
        noise_std = np.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)  # 每实维噪声方差
        noise = (self.rng.standard_normal(len(tx))
                 + 1j * self.rng.standard_normal(len(tx))) * noise_std
        rx = tx + noise
        bits_hat = self.demodulate_hard(rx)
        ber = float(np.mean(bits != bits_hat))
        return {"ber": ber, "ber_theory": self.theoretical_ber(snr_db), "n_bits": n_bits}
