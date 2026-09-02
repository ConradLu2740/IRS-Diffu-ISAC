"""一维 CA-CFAR（单元平均恒虚警，卷积向量化实现）。

经典 Rohling (1983) 形式的 1D 版本：噪声水平由训练单元（去除保护单元）
的滑窗均值估计，阈值 = alpha * 噪声估计，alpha = N * (P_fa^(-1/N) - 1)。
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


class CA_Cfar1D:
    def __init__(self, n_train: int = 16, n_guard: int = 4, p_fa: float = 1e-4):
        self.n_train = int(n_train)
        self.n_guard = int(n_guard)
        self.p_fa = float(p_fa)

    @property
    def alpha(self) -> float:
        n = 2 * self.n_train  # 两侧训练单元总数
        return n * (self.p_fa ** (-1.0 / n) - 1.0)

    def detect(self, x: np.ndarray) -> np.ndarray:
        """输入功率序列 [T]，返回布尔检测结果 [T]。"""
        x = np.asarray(x, dtype=float)
        win = self.n_train + self.n_guard
        # 滑窗均值（含保护单元），再扣除保护单元部分
        s_all = uniform_filter1d(x, size=2 * win + 1, mode="constant")
        s_guard = uniform_filter1d(x, size=2 * self.n_guard + 1, mode="constant")
        n_train_eff = 2 * self.n_train
        noise = (s_all * (2 * win + 1) - s_guard * (2 * self.n_guard + 1)) / n_train_eff
        noise[:win] = noise[-win:] = np.nan
        with np.errstate(invalid="ignore"):
            return x > self.alpha * noise

    def detect_peaks(self, x: np.ndarray) -> np.ndarray:
        """检测结果中的局部极大单元索引（去粘连）。"""
        det = self.detect(x)
        idx = np.where(det)[0]
        keep = [i for i in idx
                if i == 0 or i == len(x) - 1
                or (x[i] >= x[i - 1] and x[i] >= x[i + 1])]
        return np.array(keep, dtype=int)
