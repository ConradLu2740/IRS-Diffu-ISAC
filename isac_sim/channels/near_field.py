"""
isac_sim/channels/near_field.py — 近场 ULA 信道层（NF-2）

泛化 data_sat.get_channel_mat 的近场分析层：ULA 沿 x 轴，阵元位置
x_n，点目标 (R, y)（傍轴），逐元精确球面相位 + 球面幅度：

    H_n(R, y) = sqrt(0.1)·exp(j·2π·(sqrt(x_n² + (R−y)²) − R)/λ) / (R−y)

远场近似（平面波前）在 Δφ = (2π/λ)D²/(8R) ≤ π/8 时成立（verify_near_field_crb.py F1）。
本层提供：信道生成、Fisher 信息（数值）、CRB、以及近场/远场两种 regime 的
合成数据生成（供 ML 定位实验）。
"""

import numpy as np

C = 299792458.0


class NearFieldUla:
    """近场 ULA 信道 + CRB（2D 目标参数 (R, y)）。"""

    def __init__(self, n_elements, aperture_m, fc_hz=30e9):
        self.N = n_elements
        self.D = aperture_m
        self.lam = C / fc_hz
        self.xn = (np.arange(n_elements) - (n_elements - 1) / 2) * (aperture_m / (n_elements - 1))

    # ------------------------------------------------------------------
    def channel(self, R, y):
        """逐元复信道 [N]。"""
        d = np.sqrt(self.xn ** 2 + (R - y) ** 2)
        return (np.sqrt(0.1) * np.exp(1j * 2.0 * np.pi * (d - R) / self.lam)
                / np.maximum(d, 1e-6))

    def farfield_channel(self, R, y):
        """远场近似（平面波前，仅一阶相位）——角度墙 regime。"""
        d = R - y
        u = self.xn / np.maximum(d, 1e-6)                    # sin θ ≈ x/R
        return (np.sqrt(0.1) * np.exp(1j * 2.0 * np.pi * self.xn * u / self.lam)
                / np.maximum(d, 1e-6))

    # ------------------------------------------------------------------
    def _jac(self, R, y, fn):
        eps = 1e-4
        h0 = fn(R, y)
        hR = fn(R + eps, y)
        hy = fn(R, y + eps)
        J = np.stack([(hR - h0) / eps, (hy - h0) / eps], axis=1)   # [N, 2] 复
        return J

    def crb(self, R, y, gamma_db=20.0, fn=None):
        """FIM 数值 CRB（2×2）。fn=None 用近场信道。"""
        fn = fn or self.channel
        gamma = 10.0 ** (gamma_db / 10.0)
        J = self._jac(R, y, fn)
        FIM = 2.0 * gamma * np.real(J.conj().T @ J)
        return np.linalg.inv(FIM + 1e-20 * np.eye(2))

    def rayleigh_m(self):
        return 2.0 * self.D ** 2 / self.lam

    def crb_y_nuisance(self, R, gamma_db=20.0):
        """交叉距离 y 的 CRB（全局相位为未知干扰参数，投影 FIM）。

        无相位参考的相干接收机：h_n = A_n·exp(j(φ_n(y)+ψ))，ψ 未知。
        FIM(y,ψ) = 2γ·[[Σ|h|²φ'², Σ|h|²φ'], [Σ|h|²φ', Σ|h|²]]
        CRB_y = FIM⁻¹[0,0]（解析导数）。
        """
        gamma = 10.0 ** (gamma_db / 10.0)
        d = np.sqrt(self.xn ** 2 + R ** 2)
        h_abs2 = np.full(self.N, 1.0 / self.N)      # 单位能量归一化
        # φ_n(y) = 2π(sqrt(x_n²+(R−y)²)−R)/λ ⇒ φ'_n(0) = −2π/λ·R/d
        dphi = -2.0 * np.pi / self.lam * R / d
        # 投影形式（数值稳定）：消去全局相位等价于对 φ' 取加权方差
        w = h_abs2 / h_abs2.sum()
        var_phi = float(np.sum(w * (dphi - np.sum(w * dphi)) ** 2))
        if var_phi <= 0:
            return float("inf")
        return float(1.0 / (2.0 * gamma * var_phi))

    def crb_y_farfield(self, R, gamma_db=20.0):
        """远场 DOA 的 CRB（同样带全局相位干扰参数）。

        φ_n = 2π x_n sinθ/λ, sinθ ≈ y/R ⇒ φ'_n = 2π x_n/(Rλ)。
        """
        gamma = 10.0 ** (gamma_db / 10.0)
        d = max(R, 1e-6)
        h_abs2 = np.full(self.N, 1.0 / self.N)      # 单位能量归一化
        dphi = 2.0 * np.pi * self.xn / (d * self.lam)
        w = np.full(self.N, h_abs2 / (h_abs2 * self.N))
        var_phi = float(np.sum(w * (dphi - np.sum(w * dphi)) ** 2))
        if var_phi <= 0:
            return float("inf")
        return float(1.0 / (2.0 * gamma * var_phi))


def make_dataset(ula, n, r_range, y_range, gamma_db=20.0, farfield=False,
                 seed=0):
    """合成定位数据集：返回 X [n, 2N]（实/虚部）、Y [n, 2]（R, y）。"""
    rng = np.random.default_rng(seed)
    fn = ula.farfield_channel if farfield else ula.channel
    R = rng.uniform(*r_range, n)
    y = rng.uniform(*y_range, n)
    gamma = 10.0 ** (gamma_db / 10.0)
    X = np.empty((n, 2 * ula.N), dtype=np.float32)
    for i in range(n):
        h = fn(R[i], y[i])
        noise = (rng.standard_normal(ula.N) + 1j * rng.standard_normal(ula.N)) \
            * np.sqrt(0.5 / gamma)
        obs = h + noise
        X[i, :ula.N] = obs.real
        X[i, ula.N:] = obs.imag
    Y = np.stack([R, y], axis=1).astype(np.float32)
    return X, Y
