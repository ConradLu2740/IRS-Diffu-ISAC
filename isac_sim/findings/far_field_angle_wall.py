"""Finding 2：远场角度墙 —— 单站交叉距离定位的物理上界。

解析核心（3 个可复用函数 + isac_sat 默认场景下的预计算结论）。
完整配置扫描与热力图见 findings/README.md（P2）。

推导：
  - ROI 宽度 W 在斜距 R 处的张角 theta_ROI ≈ W / R（小角度近似）。
  - N 元间距 d=λ/2 的 ULA，Rayleigh 角度分辨率
    theta_res ≈ 0.886 * lambda / (N * d) = 1.772 / N (rad)（d=λ/2 时）。
  - 若 theta_ROI << theta_res，则阵列角度观测量无法区分 ROI 内不同
    交叉距离位置 → 单站角度定位存在物理上界。
"""

import numpy as np

C_LIGHT = 299_792_458.0


def angular_extent(width_m: float, range_m: float) -> float:
    """宽度 W（米）的目标区在斜距 R（米）处的张角（弧度，小角度近似）。"""
    return float(np.arctan(width_m / max(range_m, 1e-9)))


def rayleigh_limit_rad(n_elements: int, wavelength_m: float, spacing_m: float) -> float:
    """N 元 ULA 的 Rayleigh 角度分辨率（弧度）：0.886 * λ / (N·d)。"""
    if n_elements < 2:
        raise ValueError("need >= 2 elements")
    return 0.886 * wavelength_m / (n_elements * spacing_m)


def required_array_aperture(range_m: float, width_m: float,
                            wavelength_m: float) -> float:
    """要使 Rayleigh 分辨率 <= ROI 张角所需的最小孔径 D = N·d（米）。

    由 0.886*λ/D <= W/R 解出 D >= 0.886*λ*R/W。
    """
    return 0.886 * wavelength_m * range_m / max(width_m, 1e-9)


# isac_sat 默认场景预计算（30 GHz, ISS ~695 km 斜距, 80 m ROI）
DEFAULT_SCENARIO = {
    "carrier_hz": 30e9,
    "roi_width_m": 80.0,
    "slant_range_m": 695e3,
    "n_elements": 8,
}


def default_scenario_report() -> dict:
    lam = C_LIGHT / DEFAULT_SCENARIO["carrier_hz"]
    theta_roi = angular_extent(DEFAULT_SCENARIO["roi_width_m"],
                               DEFAULT_SCENARIO["slant_range_m"])
    theta_res = rayleigh_limit_rad(DEFAULT_SCENARIO["n_elements"], lam, lam / 2)
    return {
        "roi_angular_extent_deg": np.degrees(theta_roi),
        "ula_rayleigh_limit_deg": np.degrees(theta_res),
        "shortfall_ratio": theta_res / theta_roi,
        "required_aperture_m": required_array_aperture(
            DEFAULT_SCENARIO["slant_range_m"],
            DEFAULT_SCENARIO["roi_width_m"], lam),
    }
