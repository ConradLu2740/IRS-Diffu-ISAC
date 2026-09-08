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

import math

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


# ---------------------------------------------------------------------
# 双站反例：用第二个测距源（如已有的地面 UE）做三边定位，绕开角度墙
#
# 单站角度定位被物理上界封死，但两个站的**距离**测量（距离分辨率由带宽
# 决定，与阵列孔径无关）在目标处相交即可 2D 定位。
#
# 几何条件（v2 修正，星-地场景实测验证）：
#   - 两站视线夹角 γ 由实际单位向量计算：γ = arccos(û₁·û₂)。
#     默认星-地场景（BS 仰角 33.7°，UE 水平方向且方位差 142.5°）
#     下 γ ≈ 131°——远优于早期小角度估计（B/R ≈ 3.9°，错误地把
#     地面基线当成了视线夹角）。
#   - 退化警告：当两站方位差 Δaz → 0（UE 落入 BS-目标垂直面）时，
#     两个视线的地面投影平行，2D 定位秩亏（一阶不可观测），
#     误差与噪声同向爆炸——与 γ 大小无关。UE 必须在垂直面之外。
#
# 一阶公式的局限（v1.7 审查修正）：本模块的 trilateration_cross_range_error
# 只接收 γ，得到 σ_ρ/sinγ；但完整 GDOP 还取决于两站**仰角**（地面约束下
# 生效的是水平投影几何，BS 行向被 cos(elev) 压缩）。默认几何下一阶式低估
# 约 1.6×（0.20 vs 0.34 m 实测）。需要含站位的精确 GDOP 时用
# verify_twostation_localization.jacobian_gdop()，勿用本一阶式替代。
# ---------------------------------------------------------------------

def trilateration_cross_range_error(range_res_m: float, gamma_rad: float) -> float:
    """双站三边定位的交叉距离误差（米，一阶 GDOP 参考）。

    range_res_m: 假设的单站测距误差幅度 σ_ρ（米）。注意：σ_ρ 不是由带宽
        直接推出的量；c/(2B) 是距离分辨率。若用 c/(2B) 传入，得到的是
        “理想估计器”的下界参考，不是可达成误差。
    gamma_rad: 两站视线在目标处的夹角（用实际几何计算，勿用地面基线/斜距近似）。
    局限：只含 γ，未含两站仰角 → 地面约束定位的精确 GDOP 需站位置，
        见 verify_twostation_localization.jacobian_gdop()（默认几何下此
        一阶式低估约 1.6×）。
    秩亏（sin γ→0）返回 inf。
    """
    s = math.sin(gamma_rad)
    if s < 1e-9:
        return float("inf")
    return range_res_m / s


def scan_shortfall(n_elements_list, ranges_m, roi_width_m,
                   wavelength_m, spacing_factor: float = 0.5):
    """配置扫描： shortfall = Rayleigh 限 / ROI 张角（越大越"墙"）。

    返回 [len(n_elements_list), len(ranges_m)] 的二维数组；
    shortfall >= 1 表示阵列分辨率不足以分辨 ROI 内部（角度墙生效）。
    """
    out = np.zeros((len(n_elements_list), len(ranges_m)))
    for i, n in enumerate(n_elements_list):
        theta_res = rayleigh_limit_rad(int(n), wavelength_m,
                                       spacing_factor * wavelength_m)
        for j, r in enumerate(ranges_m):
            out[i, j] = theta_res / angular_extent(roi_width_m, r)
    return out


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
