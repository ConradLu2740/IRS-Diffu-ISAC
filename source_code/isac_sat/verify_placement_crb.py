"""verify_placement_crb.py — CRB 最优双站测距定位的 UE 部署设计（spec 2026-09-11）

把 §6.4 的"双站能破墙"升级为"怎么部署最好"：
  1. 噪声模型（Task A）：σ_i(d) = σ_ref·(d/d_ref)²（双程雷达方程 d⁴ → 测距 σ ∝ d²），
     σ_BS 固定于 695 km 参考，σ_UE 随 d_UE 扫描。诚实标注：σ 为理想估计器下界参照。
  2. CRB 闭式（Task B）：地面约束 2D、两站测距、各向异性噪声。
     记 A = cos²e_b/σ_BS²（BS 行权），B = 1/σ_UE²（UE 行权，UE→目标地面视线 cos=1），
     交叉距离方差 σ_cross² = (A + B·cos²Δaz)/(A·B·sin²Δaz)。
     **解析最优：Δaz=90° 时 σ_cross = σ_UE，且与 σ_BS 无关**（f 对 cos² 单调增 → 全局最小）。
     沿视线方差（90° 处）= σ_BS²/cos²e_b。
  3. MC 验证：Gauss-Newton 蒙特卡洛 vs 闭式（线性区一致）。
  4. 部署扫描（Task C）：Δaz × d_UE；通信足迹（UE 处卫星仰角 ≥ 20°）与 keep-out 约束下
     的最优部署；量化"通信与测距目标在本几何中同向（都偏好近/直角），约束来自部署政策"。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_sat as ss
from verify_twostation_localization import enu_from_ecef, gauss_newton_2d

SIGMA_REF_M = 0.15      # 参考测距误差（假设值，同 v1.7 σ_ρ 语义；非 c/2B）
D_UE_REF_KM = 47.7      # 默认场景 UE 地面距离
MIN_ELEV_DEG = 20.0     # 通信最低仰角
R_EARTH_KM = 6371.0


def sigma_ue(d_ue_km):
    """UE→目标双程测距误差（双程功率 ∝ d⁻⁴ → σ ∝ d²）。"""
    return SIGMA_REF_M * (d_ue_km / D_UE_REF_KM) ** 2


def crb_cross_variance(p_bs, p_ue, sigma_bs, sigma_ue):
    """交叉距离 CRB（方差，m²）。p_* 为目标 ENU 系三维站坐标（m）。"""
    c_b = np.linalg.norm(p_bs[:2]) / np.linalg.norm(p_bs)     # BS 行水平投影长度
    A = c_b ** 2 / sigma_bs ** 2
    # UE 在地面，目标在地面 → UE→目标视线水平，行投影长度 = 1
    # 关键：用相对 BS 方位的角（推导以 BS 视线水平投影为 x 轴）
    az_rel = math.atan2(p_ue[1], p_ue[0]) - math.atan2(p_bs[1], p_bs[0])
    B = 1.0 / sigma_ue ** 2
    s = math.sin(az_rel) ** 2
    if s < 1e-12:
        return float("inf")                                    # Δaz=0/180° 秩亏
    t = math.cos(az_rel) ** 2
    return (A + B * t) / (A * B * s)


def cross_rmse_mc(p_bs, p_ue, sigma_bs, sigma_ue, n_mc, seed):
    """MC：两站测距 + GN，返回 (2D RMSE, 交叉 RMSE)。"""
    rng = np.random.default_rng(seed)
    los2 = p_bs[:2] / np.linalg.norm(p_bs[:2])
    perp = np.array([-los2[1], los2[0]])
    err2, errc = [], []
    for _ in range(n_mc):
        t = np.array([rng.uniform(-40, 40), rng.uniform(-40, 40)])
        q = np.array([t[0], t[1], 0.0])
        r_bs = np.linalg.norm(q - p_bs) + rng.normal(0, sigma_bs)
        r_ue = np.linalg.norm(q - p_ue) + rng.normal(0, sigma_ue)
        est = gauss_newton_2d(p_bs, p_ue, r_bs, r_ue, x0=t + rng.normal(0, 10, 2))
        e = est - t
        err2.append(np.linalg.norm(e))
        errc.append(abs(e @ perp))
    return float(np.sqrt(np.mean(np.square(err2)))), float(np.sqrt(np.mean(np.square(errc))))


def main():
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    mid = frames[len(frames) // 2]
    R = enu_from_ecef(mid["target_pos"])
    center = np.asarray(mid["target_pos"]) * 1000.0
    p_bs = R @ (np.asarray(mid["sat_pos"]) * 1000.0 - center)
    e_b = math.asin(p_bs[2] / np.linalg.norm(p_bs))
    az_bs = math.atan2(p_bs[1], p_bs[0])
    c_b = math.cos(e_b)
    sigma_bs = SIGMA_REF_M
    n_mc = 2000

    print(f"几何：BS 仰角 {math.degrees(e_b):.1f}°（cos={c_b:.3f}），水平距 "
          f"{np.linalg.norm(p_bs[:2])/1e3:.0f} km；σ_BS={sigma_bs} m（695 km 参考）")

    # ---- 1. CRB 闭式 vs MC：Δaz 扫描（d_UE 固定 47.7 km）----
    print("\n[1] Δaz 扫描（d_UE=47.7 km）：闭式 σ_cross vs MC")
    print(f"{'Δaz':>6} {'CRB σ_cross':>12} {'MC σ_cross':>12} {'MC 2D':>10}")
    d_ue = D_UE_REF_KM * 1e3
    for daz in (30, 45, 60, 90, 120, 142, 180):
        az = az_bs + math.radians(daz)
        p_ue = np.array([d_ue * math.cos(az), d_ue * math.sin(az), 0.0])
        var = crb_cross_variance(p_bs, p_ue, sigma_bs, sigma_ue(d_ue / 1e3))
        _, mc = cross_rmse_mc(p_bs, p_ue, sigma_bs, sigma_ue(d_ue / 1e3), n_mc, 42 + daz)
        print(f"{daz:>5}° {math.sqrt(var):>11.3f}m {mc:>11.3f}m")

    # ---- 2. 解析最优性质验证：Δaz=90° 时 σ_cross = σ_UE，与 σ_BS 无关 ----
    print("\n[2] 解析最优（Δaz=90°）：σ_cross = σ_UE，与 σ_BS 无关")
    az = az_bs + math.radians(90)
    p_ue90 = np.array([d_ue * math.cos(az), d_ue * math.sin(az), 0.0])
    for sb in (0.05, 0.15, 1.0, 8.0):
        v = crb_cross_variance(p_bs, p_ue90, sb, sigma_ue(d_ue / 1e3))
        assert abs(math.sqrt(v) - sigma_ue(d_ue / 1e3)) < 1e-9, (sb, math.sqrt(v))
    print(f"  σ_BS ∈ {{0.05,0.15,1,8}} m → σ_cross 恒 = {sigma_ue(d_ue/1e3):.3f} m  ✓（解析已证）")
    print(f"  沿视线 CRB @90° = σ_BS/cos(e_b) = {sigma_bs/c_b:.3f} m")

    # ---- 3. d_UE 扫描 + 通信足迹 + keep-out（Task C）----
    print("\n[3] d_UE 扫描（Δaz=90°）：σ_cross = σ_UE(d) 与通信足迹")
    z_bs = p_bs[2]
    print(f"{'d_UE km':>8} {'σ_cross m':>10} {'UE处卫星仰角':>12} {'通信可行':>8}")
    for d_km in (1, 5, 10, 47.7, 100, 300, 1000):
        az = az_bs + math.radians(90)
        p_ue = np.array([d_km * 1e3 * math.cos(az), d_km * 1e3 * math.sin(az), 0.0])
        dh = np.linalg.norm(p_bs[:2] - p_ue[:2])
        elev_ue = math.degrees(math.atan2(z_bs, dh))
        ok = elev_ue >= MIN_ELEV_DEG
        print(f"{d_km:>8.1f} {sigma_ue(d_km):>10.3f} {elev_ue:>11.1f}° "
              f"{'✓' if ok else '✗':>8}")
    d_max = z_bs / math.tan(math.radians(MIN_ELEV_DEG)) / 1e3
    print(f"  足迹上界（Δaz=90°，仰角≥{MIN_ELEV_DEG}°）：d_UE ≤ {d_max:.0f} km"
          f"（通信足迹几乎不约束，约束来自部署政策/keep-out）")

    # ---- 4. 达到目标 RMSE 所需的部署 ----
    print("\n[4] 设计规则：达到目标交叉 RMSE 所需 d_UE（Δaz=90°）")
    for target in (0.15, 0.5, 1.0, 5.0):
        d_need = D_UE_REF_KM * math.sqrt(target / SIGMA_REF_M)
        print(f"  σ_cross ≤ {target:.2f} m → d_UE ≤ {d_need:.1f} km（Δaz=90°）")

    print("\n结论：")
    print("  1. 交叉距离 CRB 在 Δaz=90° 全局最优，最优值 = σ_UE 且与 σ_BS 无关（解析已证+MC 验证）。")
    print("  2. σ_UE ∝ d_UE²（双程雷达方程）→ 定位与通信目标在本几何中同向（都偏好近），")
    print("     约束来自部署政策（keep-out/服务区），不是通信足迹。")
    print("  3. σ_BS 只影响沿视线误差（σ_BS/cos e_b），不进入最优交叉距离——破墙靠第二测距源的质量。")


if __name__ == "__main__":
    main()
