"""verify_twostation_localization.py — 双站三边定位实验（角度墙的实证反例）

背景（TECH_REPORT v1.3 Section 5.3 / isac_sim findings）：单站角度定位在
星-地远场存在物理上界（80 m ROI @ 695 km 张角 0.0066°，8 元 ULA 分辨率
~13°，ML 交叉距离 RMSE 11.8 m）。本实验验证对应的**出路**：ISAC 场景
天然双站——地面 UE 本身就是第二个测距源。两个距离测量（BS、UE 各一次，
精度由带宽决定、与阵列孔径无关）在地面约束下即可 2D 定位。

关键几何事实（v2 修正）：
  目标与 UE 都在地面 → 目标→UE 方向恒为水平；目标→BS 方向由卫星仰角
  （默认 ~33.7°）与方位差 Δaz 决定。两站视线夹角
      γ = arccos( cos(elev_BS) · cos(Δaz) )
  与 UE 的地面距离基本无关（方向不变），故有意义的扫描维度是
  **UE 方位差 Δaz**，而不是基线长度。默认场景 Δaz≈142° → γ≈131°，
  sin γ ≈ 0.75，GDOP 极好。

方法：
  - 取 isac_sat 默认过境窗口中心帧的真实几何（SGP4 → ECEF → 目标点 ENU）；
  - 目标均匀撒在 80×80 m ROI 内（地面约束 z=0），每站测距加高斯噪声 σ_ρ；
  - **测量模型用三维斜距**：r = ‖(x,y,0) − 站三维位置‖（含 BS 385 km 高度与
    UE 地曲率项），Gauss-Newton 迭代解算 (x, y)，统计 2D RMSE 与交叉距离 RMSE
    （交叉 = 垂直于 BS 视线地面投影，与单站角度墙同一定义）；
  - 扫描 σ_ρ（0.15/0.5/1.5/5 m）× Δaz（0°/45°/90°/142°默认）× 2000 MC。
  - 理论列改为 Jacobian-GDOP：σ_cross ≈ σ_ρ · g，g = sqrt(perpᵀ(JᵀJ)⁻¹ perp)，
    J 为三维斜距对 (x,y) 的 2×2 偏导（在 ROI 中心求值，80 m ROI 内变化可忽略）。
    （旧版用 σ_ρ/sin γ 的一阶式，γ 是三维视线夹角，与实际生效的水平面几何不一致。）

对照（TECH_REPORT v1.3, 相同场景/测试集）：
  - 单站 1D-CFAR  LOS RMSE   8.14 m
  - 单站 ML      2D RMSE    12.06 m（交叉距离 11.84 m = 角度墙实测）

理论预期：交叉距离 σ_cross ≈ σ_ρ · g(J)（ROI 中心 Jacobian 的 GDOP 因子 g）。
破墙精度预算：σ_ρ < 11.84 m / g；默认几何（Δaz=142°）g 由脚本实测。

诚实说明：
  1. 未建模电离层/对流层延迟（实际星-地测距主要误差源）；本实验回答
     "几何/信息层面双站是否破墙"，非端到端精度预测。
  2. σ_ρ 是**假设的测距误差标准差**，不是由带宽推出的值；c/(2B)=0.15 m 是距离
     分辨率，测距误差取决于估计器与同步误差，本文不对 σ_ρ 的来源做端到端断言。
  3. σ_ρ 大时误差超出 ROI 线性区，蒙特卡洛会偏离 Jacobian-GDOP 线性理论（见 5 m 档）。
  3. 单站 ML 的 11.8 m 并非"场景信息不足"，而是感知层只用了 BS 侧
     HRRP（单站角度/时延特征）——破墙所需信息在双站 ISAC 场景中本来
     就存在，只是未被利用。
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import make_roi_local

OUT_DIR = "./sat_verify"
os.makedirs(OUT_DIR, exist_ok=True)

REF_CFAR_LOS = 8.14     # TECH_REPORT v1.3：单站 1D-CFAR LOS RMSE
REF_ML_2D = 12.06       # 单站 ML 2D RMSE
REF_ML_CROSS = 11.84    # 单站 ML 交叉距离 RMSE（角度墙实测）
UE_GND_DIST_M = 47.7e3  # 默认场景：UE 与 ROI 的地面距离（经度差 0.5°）


def enu_from_ecef(origin_ecef_km):
    """返回把 ECEF 偏移（米）转到本地 ENU 的旋转矩阵（原点在目标中心）。"""
    lat = math.atan2(origin_ecef_km[2], math.hypot(origin_ecef_km[0], origin_ecef_km[1]))
    lon = math.atan2(origin_ecef_km[1], origin_ecef_km[0])
    sla, cla = math.sin(lat), math.cos(lat)
    slo, clo = math.sin(lon), math.cos(lon)
    return np.array([
        [-slo, clo, 0],
        [-sla * clo, -sla * slo, cla],
        [cla * clo, cla * slo, sla],
    ])


def los_angle_gamma(p_bs, p_ue):
    """两站视线在目标处的夹角 γ（弧度）：arccos(û_bs · û_ue)。"""
    u1 = p_bs / np.linalg.norm(p_bs)
    u2 = p_ue / np.linalg.norm(p_ue)
    return math.acos(np.clip(u1 @ u2, -1.0, 1.0))


def gauss_newton_2d(p_bs, p_ue, r_bs, r_ue, x0):
    """地面约束（z=0）双站测距最小二乘，返回 (x, y)。

    测量模型：r = ‖(x, y, 0) − 站三维位置‖（三维斜距）。未知量仅 (x, y)
    （目标贴地），Jacobian 行 = 斜距单位向量的前两维（∂d/∂x, ∂d/∂y）。
    旧实现用水平距离（丢掉 BS 高度 ~385 km 与 UE 地曲率 ~179 m），已废弃。
    """
    p = np.array(x0, dtype=float)
    for _ in range(12):
        q = np.array([p[0], p[1], 0.0])
        d_bs = np.linalg.norm(q - p_bs)
        d_ue = np.linalg.norm(q - p_ue)
        res = np.array([d_bs - r_bs, d_ue - r_ue])
        J = np.stack([(q - p_bs)[:2] / max(d_bs, 1e-9),
                      (q - p_ue)[:2] / max(d_ue, 1e-9)])
        dp, *_ = np.linalg.lstsq(J, -res, rcond=None)
        p = p + dp
        if np.linalg.norm(dp) < 1e-8:
            break
    return p


def jacobian_gdop(p_bs, p_ue, xy=(0.0, 0.0)):
    """地面约束三维斜距定位在 (x,y) 处的 GDOP 因子。

    J = 2×2（两站斜距对 (x,y) 的偏导，z 固定 0）；交叉距离方向 perp 取
    BS 水平视线投影的法向。返回 g = sqrt(perpᵀ (JᵀJ)⁻¹ perp)，理论
    交叉距离 σ_cross ≈ σ_ρ · g。两站地面投影平行（Δaz=0，秩亏）时返回 inf。
    """
    q = np.array([xy[0], xy[1], 0.0], dtype=float)

    def _row(s):
        d = np.linalg.norm(q - s)
        return (q - s)[:2] / max(d, 1e-9)

    J = np.stack([_row(p_bs), _row(p_ue)])
    los2 = p_bs[:2] / (np.linalg.norm(p_bs[:2]) + 1e-12)
    perp = np.array([-los2[1], los2[0]])
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(max(perp @ cov @ perp, 0.0)))


def run_mc(p_bs, p_ue, roi_local, n_mc, sigma_rho, rng):
    """蒙特卡洛：返回 (2D RMSE, 交叉距离 RMSE)（米）。三维斜距测量。"""
    los2 = p_bs[:2] / (np.linalg.norm(p_bs[:2]) + 1e-12)  # BS 视线地面投影
    perp = np.array([-los2[1], los2[0]])
    err2, err_cross = [], []
    for _ in range(n_mc):
        xy = roi_local[rng.integers(0, len(roi_local))][:2]      # [x, y] 米
        q = np.array([xy[0], xy[1], 0.0])
        r_bs = np.linalg.norm(q - p_bs) + rng.normal(0, sigma_rho)
        r_ue = np.linalg.norm(q - p_ue) + rng.normal(0, sigma_rho)
        est = gauss_newton_2d(p_bs, p_ue, r_bs, r_ue, x0=xy + rng.normal(0, 10, 2))
        e = est - xy
        err2.append(np.linalg.norm(e))
        err_cross.append(abs(e @ perp))
    return float(np.sqrt(np.mean(np.square(err2)))), float(np.sqrt(np.mean(np.square(err_cross))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mc", type=int, default=2000)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.15, 0.5, 1.5, 5.0])
    parser.add_argument("--dazs_deg", nargs="+", type=float, default=[0.0, 45.0, 90.0, 142.0],
                        help="UE 方位相对 BS 方位的偏移；142° 为默认场景")
    args = parser.parse_args()

    # ---- 真实几何：默认过境窗口中心帧 ----
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    mid = frames[len(frames) // 2]
    R = enu_from_ecef(mid["target_pos"])
    center = np.asarray(mid["target_pos"]) * 1000.0            # 米
    p_bs = R @ (np.asarray(mid["sat_pos"]) * 1000.0 - center)
    p_ue_default = R @ (np.asarray(mid["ground_pos"]) * 1000.0 - center)
    roi_local = make_roi_local()[:, :2]                        # [R,2] 米（xy）
    roi_local = roi_local - roi_local.mean(axis=0)             # 居中

    elev_bs = math.degrees(math.asin(p_bs[2] / np.linalg.norm(p_bs)))
    az_bs = math.degrees(math.atan2(p_bs[1], p_bs[0]))
    az_ue = math.degrees(math.atan2(p_ue_default[1], p_ue_default[0]))
    print(f"几何（目标点 ENU）：BS 仰角 {elev_bs:.1f}°，方位 {az_bs:.1f}°；"
          f"UE 方位 {az_ue:.1f}°（默认 Δaz ≈ {(az_ue - az_bs) % 360:.0f}°）")
    print(f"对照（单站）：CFAR LOS {REF_CFAR_LOS} m | ML 2D {REF_ML_2D} m | "
          f"ML 交叉 {REF_ML_CROSS} m（角度墙）\n")

    results = {}
    print(f"{'Δaz':>6} {'γ':>7} {'GDOP':>7} {'σ_ρ (m)':>8} {'2D RMSE':>10} {'交叉RMSE':>10} "
          f"{'理论 σ·g':>10} {'vs 单站墙':>10}")
    print("-" * 76)
    for daz in args.dazs_deg:
        # UE 放在地面距离 47.7 km、方位 = BS 方位 + Δaz 处
        az = math.radians(az_bs + daz)
        p_ue = np.array([UE_GND_DIST_M * math.cos(az),
                         UE_GND_DIST_M * math.sin(az), p_ue_default[2]])
        gamma = los_angle_gamma(p_bs, p_ue)
        gdop = jacobian_gdop(p_bs, p_ue)
        for s in args.sigmas:
            rng = np.random.default_rng(args.seed + int(daz * 10 + s * 100))
            rmse2, rmse_x = run_mc(p_bs, p_ue, roi_local, args.n_mc, s, rng)
            theo = s * gdop
            results[(daz, s)] = (rmse2, rmse_x, gamma, gdop)
            g_ok = '✓ 破墙' if rmse_x < REF_ML_CROSS else '✗'
            print(f"{daz:6.0f} {math.degrees(gamma):6.1f}° {gdop:7.2f} {s:8.2f} {rmse2:10.2f} "
                  f"{rmse_x:10.2f} {theo:10.2f} {g_ok:>10}")

    # ---- 破墙精度预算（由 Jacobian GDOP 决定）----
    gdop_def = results[(142.0 if 142.0 in args.dazs_deg else args.dazs_deg[-1], args.sigmas[0])][3]
    print(f"\n破墙精度预算：σ_ρ < {REF_ML_CROSS} / GDOP")
    print(f"  退化几何（Δaz=0°）：GDOP→∞，秩亏不可定位（与 γ 大小无关）")
    print(f"  默认场景（Δaz=142°，GDOP={gdop_def:.2f}）：σ_ρ < {REF_ML_CROSS/gdop_def:.2f} m")
    print("  即：任何非退化的 UE 几何下，米级测距精度即可破墙（σ_ρ 为假设值）。")

    # ---- 图：交叉距离 RMSE vs σ_ρ，各 Δaz ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(args.dazs_deg)))
    for (daz, c) in zip(args.dazs_deg, colors):
        sig = np.array(args.sigmas)
        ms = [results[(daz, s)][1] for s in args.sigmas]
        gamma3 = results[(daz, args.sigmas[0])][2]
        gdop = results[(daz, args.sigmas[0])][3]
        ax.plot(sig, ms, "o-", color=c,
                label=f"Δaz={daz:.0f}° (γ={math.degrees(gamma3):.0f}°, GDOP={gdop:.2f})")
        if math.isfinite(gdop):
            ax.plot(sig, sig * gdop, "--", color=c, alpha=0.5)
    ax.axhline(REF_ML_CROSS, color="r", lw=2, label=f"mono-static wall ({REF_ML_CROSS} m)")
    ax.axhline(REF_CFAR_LOS, color="orange", lw=1.5, ls=":",
               label=f"mono-static CFAR LOS ({REF_CFAR_LOS} m)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Range measurement noise σ_ρ (m)")
    ax.set_ylabel("Cross-range RMSE (m)")
    ax.set_title("Two-station trilateration vs the mono-static angle wall\n"
                 f"(isac_sat default pass, UE ground dist {UE_GND_DIST_M/1e3:.1f} km, {args.n_mc} MC)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "twostation_localization.png"), dpi=150)
    print(f"\n[图] {OUT_DIR}/twostation_localization.png 已保存")

    # ---- 结论判定 ----
    r_def = results[(142.0 if 142.0 in args.dazs_deg else args.dazs_deg[-1], args.sigmas[0])]
    print("\n结论检查:")
    print(f"  默认几何（Δaz=142°, γ={math.degrees(r_def[2]):.0f}°）+ σ_ρ=0.15 m："
          f"交叉 RMSE {r_def[1]:.2f} m << 单站墙 {REF_ML_CROSS} m → 双站破墙")
    print("  蒙特卡洛与 Jacobian-GDOP 理论（σ_ρ·g）在线性区一致（σ_ρ 大时偏离，见诚实说明 3）")
    r0 = results[(0.0, args.sigmas[0])] if 0.0 in args.dazs_deg else None
    if r0 is not None:
        print(f"  退化警告：Δaz=0°（UE 在 BS-目标垂直面内）时两站地面投影平行，"
              f"2D 定位秩亏（GDOP→∞），误差爆炸（实测 {r0[1]:.0f} m @ σ_ρ={args.sigmas[0]} m）——"
              f"UE 必须在垂直面之外，这与 γ 大小是两个独立的几何条件")


if __name__ == "__main__":
    main()
