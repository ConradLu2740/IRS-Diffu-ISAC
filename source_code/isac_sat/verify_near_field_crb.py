"""
verify_near_field_crb.py — 近场 XL-RIS：角度墙的第三条破墙路径（Wall Map 的几何方向）

背景（§5.3 角度墙）：远场单站交叉距离不可用（80m ROI @695km 需 77m 孔径）。
本脚本验证第三条破墙路径——**大孔径近场**：当面板跨度 D 与距离 R 相比不可
忽略（球面波前曲率可辨），单站即可同时测距测角，无需第二站。

三项验证（numpy-only，对齐 isac_sim/findings 风格）：
  F1 远场有效性证书：孔径 D 上的球面波前二次相位偏差
     Δφ_max = (2π/λ)·D²/(8R) ≤ π/8 ⟺ D ≤ D* = sqrt(λR/2)
     ——用逐元精确距离模型验证该闭式界。
  F2 近场测距 CRB：N 元 ULA、点目标 (R, y) 傍轴，
     CRB(R) = R⁴λ² / (2π²γ Σx_n⁴)，MC（MLE/Gauss-Newton）验证。
  F3 近场窗口：Rayleigh 距离 R_F = 2D²/λ ——窗外曲率参数不可辨识
     （FIM 条件数爆炸），MC 误差发散；窗内单站破墙。

场景锚点（仓库既有几何）：
  - 星载 10m 面板 @695km：D* = 59m ⇒ 远场（正面证书化 §2 假设）
  - 地面 1m 面板 @47.7km：D* = 15.4m ⇒ 远场
  - UAV 载 1m 面板 @100m：D* = 0.71m ⇒ **真正近场**（低空场景）
"""

import os
import json
import argparse
import numpy as np

C = 299792458.0


# ----------------------------------------------------------------------
def exact_phase_profile(d_m, wavelength_m):
    """逐元精确相位（球面波前）。"""
    return 2.0 * np.pi * d_m / wavelength_m


def f1_farfield_certificate(wavelength_m, D, R, n=4096):
    """F1：闭式远场界 vs 逐元精确模型。

    ULA 沿 x 轴，孔径 D，目标在轴向距离 R 处（正入射）。
    精确相位差（相对阵列中心）：Δφ(x) = (2π/λ)(sqrt(R²+x²) − R)。
    小 x 展开：Δφ ≈ (2π/λ)·x²/(2R) ⇒ 最大偏差（x=D/2）=(2π/λ)·D²/(8R)。
    """
    x = np.linspace(-D / 2, D / 2, n)
    exact = exact_phase_profile(np.sqrt(R ** 2 + x ** 2), wavelength_m) - \
        exact_phase_profile(np.array([R]), wavelength_m)[0]
    quadratic = (2.0 * np.pi / wavelength_m) * x ** 2 / (2.0 * R)
    max_exact = float(np.max(np.abs(exact - quadratic)))
    closed_form = (2.0 * np.pi / wavelength_m) * D ** 2 / (8.0 * R)
    D_star = np.sqrt(wavelength_m * R / 2.0)          # Δφ_max = π/8 的孔径
    return {
        "D_m": D, "R_m": R, "wavelength_m": wavelength_m,
        "closed_form_quad_phase_rad": float(closed_form),
        "exact_minus_quad_max_rad": max_exact,
        "closed_form_accurate": bool(max_exact < 0.02 * max(closed_form, 1e-9)),
        "D_star_m": float(D_star),
        "farfield_valid": bool(closed_form <= np.pi / 8),
        "resid_ratio": float(max_exact / max(closed_form, 1e-12)),
    }


def f2_nearfield_crb(wavelength_m, N, d_el, R, y_off, gamma_db, n_mc=400, seed=0):
    """F2：近场测距/定位 CRB 闭式 vs MC（MLE）。

    模型：ULA 位于 x=0，阵元 x_n = (n-(N-1)/2)·d_el；点目标 (R, y)。
    远场近似失效，用精确球面相位：
      φ_n(R, y) = (2π/λ)(sqrt((x_n)² + (R-y)²) − R)
    每元 SNR γ（复 AWGN）。CRB 由 FIM 数值求逆给出（2×2），
    同时与傍轴闭式 CRB(R) = R⁴λ²/(2π²γ Σx_n⁴) 对表。
    """
    rng = np.random.default_rng(seed)
    xn = (np.arange(N) - (N - 1) / 2) * d_el
    gamma = 10.0 ** (gamma_db / 10.0)

    def phase(R_, y_):
        return 2.0 * np.pi / wavelength_m * (np.sqrt(xn ** 2 + (R_ - y_) ** 2) - R_)

    # FIM（数值导数）
    def num_jac(R_, y_):
        eps = 1e-3
        p0 = phase(R_, y_)
        pR = phase(R_ + eps, y_)
        py = phase(R_, y_ + eps)
        return np.stack([(pR - p0) / eps, (py - p0) / eps], axis=1)  # [N, 2]

    J = num_jac(R, y_off)
    FIM = 2.0 * gamma * (J.T @ J)
    crb = np.linalg.inv(FIM)
    sigma_x4 = float(np.sum(xn ** 4))
    crb_closed = R ** 4 * wavelength_m ** 2 / (2.0 * np.pi ** 2 * gamma * sigma_x4)

    # MC：MLE（Gauss-Newton from true）
    errs = []
    for _ in range(n_mc):
        obs = np.exp(1j * phase(R, y_off)) + \
            (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(0.5 / gamma)
        Rh, yh = R, y_off
        for _it in range(50):
            e = np.exp(1j * phase(Rh, yh))
            r = obs - e
            Jn = num_jac(Rh, yh)
            grad = 2.0 * gamma * np.real(Jn.T @ (-1j * (obs * np.conj(e) - 1.0)))
            H = 2.0 * gamma * (Jn.T @ Jn)
            try:
                step = np.linalg.solve(H + 1e-12 * np.eye(2), grad)
            except np.linalg.LinAlgError:
                break
            Rh, yh = Rh + step[0], yh + step[1]
            if np.linalg.norm(step) < 1e-9:
                break
        errs.append([Rh - R, yh - y_off])
    E = np.array(errs)
    mc_var = E.var(axis=0)
    return {
        "N": N, "d_el_m": d_el, "R_m": R, "y_off_m": y_off,
        "aperture_m": float((N - 1) * d_el),
        "crb_R_m2": float(crb[0, 0]), "crb_y_m2": float(crb[1, 1]),
        "crb_closed_R_m2": float(crb_closed),
        "mc_var_R": float(mc_var[0]), "mc_var_y": float(mc_var[1]),
        "mc_over_crb_R": float(mc_var[0] / crb[0, 0]),
        "n_mc": n_mc,
    }


def f3_rayleigh_window(wavelength_m, N, d_el, gamma_db, n_mc=300, seed=1):
    """F3：近场窗口 R_F = 2D²/λ 内外的 FIM 条件数与 MC 误差。"""
    D = (N - 1) * d_el
    R_F = 2.0 * D ** 2 / wavelength_m
    out = {"aperture_m": D, "rayleigh_m": float(R_F), "points": []}
    rng = np.random.default_rng(seed)
    gamma = 10.0 ** (gamma_db / 10.0)
    xn = (np.arange(N) - (N - 1) / 2) * d_el
    for factor in [0.25, 0.5, 1.0, 2.0, 4.0]:
        R = factor * R_F
        y_off = 1.0

        def phase(R_, y_):
            return 2.0 * np.pi / wavelength_m * (np.sqrt(xn ** 2 + (R_ - y_) ** 2) - R_)

        eps = 1e-3
        J = np.stack([(phase(R + eps, y_off) - phase(R, y_off)) / eps,
                      (phase(R, y_off + eps) - phase(R, y_off)) / eps], axis=1)
        FIM = 2.0 * gamma * (J.T @ J)
        cond = float(np.linalg.cond(FIM))
        errs = []
        for _ in range(n_mc):
            obs = np.exp(1j * phase(R, y_off)) + \
                (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(0.5 / gamma)
            Rh, yh = R, y_off
            for _it in range(50):
                e = np.exp(1j * phase(Rh, yh))
                r = obs - e
                Jn = np.stack([(phase(Rh + eps, yh) - phase(Rh, yh)) / eps,
                               (phase(Rh, yh + eps) - phase(Rh, yh)) / eps], axis=1)
                grad = 2.0 * gamma * np.real(Jn.T @ (-1j * (obs * np.conj(e) - 1.0)))
                H = 2.0 * gamma * (Jn.T @ Jn)
                try:
                    step = np.linalg.solve(H + 1e-12 * np.eye(2), grad)
                except np.linalg.LinAlgError:
                    break
                Rh, yh = Rh + step[0], yh + step[1]
                if np.linalg.norm(step) < 1e-9:
                    break
            errs.append([Rh - R, yh - y_off])
        mc_rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        out["points"].append({"R_over_RF": factor, "R_m": float(R),
                              "fim_cond": cond, "mc_rmse_m": mc_rmse})
    return out


def main(args):
    lam = C / args.fc_hz
    print(f"\n{'=' * 74}")
    print(f"近场 XL-RIS 验证（fc={args.fc_hz / 1e9:.0f} GHz, λ={lam * 100:.2f} cm）")
    print(f"{'=' * 74}")

    print("\nF1 远场有效性证书（闭式界 vs 逐元精确模型）:")
    f1_rows = []
    for name, D, R in [("星载 10m @695km", 10.0, 695e3),
                       ("地面 1m @47.7km", 1.0, 47.7e3),
                       ("UAV 1m @100m", 1.0, 100.0),
                       ("UAV 1m @30m", 1.0, 30.0)]:
        r = f1_farfield_certificate(lam, D, R)
        f1_rows.append({"scenario": name, **r})
        print(f"  {name:>18}: D*={r['D_star_m']:>7.2f}m  二次相位={r['closed_form_quad_phase_rad']:.2e} rad "
              f"({'≤' if r['farfield_valid'] else '>'}π/8)  闭式准确={r['closed_form_accurate']}")

    print("\nF2 近场测距 CRB（闭式 vs MC，ULA-N）:")
    f2_rows = []
    for name, N, d_el, R in [("XL 孔径1m N=200 @100m", 200, 1.0 / 199, 100.0),
                             ("XL 孔径1m N=200 @30m", 200, 1.0 / 199, 30.0),
                             ("面板0.5m N=100 @100m", 100, 0.5 / 99, 100.0)]:
        r = f2_nearfield_crb(lam, N, d_el, R, y_off=1.0, gamma_db=args.gamma_db,
                             n_mc=args.n_mc)
        f2_rows.append({"scenario": name, **r})
        print(f"  {name:>16}: 孔径={r['aperture_m']:>6.2f}m  CRB(R)={r['crb_R_m2'] * 1e3:>8.2f}mm² "
              f"(闭式 {r['crb_closed_R_m2'] * 1e3:.2f})  MC/CRB={r['mc_over_crb_R']:.2f}")

    print("\nF3 近场窗口（R/R_F 扫描，N=16 λ/2）:")
    f3 = f3_rayleigh_window(lam, 200, 1.0 / 199, args.gamma_db, n_mc=args.n_mc)
    for p in f3["points"]:
        print(f"  R/R_F={p['R_over_RF']:>5.2f} (R={p['R_m']:>7.1f}m): FIM 条件数={p['fim_cond']:>10.1f} "
              f"MC RMSE={p['mc_rmse_m']:>8.3f}m")

    out = {"f1": f1_rows, "f2": f2_rows, "f3": f3,
           "protocol": {"fc_hz": args.fc_hz, "gamma_db": args.gamma_db,
                        "n_mc": args.n_mc}}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "near_field_crb.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="近场 XL-RIS CRB 验证")
    parser.add_argument("--fc_hz", type=float, default=30e9)
    parser.add_argument("--gamma_db", type=float, default=20.0)
    parser.add_argument("--n_mc", type=int, default=300)
    args = parser.parse_args()
    main(args)
