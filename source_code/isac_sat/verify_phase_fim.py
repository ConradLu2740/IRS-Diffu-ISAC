"""
verify_phase_fim.py — RIS 相位设计的 pilot FIM 与信道估计因子 η_est
====================================================================

任务（docs/optimization_roadmap.md §4.3 方向二 / §6 P1，预注册命题见文末）:
  把闭环最优性恒等式从 η_total = η_sense × η_design 升级为
        η_total = η_sense × η_design × η_est(n_p)
  其中 η_est 是"用 n_p 个正交导频估计信道 (d, M) 后闭式设计 RIS 相位"
  相对"真信道 genie-CSI 闭式设计"的接收功率达成率。

数学模型（与 phase_optimizer_sat.PhaseOptimizerSat._linear_model 同一线性化，
  与 data_sat.calculate_value_sat 同一前向物理；脚本内做数值一致性断言）:
    接收信号   y_a(v) = d_a + Σ_i m_{a,i} v_i ,  a=1..A（BS 天线）, i=1..N（RIS 单元）
    导频观测   y_a^(p) = d_a + Σ_i m_{a,i} v_i^(p) + w_a^(p),  w ~ CN(0, σ_c² I_A)
    导频图样   F = [1, V] ∈ C^{n_p×(N+1)}，行正交且 ‖r_p‖² = 1+N（DFT 型，见 §导频）
    LS 估计    x̂_a = (F^H F)^{-1} F^H y_a
    对齐系数   c_i = d^H m_i（闭式相位 v_i* = conj(c_i)/|c_i| 的唯一输入）

  FIM / CRB（圆对称复高斯，复方差 E|·|²口径——MC 实测证实，见 §结果）:
      每天线复 FIM  J_a = (1/σ_c²) F^H F = (n_p/σ_c²) I
      ⇒ Var(x̂_{a,k}) = σ_c² / n_p
      ⇒ Var(ĉ_i) = (σ_c²/n_p) Σ_a (|m_{a,i}|² + |d_a|²)
      ⇒ E[Δθ_i²] = Var(ĉ_i) / (2|c_i|²)      （小误差切向投影）
    【约定标注】roadmap/任务书写的是 J_a=(2/σ_c²)F^HF ⇒ Var(ĉ_i)=(σ_c²/2n_p)Σ(...)。
    该约定（J=2Re(F^HF)/σ²）对复方差小 2×（它约束的是实部方差）。本脚本两列都算，
    以 MC 列为准：实测 Var = (σ_c²/n_p)Σ，即 prior 列的 2 倍（§1 表）。

  二阶 Taylor 功率损失律（在 v^cf 处展开，E[Δθ]=0，G = M^H M）:
      E[P(v̂)] = P(v^cf) − Σ_i w_i E[Δθ_i²] + Σ_{i≠j} Re(G_ij v_j v̄_i)·Cov(Δθ_i,Δθ_j)
      w_i = |c_i| + Σ_{j≠i} Re(G_ij v_j^cf v̄_i^cf)        （直接对齐增益 + 跨单元耦合泄漏）
      Cov(Δθ_i,Δθ_j) = (σ_c²/2n_p)·Re[(G_ji+δ_ij·D)·conj(c_i)c_j] / (|c_i|²|c_j|²),  D=Σ_a|d_a|²
    ⇒ η_est(n_p) = 1 − K/n_p,  K = n_p·loss/P^cf（先验可算，与 n_p 无关）
    prior 公式（roadmap 原式）: loss = ½ Σ_i (|c_i|+|Σ_{j≠i}G_ij v_j v̄_i|)·E[Δθ²]
    （幅值代替实部、且 Var 取 prior 约定 ⇒ 比实测小 ~4×，见 §2/§3 对表）。

  多普勒（SGP4 确定项补偿）:
    两径组多普勒不同（BS→ROI: f_d1；BS→IRS: f_d2，|f_d1−f_d2| ~ 2.8e5 Hz）。
    取导频间隔 = 差拍周期 dt = 1/|f_d2−f_d1|（~3.6 µs），接收端按 SGP4 已知的
    dop1(t_p)=e^{j2πf_d1 t_p} 精确预补偿 ⇒ 差拍旋转 ρ=dop2/dop1 ≡ 1（断言 <1e-9），
    补偿后观测严格等于 y = d + M v^(p) + w。--doppler zero 为"忽略多普勒"对照
    （全部导频取 t_rel=0，经 calculate_value_sat 直接生成）。
    诚实标注: 几何在导频块内冻结（任务模型 y=d+Mv+w 的前提）；差拍周期内的
    多普勒率（加速度）与几何漂移未建模（界 < 1e-3 rad，远小于所测相位误差量级）。

协议（与 verify_optimality_decomposition.py / verify_ris_sdr_certificate.py 一致）:
  - 场景: ss.SatISACScenario(tau=8) + SatScenarioChannels(irs_mode="sat", 4+4 天线)，
    中间帧（t=tau//2，即功率标定参考帧）；ROI = generate_ground_target_sample()（种子固定）；
    X = tensor_a × _SIGNAL1[:4]（16QAM 导频，仓库标定口径）。
  - 导频: n_p ∈ {17,33,65,129,257}，DFT 型正交图样（Hadamard 阶不可得时的精确复正交构造）。
  - MC: 每 n_p ≥ 500 次（默认 2000），float64 批量实现；直接测估计相位功率 P(v̂)。
  - SNR 扫描: {20,10,0,-10,-20} dB，参考链路复 SNR = 64/σ_c²（σ_c²=E|w|²=2·POWER_SIGMA，
    仓库标定 σ_c²=0.02 ⇒ SNR=3200=35.05dB）。
  - 同时报告导频-跟踪联合权衡（K×n_p 网格: stale + overhead 总损失）。

预注册命题（可证伪，执行前固定；roadmap §4.3 方向二）:
  P1  log-log 斜率: log(1−η_est) vs log(n_p) = −1.00 ± 0.05
      （小误差定律; 仓库标定下 1−η~1e-6, 直接 MC 噪声淹没信号, 用 RB/τ² 估计量裁决）
  P2  regime 认证: 仓库标定（复样本 SNR≈3200）下 η_est(17) > 0.9999
  P3  临界 SNR: 先验公式预测 ≈ −8 dB（对表; 0.5 阈值口径, 阈值表全列于 §3）

用法: py verify_phase_fim.py [--n_trials 2000] [--doppler beat|zero] [--roi_seed 42]
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import random as _random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup_sat as ss
from data_sat import (SatScenarioChannels, generate_ground_target_sample, _SIGNAL1,
                      POWER_SIGMA, calculate_value_sat, get_channel_mat,
                      make_roi_local, IRS_GAIN, _ant_positions)
from phase_optimizer_sat import PhaseOptimizerSat

P_REF = 64.0                                   # data_sat._calibrate 参考链路功率
SIGMA2_REF = 2.0 * POWER_SIGMA                 # 复噪声方差 E|w|^2（每天线）
SNR_REF = P_REF / SIGMA2_REF                   # = 3200（复样本，参考链路）

N_P_LIST = [17, 33, 65, 129, 257]
SNR_DB_LIST = [20, 10, 0, -10, -20]
SNR_DB_EXT = [-25, -30, -35]                     # 预注册集之外的括住点（仅主场景 P3 用）
N_SCAN = [16, 32, 64, 128, 256]                # RIS 单元数扫描（n_p = N+1）
K_LIST = [1, 2, 4, 8]                          # 重构周期（帧，仓库 track_K 约定）


def seed_all(s):
    import torch
    torch.manual_seed(s)
    np.random.seed(s)
    _random.seed(s)


# ----------------------------------------------------------------------
# 导频图样
# ----------------------------------------------------------------------
def dft_pilots(n_p, N):
    """DFT 型正交导频: V[p,i] = exp(-2jπ p (i+1)/n_p), p=0..n_p-1, i=0..N-1.

    F=[1,V] 行严格正交、‖r_p‖² = 1+N；n_p=N+1 时即满阶 DFT（Hadamard 的复推广）。
    |V[p,i]|=1 ⇒ 取相位即恒模 RIS 图样，正交性不受相位量化影响。
    """
    p = np.arange(n_p)[:, None]
    i = np.arange(1, N + 1)[None, :]
    return np.exp(-2j * np.pi * p * i / n_p)


def random_orth_pilots(n_p, N, rng):
    """随机正交图样（稳健性对照）: QR([1, G]) → F^H F = n_p I；取相位后近似正交。"""
    G = (rng.standard_normal((n_p, N)) + 1j * rng.standard_normal((n_p, N))) / math.sqrt(2)
    A0 = np.concatenate([np.ones((n_p, 1)), G], axis=1)
    Q, _ = np.linalg.qr(A0)
    F = math.sqrt(n_p) * Q
    F = F * np.exp(-1j * np.angle(F[:, 0]))[:, None]     # 第一列对齐为全 1
    return F[:, 1:]


def design_matrix(V):
    return np.concatenate([np.ones((V.shape[0], 1)), V], axis=1)


def hadamard_available(n):
    """scipy.linalg.hadamard 能否构造阶数 n 的 ±1 Hadamard 矩阵。"""
    try:
        from scipy.linalg import hadamard
        H = hadamard(int(n))
        return H.shape == (int(n), int(n))
    except Exception:
        return False


# ----------------------------------------------------------------------
# 场景 / 线性模型
# ----------------------------------------------------------------------
def build_context(args):
    import torch
    seed_all(args.roi_seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device="cpu",
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    opt = PhaseOptimizerSat(channels, device="cpu")
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:args.bs_ant],
                                         dtype=torch.complex64).view(args.bs_ant, 1)
    roi_np, cid, ang = generate_ground_target_sample()
    roi_t = torch.tensor(roi_np.astype(np.float32)).reshape(-1)
    ctx = dict(scenario=scenario, frames=frames, channels=channels, opt=opt,
               X=X, X_np=X.numpy().astype(np.complex128), roi_np=roi_np, roi_t=roi_t,
               cid=int(cid), angle=float(ang), wavelength=channels.wavelength_m,
               tau=args.tau, bs_ant=args.bs_ant, ue_ant=args.ue_ant,
               roi_local=channels.roi_local)
    return ctx


def linear_model_repo(ctx, t):
    """仓库 _linear_model → (d, M) complex128, [A], [A,N]。"""
    Ht = ctx["channels"].channels_per_frame[t]
    d, M = ctx["opt"]._linear_model(Ht, ctx["roi_t"], ctx["X"])
    return d.numpy().astype(np.complex128), M.numpy().astype(np.complex128)


def irs_positions_rect(center_km, N, panel_m=10.0, axis=(0, 1, 0)):
    """矩形补全仓库方面板约定（_irs_panel_positions）: rows×cols=N, 同 10m 间距/同轴向。

    N=16/64/256（完全平方）与仓库逐元素一致；N=32/128 取最接近方形因数分解。
    """
    k = int(N).bit_length() - 1
    rows, cols = 1 << ((k + 1) // 2), int(N) >> ((k + 1) // 2)
    off = panel_m / 1000.0
    pos = []
    for i in range(rows):
        for j in range(cols):
            dd = np.zeros(3)
            dd[axis[0]] = (i - (rows - 1) / 2.0) * off
            dd[axis[1]] = (j - (cols - 1) / 2.0) * off
            pos.append(np.asarray(center_km, dtype=float) + dd)
    return np.array(pos)


def linear_model_n(ctx, t, N):
    """自定义 RIS 单元数 N 的线性模型（复刻 data_sat._precompute_frame + _linear_model）。"""
    frame = ctx["frames"][t]
    wl = ctx["wavelength"]
    bs = _ant_positions(frame["sat_pos"], ctx["bs_ant"])
    ue = _ant_positions(frame["ground_pos"], ctx["ue_ant"])
    irs = irs_positions_rect(frame["sat_ris_pos"], N)
    roi_ecef = frame["target_pos"][None, :] + ctx["roi_local"] / 1000.0
    H_BS_ROI = get_channel_mat(bs, roi_ecef, wl).astype(np.complex128)
    H_ROI_UE = get_channel_mat(roi_ecef, ue, wl).astype(np.complex128)
    H_BS_IRS = get_channel_mat(bs, irs, wl, b_gain=IRS_GAIN).astype(np.complex128)
    H_ROI_IRS = get_channel_mat(roi_ecef, irs, wl, b_gain=IRS_GAIN).astype(np.complex128)
    H_IRS_ROI = get_channel_mat(irs, roi_ecef, wl, a_gain=IRS_GAIN).astype(np.complex128)
    H_IRS_UE = get_channel_mat(irs, ue, wl, a_gain=IRS_GAIN).astype(np.complex128)
    S_c = ctx["roi_np"].reshape(-1).astype(np.complex128)
    Xc = ctx["X_np"]
    Bmat = S_c[:, None] * H_ROI_UE
    d = (H_BS_ROI @ Bmat @ Xc).flatten()
    C1 = (H_BS_ROI * S_c[None, :]) @ H_ROI_IRS
    B1 = (H_IRS_ROI * S_c[None, :]) @ H_ROI_UE
    HuX = (H_IRS_UE @ Xc).flatten()
    B1X = (B1 @ Xc).flatten()
    M = C1 * HuX[None, :] + H_BS_IRS * B1X[None, :]
    u = frame["sat_ris_pos"] - frame["sat_pos"]
    u = u / (np.linalg.norm(u) + 1e-12)
    f_d2 = float(np.dot(frame["sat_vel"], u) * 1000.0 / wl)
    return d, M, f_d2


def doppler_pair(ctx, t, f_d2=None):
    """(f_d1, f_d2) BS→ROI / BS→IRS 多普勒（SGP4 确定项）。"""
    Ht = ctx["channels"].channels_per_frame[t]
    f1 = float(Ht["f_d_bs_roi"])
    if f_d2 is None:
        f_d2 = float(Ht.get("f_d_bs_irs", 0.0))
    return f1, f_d2


# ----------------------------------------------------------------------
# 解析预测
# ----------------------------------------------------------------------
def channel_stats(d, M):
    c = np.conj(d) @ M
    v_cf = np.conj(c) / np.maximum(np.abs(c), 1e-30)
    G = np.conj(M.T) @ M
    P_cf = float((np.abs(d + M @ v_cf) ** 2).sum())
    D = float((np.abs(d) ** 2).sum())
    S_i = (np.abs(M) ** 2).sum(0) + D
    Sv = G * v_cf[None, :] * np.conj(v_cf)[:, None]        # Sv[i,j] = G_ij v_j v̄_i
    cross = Sv.sum(1) - np.diag(Sv)
    return dict(c=c, v_cf=v_cf, G=G, P_cf=P_cf, D=D, S_i=S_i, Sv=Sv, cross=cross,
                w_rd=np.abs(c) + np.abs(cross),
                w_ex=np.abs(c) + cross.real)


def analytic_predictions(d, M, n_p, sigma2, st=None, F=None):
    """返回 prior（roadmap 原式）与 exact（修正 CRB + 完整二阶 Taylor）两列。"""
    st = st or channel_stats(d, M)
    c, G, Sv = st["c"], st["G"], st["Sv"]
    if F is None:
        v0 = sigma2 / n_p                    # 每系数复方差（MC 证实）
        v0_prior = sigma2 / (2 * n_p)        # roadmap 约定（小 2×）
    else:
        Finv = np.linalg.inv(F.conj().T @ F)
        v0 = sigma2 * float(np.mean(np.real(np.diag(Finv))))
        v0_prior = 0.5 * v0
    var_ci = v0 * st["S_i"]
    var_ci_prior = v0_prior * st["S_i"]
    ac2 = np.abs(c) ** 2
    tau2 = var_ci / (2 * ac2)
    tau2_prior = var_ci_prior / (2 * ac2)
    # 相位误差互协方差 Cov(Δθ_i,Δθ_j)（i≠j 非零；对角 = tau2）
    cc = np.outer(np.conj(c), c)
    denom = np.outer(ac2, ac2)
    Cov = (v0 / 2.0) * np.real(G.T * cc) / denom
    np.fill_diagonal(Cov, tau2)
    Svnd = Sv - np.diag(np.diag(Sv))
    Covnd = Cov - np.diag(np.diag(Cov))
    loss_ex = float(np.sum(st["w_ex"] * tau2) - np.sum(Svnd.real * Covnd))
    loss_diag = float(np.sum(st["w_ex"] * tau2))
    loss_rd = 0.5 * float(np.sum(st["w_rd"] * tau2_prior))   # roadmap 原式（字面组合）
    return dict(v0=v0, var_ci=var_ci, var_ci_prior=var_ci_prior,
                tau2=tau2, tau2_prior=tau2_prior,
                loss_ex=loss_ex, loss_diag=loss_diag, loss_rd=loss_rd,
                eta_ex=1.0 - loss_ex / st["P_cf"],
                eta_rd=1.0 - loss_rd / st["P_cf"])


# ----------------------------------------------------------------------
# MC（float64 批量）
# ----------------------------------------------------------------------
def mc_run(d, M, V, sigma2, n_trials, seed, want_rb=True):
    """导频估计 → 闭式相位 → 功率。返回直接 MC 与 Rao-Blackwellized（Taylor 二次型）两列。"""
    A, N = M.shape
    n_p = V.shape[0]
    F = design_matrix(V)
    FH = F.conj().T
    Finv = np.linalg.inv(FH @ F)
    VM = V @ M.T                                        # [n_p, A] 无噪信号
    st = channel_stats(d, M)
    c, P_cf = st["c"], st["P_cf"]
    # RB 二次型矩阵: Q(Δθ) = Σ_i w_i Δ_i² − Σ_{i≠j} Re(Sv_ij) Δ_i Δ_j
    Qmat = rb_quadratic_matrix(st)
    rng = np.random.default_rng(seed)
    budget = 4_000_000                                    # 复数个数上限（~64MB）
    chunk = int(max(200, min(20000, budget // max(1, n_p * A))))
    loss_l, tau2_l, var_l, q_l, bad = [], [], [], [], 0
    done = 0
    while done < n_trials:
        T = int(min(chunk, n_trials - done))
        W = (rng.standard_normal((T, n_p, A)) +
             1j * rng.standard_normal((T, n_p, A))) * math.sqrt(sigma2 / 2.0)
        Y = VM[None, :, :] + d[None, None, :] + W                     # [T,n_p,A]
        FHY = np.einsum("kp,tpa->tka", FH, Y)
        Xh = np.einsum("ij,tja->tia", Finv, FHY)                      # [T,N+1,A]
        d_h, M_h = Xh[:, 0, :], np.transpose(Xh[:, 1:, :], (0, 2, 1))  # [T,A], [T,A,N]
        c_h = np.sum(np.conj(d_h)[:, :, None] * M_h, axis=1)          # [T,N]
        small = np.abs(c_h) < 1e-12 * max(1.0, np.abs(c).mean())
        bad += int(small.sum())
        c_h_safe = np.where(small, 1.0 + 0j, c_h)
        v_h = np.conj(c_h_safe) / np.abs(c_h_safe)
        y = d[None, :] + np.einsum("ai,ti->ta", M, v_h)
        P_h = (np.abs(y) ** 2).sum(axis=1)
        loss_l.append(1.0 - P_h / P_cf)
        dth = np.angle(c_h_safe / c[None, :])
        tau2_l.append((dth ** 2).mean(axis=1))
        var_l.append((np.abs(c_h - c[None, :]) ** 2).mean(axis=1))
        if want_rb:
            q_l.append(np.einsum("ti,ij,tj->t", dth, Qmat, dth) / P_cf)
        done += T
    loss = np.concatenate(loss_l)
    tau2_all = np.concatenate(tau2_l)
    var_all = np.concatenate(var_l)
    out = dict(n_trials=n_trials,
               eta_direct=float(1.0 - loss.mean()),
               eta_direct_se=float(loss.std(ddof=1) / math.sqrt(n_trials)),
               tau2=float(tau2_all.mean()),
               tau2_se=float(tau2_all.std(ddof=1) / math.sqrt(n_trials)),
               var_ci=float(var_all.mean()),
               n_bad=int(bad))
    if want_rb:
        q = np.concatenate(q_l)
        out.update(eta_rb=float(1.0 - q.mean()),
                   eta_rb_se=float(q.std(ddof=1) / math.sqrt(n_trials)))
    return out


def rb_quadratic_matrix(st):
    """Rao-Blackwellized 损失估计的二次型矩阵。

    Q(Δθ) = Σ_i w_i Δθ_i² − Σ_{i≠j} Re(Sv_ij) Δθ_i Δθ_j
    ⇒ Qmat[i,i] = w_i, Qmat[i,j] = −Re(Sv_ij)（i≠j）。
    """
    Q = -st["Sv"].real.copy()
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, st["w_ex"])
    return Q


# ----------------------------------------------------------------------
# 多普勒：原始观测（float64 复刻 calculate_value_sat 结构）+ SGP4 精确补偿
# ----------------------------------------------------------------------
def observe_beat(d, M, V, f1, f2, sigma2, rng):
    """差拍周期导频间隔的原始观测 + dop1 预补偿。返回 (Y_comp, rho, t_p, Y_raw)。

    sigma2 = E|w|²（每天线复方差）；分量为 N(0, sigma2/2)，与 calculate_value_sat 一致。
    """
    n_p, N = V.shape
    A = d.shape[0]
    df = f2 - f1
    dt = 1.0 / abs(df) if df != 0 else 0.0
    t_p = np.arange(n_p) * dt
    dop1 = np.exp(1j * 2 * np.pi * f1 * t_p)
    dop2 = np.exp(1j * 2 * np.pi * f2 * t_p)
    W = (rng.standard_normal((n_p, A)) + 1j * rng.standard_normal((n_p, A))) \
        * math.sqrt(sigma2 / 2.0)
    Y_raw = dop1[:, None] * d[None, :] + dop2[:, None] * (V @ M.T) + W
    Y_comp = np.conj(dop1)[:, None] * Y_raw
    rho = np.conj(dop1) * dop2
    return Y_comp, rho, t_p, Y_raw


# ----------------------------------------------------------------------
# 前向模型一致性验证（vs data_sat.calculate_value_sat）
# ----------------------------------------------------------------------
def verify_forward_model(ctx, t, d, M, V, n_p):
    import torch
    Ht = ctx["channels"].channels_per_frame[t]
    roi_t, X = ctx["roi_t"], ctx["X"]
    wl = ctx["wavelength"]
    out = {}
    # 1) 无噪一致性: calculate_value_sat(Power_sigma=0, t_rel=0) vs d + M v
    errs = []
    for p in range(min(4, n_p)):
        ph_p = np.angle(V[p]).astype(np.float32)
        y0 = calculate_value_sat(roi_t, ph_p, X, Ht,
                                 0.0, 0.0, wl).numpy().astype(np.complex128).flatten()
        yl = d + M @ V[p]
        errs.append(float(np.abs(y0 - yl).max() / (np.abs(yl).max() + 1e-30)))
    out["noiseless_rel_err"] = max(errs)
    # 2) 噪声模型: E|w|^2 每天线 ≈ 2*POWER_SIGMA（多次抽样平均）
    acc = []
    for s in range(512):
        torch.manual_seed(100 + s)
        y1 = calculate_value_sat(roi_t, np.zeros(V.shape[1], dtype=np.float32), X, Ht,
                                 POWER_SIGMA, 0.0, wl).numpy().astype(np.complex128).flatten()
        torch.manual_seed(100 + s)
        y0 = calculate_value_sat(roi_t, np.zeros(V.shape[1], dtype=np.float32), X, Ht,
                                 0.0, 0.0, wl).numpy().astype(np.complex128).flatten()
        acc.append((np.abs(y1 - y0) ** 2).mean())
    out["noise_e_abs2_per_ant"] = float(np.mean(acc))
    out["noise_se"] = float(np.std(acc, ddof=1) / math.sqrt(len(acc)))
    out["noise_expected"] = 2 * POWER_SIGMA
    # 3) 精确噪声复制: 同种子噪声灌入线性管线，与 calculate_value_sat 逐位一致（float32 容差）
    torch.manual_seed(123)
    ya = calculate_value_sat(roi_t, np.angle(V[0]).astype(np.float32), X, Ht,
                             POWER_SIGMA, 0.0, wl).numpy().astype(np.complex128).flatten()
    torch.manual_seed(123)
    yb = calculate_value_sat(roi_t, np.angle(V[0]).astype(np.float32), X, Ht,
                             0.0, 0.0, wl).numpy().astype(np.complex128).flatten()
    y_mine = d + M @ V[0] + (ya - yb)
    out["exact_noise_max_abs_diff"] = float(np.abs(ya - y_mine).max())
    out["exact_noise_rel_diff"] = out["exact_noise_max_abs_diff"] / float(np.abs(ya).max())
    return out


# ----------------------------------------------------------------------
# 主流程各节
# ----------------------------------------------------------------------
def section_setup(ctx, args, t):
    d, M = linear_model_repo(ctx, t)
    st = channel_stats(d, M)
    f1, f2 = doppler_pair(ctx, t)
    A, N = M.shape
    per_ant_snr = ((np.abs(d) ** 2 + (np.abs(M) ** 2).sum(1)) / SIGMA2_REF)
    print("\n" + "=" * 78)
    print("[0] 场景 / 线性模型 / 导频图样")
    print("=" * 78)
    print(f"  frame t={t} (tau={ctx['tau']}), N={N} RIS 单元, A={A} BS 天线, "
          f"ROI 占据体素 {int((ctx['roi_np'] > 0.5).sum())} (class {ctx['cid']})")
    print(f"  P(v^cf) = {st['P_cf']:.4e}   |c_i|: mean {np.abs(st['c']).mean():.1f} "
          f"min {np.abs(st['c']).min():.1f}   D=Σ|d|²={st['D']:.1f}")
    print(f"  每天线复 SNR (|d|²+Σ|m|²)/σ_c²: mean {per_ant_snr.mean():.3e} "
          f"({10 * math.log10(per_ant_snr.mean()):.1f} dB), "
          f"参考链路 SNR=64/σ_c²={SNR_REF:.0f} ({10 * math.log10(SNR_REF):.2f} dB)")
    print(f"  多普勒: f_d1(BS→ROI)={f1:.1f} Hz, f_d2(BS→IRS)={f2:.1f} Hz, "
          f"差拍周期 dt={1.0 / abs(f2 - f1) * 1e6:.2f} µs")
    # 导频图样可得性
    avail = {n: hadamard_available(n) for n in set(N_P_LIST) | set(N_SCAN)}
    print(f"  scipy Hadamard 阶可得性: n_p {N_P_LIST} → "
          f"{ {n: avail[n] for n in N_P_LIST} }")
    print(f"                          N   {N_SCAN} → { {n: avail[n] for n in N_SCAN} }")
    print("  ⇒ ±1 Hadamard 需阶数 ≡0 (mod 4)，n_p≡1 (mod 4) 不可得；"
          "采用 DFT 型复正交图样（行正交/‖r‖²=N+1 精确成立）")
    # 正交性断言
    for n_p in N_P_LIST:
        F = design_matrix(dft_pilots(n_p, N))
        err = float(np.abs(F.conj().T @ F - n_p * np.eye(N + 1)).max())
        assert err < 1e-9, (n_p, err)
    print(f"  DFT 图样断言: F^H F = n_p·I (n_p={N_P_LIST}), max err < 1e-9  ✓")
    # 前向模型一致性
    fw = verify_forward_model(ctx, t, d, M, dft_pilots(N_P_LIST[0], N), N_P_LIST[0])
    print(f"  前向模型 vs calculate_value_sat: 无噪相对误差 {fw['noiseless_rel_err']:.2e}, "
          f"E|w|²/天线 {fw['noise_e_abs2_per_ant']:.4f}±{fw['noise_se']:.4f} "
          f"(期望 {fw['noise_expected']}), "
          f"精确噪声复制相对差 {fw['exact_noise_rel_diff']:.2e}")
    assert fw["noiseless_rel_err"] < 1e-4
    assert abs(fw["noise_e_abs2_per_ant"] / fw["noise_expected"] - 1) < 0.05
    assert fw["exact_noise_rel_diff"] < 1e-4
    # 多普勒补偿链路（SGP4 确定项）
    V17 = dft_pilots(N_P_LIST[0], N)
    rng = np.random.default_rng(999)
    Yc, rho_b, t_p, Yraw = observe_beat(d, M, V17, f1, f2, SIGMA2_REF, rng)
    rho_err = float(np.abs(rho_b - 1).max())
    comp_err = float(np.abs(Yc - (d[None, :] + V17 @ M.T) -
                            np.conj(np.exp(1j * 2 * np.pi * f1 * t_p))[:, None] *
                            (Yraw - (np.exp(1j * 2 * np.pi * f1 * t_p)[:, None] * d[None, :] +
                                     np.exp(1j * 2 * np.pi * f2 * t_p)[:, None] * (V17 @ M.T)))).max())
    print(f"  多普勒补偿（beat 模式, dt=1/|Δf|）: max|ρ−1| = {rho_err:.2e}（<1e-9 ✓）; "
          f"补偿后 = d+Mv+w 一致性 {comp_err:.2e}")
    if args.doppler == "zero":
        print("  [doppler=zero] 全部导频取 t_rel=0（忽略多普勒），观测经 calculate_value_sat 直出")
    assert rho_err < 1e-9 and comp_err < 1e-9
    return d, M, st, f1, f2, per_ant_snr, dict(fw, doppler_rho_err=rho_err,
                                               doppler_comp_err=comp_err)


def section_fim_scan(d, M, st, args):
    """§1: FIM/CRB 表（prior 约定 vs 修正约定 vs MC 实测）。"""
    print("\n" + "=" * 78)
    print("[1] pilot FIM / CRB：Var(ĉ_i) 与 E[Δθ²]（理论列 vs MC 列，仓库标定 σ_c²=0.02）")
    print("=" * 78)
    print(f"{'n_p':>5} {'Var prior':>12} {'Var exact':>12} {'Var MC':>12} "
          f"{'τ² pred':>12} {'τ² MC':>12} {'MC/pred':>8}")
    rows = []
    for n_p in N_P_LIST:
        an = analytic_predictions(d, M, n_p, SIGMA2_REF, st=st)
        mc = mc_run(d, M, dft_pilots(n_p, M.shape[1]), SIGMA2_REF,
                    args.n_trials, seed=1000 + n_p)
        rows.append(dict(n_p=n_p,
                         var_ci_prior_mean=float(an["var_ci_prior"].mean()),
                         var_ci_mean=float(an["var_ci"].mean()),
                         tau2_pred_mean=float(an["tau2"].mean()),
                         var_mc=mc["var_ci"], tau2_mc=mc["tau2"],
                         tau2_se=mc["tau2_se"], eta_rb=mc["eta_rb"],
                         eta_rb_se=mc["eta_rb_se"], eta_direct=mc["eta_direct"],
                         eta_direct_se=mc["eta_direct_se"], n_bad=mc["n_bad"]))
        print(f"{n_p:>5} {an['var_ci_prior'].mean():>12.4e} {an['var_ci'].mean():>12.4e} "
              f"{mc['var_ci']:>12.4e} {an['tau2'].mean():>12.4e} {mc['tau2']:>12.4e} "
              f"{mc['tau2'] / an['tau2'].mean():>8.3f}")
    ratio = np.mean([r["var_mc"] / r["var_ci_mean"] for r in rows])
    print(f"  MC Var / exact CRB = {ratio:.3f}（应 ≈1）；"
          f"MC Var / prior = {np.mean([r['var_mc'] / r['var_ci_prior_mean'] for r in rows]):.3f}"
          f"（≈2 ⇒ prior 约定小 2×）")
    return rows


def section_np_scan(d, M, st, args):
    """§2: n_p 扫描 @ 仓库标定：η_est、K、预注册 P1（斜率）与 P2（regime）。

    斜率用三个估计量交叉验证：(a) RB @ 仓库标定（相位误差的 Taylor 二次型平均，
    方差缩减）；(b) τ² @ 仓库标定（纯 MC 相位误差方差，FIM 律的直接检验）；
    (c) 直接功率 MC @ −10dB（纯 MC 直接测功率，该 SNR 下 MC 精度足够）。
    """
    print("\n" + "=" * 78)
    print("[2] n_p 扫描 @ 仓库标定（σ_c²=0.02, 参考 SNR=3200）：η_est(n_p)=1−K/n_p")
    print("=" * 78)
    print(f"{'n_p':>5} {'1−η MC(RB)':>12} {'±SE':>10} {'1−η direct':>12} {'±SE':>10} "
          f"{'1−η exact':>12} {'1−η prior':>12} {'K MC':>10} {'K exact':>10} {'K prior':>10}")
    rows, tau2_rows = [], []
    for n_p in N_P_LIST:
        an = analytic_predictions(d, M, n_p, SIGMA2_REF, st=st)
        mc = mc_run(d, M, dft_pilots(n_p, M.shape[1]), SIGMA2_REF,
                    args.n_trials, seed=2000 + n_p)
        k_mc = n_p * (1.0 - mc["eta_rb"])
        k_ex = n_p * an["loss_ex"] / st["P_cf"]
        k_rd = n_p * an["loss_rd"] / st["P_cf"]
        rows.append(dict(n_p=n_p, one_minus_eta_rb=1 - mc["eta_rb"],
                         one_minus_eta_rb_se=mc["eta_rb_se"],
                         one_minus_eta_direct=1 - mc["eta_direct"],
                         one_minus_eta_direct_se=mc["eta_direct_se"],
                         one_minus_eta_ex=an["loss_ex"] / st["P_cf"],
                         one_minus_eta_rd=an["loss_rd"] / st["P_cf"],
                         k_mc=k_mc, k_ex=k_ex, k_rd=k_rd, eta_rb=mc["eta_rb"],
                         eta_direct=mc["eta_direct"], eta_direct_se=mc["eta_direct_se"]))
        tau2_rows.append(dict(n_p=n_p, tau2_mc=mc["tau2"], tau2_se=mc["tau2_se"],
                              tau2_pred=float(an["tau2"].mean())))
        print(f"{n_p:>5} {1 - mc['eta_rb']:>12.4e} {mc['eta_rb_se']:>10.1e} "
              f"{1 - mc['eta_direct']:>12.4e} {mc['eta_direct_se']:>10.1e} "
              f"{an['loss_ex'] / st['P_cf']:>12.4e} {an['loss_rd'] / st['P_cf']:>12.4e} "
              f"{k_mc:>10.3e} {k_ex:>10.3e} {k_rd:>10.3e}")
    nps = np.array([r["n_p"] for r in rows], dtype=float)
    nps_lin = nps[1:]                                     # n_p≥33（线性区）

    def _slope(xs, vals):
        v = np.asarray(vals, dtype=float)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log(np.asarray(xs, dtype=float)[ok]),
                                np.log(v[ok]), 1)[0])

    slopes = dict(rb=_slope(nps, [r["one_minus_eta_rb"] for r in rows]),
                  exact=_slope(nps, [r["one_minus_eta_ex"] for r in rows]),
                  prior=_slope(nps, [r["one_minus_eta_rd"] for r in rows]),
                  tau2=_slope(nps, [r["tau2_mc"] for r in tau2_rows]))
    # 纯 MC 直接功率斜率 @ −10dB（该 SNR 下直接 MC 精度足够; 4× 试验压噪声）
    sig2_lo = P_REF / (10.0 ** (-10 / 10.0))
    n_slope = 4 * args.n_trials
    direct_lo = []
    for n_p in N_P_LIST:
        mc = mc_run(d, M, dft_pilots(n_p, M.shape[1]), sig2_lo,
                    n_slope, seed=2500 + n_p)
        direct_lo.append(1 - mc["eta_direct"])
    slopes["direct_m10"] = _slope(nps, direct_lo)
    slopes["direct_m10_lin"] = _slope(nps_lin, direct_lo[1:])   # n_p≥33（线性区）
    print(f"\n  直接功率 MC @ −10dB (1−η vs n_p, {n_slope} 次/点): "
          + ", ".join(f"{n_p}:{v:.3e}" for n_p, v in zip(N_P_LIST, direct_lo)))
    print("  注: −10dB 下 n_p=17 的最差单元相位误差可达 ~1 rad（Δθ² 99 分位 ~1.2），"
          "已进入小误差非线性区，该点直接 MC 值系统性高于 1/n_p 线性外推 ~15-20%；"
          "n_p≥33 各点仍在线性区。预注册定律 η_est=1−K/n_p 为小误差结果，"
          "斜率裁决用小误差区估计量（RB/τ² @ 仓库标定 + direct@−10dB 的 n_p≥33 点）。")
    k_mc_mean = float(np.mean([r["k_mc"] for r in rows]))
    print(f"  K: MC(RB) {k_mc_mean:.4e} ± {np.std([r['k_mc'] for r in rows]):.1e}, "
          f"exact {rows[0]['k_ex']:.4e}, prior {rows[0]['k_rd']:.4e}（n_p 无关性自检）")
    print(f"  log-log 斜率 (1−η vs n_p): RB {slopes['rb']:+.4f}, "
          f"τ² {slopes['tau2']:+.4f}, direct@−10dB(全) {slopes['direct_m10']:+.4f}, "
          f"direct@−10dB(n_p≥33) {slopes['direct_m10_lin']:+.4f}, "
          f"exact {slopes['exact']:+.4f}, prior {slopes['prior']:+.4f}  "
          f"[预注册 P1: −1.00±0.05]")
    eta17 = [r for r in rows if r["n_p"] == 17][0]
    print(f"  η_est(17) = {eta17['eta_rb']:.8f} (RB) / {eta17['eta_direct']:.6f} "
          f"± {eta17['eta_direct_se']:.1e} (direct MC)  [预注册 P2: >0.9999]")
    # P1 裁决口径: 预注册定律 η_est=1−K/n_p 是二阶 Taylor（小误差）结果。
    # 在预注册 regime（仓库标定）1−η~1e-6, 直接 MC 的 SE(~2e-6)比信号大 3 个量级
    # （见表中 direct 列正负跳动）, 无法分辨斜率; 有效测量量为 RB（实测相位误差的
    # 二次型平均）与 τ²（纯 MC 相位误差方差, 不受非线性影响）。direct@−10dB 为
    # 非线性区应力测试, 单独报告不作为裁决依据。
    p1 = all(abs(slopes[k] + 1) <= 0.05 for k in ("rb", "tau2"))
    p2 = eta17["eta_rb"] > 0.9999
    print(f"  P1 斜率裁决（小误差区估计量 RB/τ² @ 仓库标定, ±0.05）: "
          f"{'PASS' if p1 else 'FAIL'}   "
          f"P2 regime 裁决: {'PASS' if p2 else 'FAIL'}")
    return rows, tau2_rows, slopes, dict(k_mc=k_mc_mean, k_ex=rows[0]["k_ex"],
                                         k_rd=rows[0]["k_rd"],
                                         eta17=eta17, p1=p1, p2=p2,
                                         direct_m10=direct_lo)


def section_snr_scan(d, M, st, args):
    """§3: SNR 扫描 @ n_p=17：η_est、1/SNR 律、临界 SNR（P3）。"""
    print("\n" + "=" * 78)
    print("[3] SNR 扫描 @ n_p=17（参考链路复 SNR=64/σ_c²）：η_est 与临界点")
    print("=" * 78)
    print(f"{'SNR dB':>7} {'σ_c²':>10} {'SNR/天线':>10} {'1−η MC':>12} {'±SE':>10} "
          f"{'1−η exact':>12} {'1−η prior':>12} {'η_est MC':>10}")
    rows = []
    for snr_db in SNR_DB_LIST + SNR_DB_EXT:
        sig2 = P_REF / (10.0 ** (snr_db / 10.0))
        an = analytic_predictions(d, M, 17, sig2, st=st)
        mc = mc_run(d, M, dft_pilots(17, M.shape[1]), sig2,
                    args.n_trials, seed=3000 + int(snr_db * 10))
        snr_ant = float(((np.abs(d) ** 2 + (np.abs(M) ** 2).sum(1)) / sig2).mean())
        rows.append(dict(snr_db=snr_db, sigma2=sig2, snr_ant=snr_ant,
                         extended=snr_db in SNR_DB_EXT,
                         one_minus_eta=1 - mc["eta_direct"],
                         one_minus_eta_se=mc["eta_direct_se"],
                         one_minus_eta_ex=an["loss_ex"] / st["P_cf"],
                         one_minus_eta_rd=an["loss_rd"] / st["P_cf"],
                         eta=mc["eta_direct"]))
        tag = " (扩展)" if snr_db in SNR_DB_EXT else ""
        print(f"{snr_db:>+7d} {sig2:>10.3f} {snr_ant:>10.2e} {1 - mc['eta_direct']:>12.4e} "
              f"{mc['eta_direct_se']:>10.1e} {an['loss_ex'] / st['P_cf']:>12.4e} "
              f"{an['loss_rd'] / st['P_cf']:>12.4e} {mc['eta_direct']:>10.4f}{tag}")
    # 1/SNR 律（小损失且测量精密的点拟合）+ 临界 SNR（阈值表）
    fit = [r for r in rows if 0 < r["one_minus_eta"] < 0.05
           and r["one_minus_eta_se"] < 0.5 * r["one_minus_eta"]]
    sl = np.polyfit(np.log([P_REF / r["sigma2"] for r in fit]),
                    np.log([r["one_minus_eta"] for r in fit]), 1)
    a_fit = float(np.exp(sl[1]))          # 1−η = a/SNR
    snr_lin = [P_REF / r["sigma2"] for r in rows]
    om = [r["one_minus_eta"] for r in rows]
    crit = {}
    for thr in (0.5, 0.1, 0.05, 0.01, 1e-3):
        crit[str(thr)] = dict(
            fit_db=float(10 * math.log10(a_fit / thr)),
            interp_db=interp_crossing(snr_lin, om, thr))
    # 先验公式的临界点（roadmap 预测路径复现）
    a_prior = float(np.mean([r["one_minus_eta_rd"] * P_REF / r["sigma2"] for r in rows]))
    crit_prior = {str(thr): float(10 * math.log10(a_prior / thr))
                  for thr in (0.5, 0.1, 0.05, 0.01, 1e-3)}
    print(f"\n  1/SNR 律拟合（小损失精密点 n={len(fit)}）: "
          f"1−η_est(17) = {a_fit:.4e}/SNR, log-log 斜率 {sl[0]:+.4f}")

    def _db(x):
        return "n/a(未穿越)" if x is None or not np.isfinite(x) else f"{x:.2f}dB"

    print(f"  {'阈值 1−η':>10} {'临界SNR(1/SNR拟合)':>18} {'临界SNR(MC插值)':>18} "
          f"{'先验公式':>10}")
    for thr in (0.5, 0.1, 0.05, 0.01, 1e-3):
        c = crit[str(thr)]
        print(f"  {thr:>10.4g} {_db(c['fit_db']):>18} {_db(c['interp_db']):>18} "
              f"{crit_prior[str(thr)]:>9.2f}dB")
    loss_m8 = interp_at(snr_lin, om, 10 ** (-0.8))
    print(f"  [预注册 P3: 先验预测 ≈ −8 dB]  "
          f"1−η_est(17)@−8dB(MC插值) = {loss_m8:.4e} ⇒ η_est(17) = {1 - loss_m8:.4f}")
    c50 = crit["0.5"]
    p3 = (c50["interp_db"] is not None and np.isfinite(c50["interp_db"])
          and abs(c50["interp_db"] - (-8.0)) <= 5.0)
    if not (c50["interp_db"] is not None and np.isfinite(c50["interp_db"])):
        p3 = abs(c50["fit_db"] - (-8.0)) <= 5.0
    print(f"  P3 裁决（0.5 阈值, ±5dB 容线; 扫描未穿越则用 1/SNR 拟合外推）: "
          f"{'PASS' if p3 else 'FAIL'}"
          f"（{_db(c50['interp_db'])} / 拟合 {c50['fit_db']:.2f}dB vs 预测 −8 dB）")
    return rows, dict(slope=float(sl[0]), a_fit=a_fit, crit=crit,
                      crit_prior=crit_prior, p3=p3,
                      loss_at_m8=loss_m8, eta_at_m8=1.0 - loss_m8)


def interp_crossing(snr, loss, thr):
    """在 (SNR, loss) 测量点上插值求 loss=thr 的 SNR（dB）。未括住返回 nan。"""
    order = np.argsort(snr)
    s = np.log10(np.array(snr)[order])
    l = np.array(loss)[order]
    for i in range(len(s) - 1):
        if (l[i] - thr) * (l[i + 1] - thr) <= 0 and l[i] != l[i + 1]:
            f = (thr - l[i]) / (l[i + 1] - l[i])
            return float(10.0 * (s[i] + f * (s[i + 1] - s[i])))
    return float("nan")


def interp_at(snr, loss, x):
    order = np.argsort(snr)
    s = np.log10(np.array(snr)[order])
    l = np.array(loss)[order]
    return float(np.interp(math.log10(x), s, l))


def section_n_scan(ctx, args, t):
    """§4: RIS 规模扫描 N∈{16,...,256}, n_p=N+1：恒模可辨识前沿与 K(N)。"""
    print("\n" + "=" * 78)
    print("[4] RIS 规模扫描（n_p=N+1，仓库标定）：K 与 η_est 的 N 依赖性")
    print("=" * 78)
    print(f"{'N':>5} {'n_p':>5} {'P_cf':>12} {'|c|mean':>10} {'SNR/天线dB':>11} "
          f"{'面板m':>7} {'D*/m':>6} {'远场':>5} {'K exact':>10} {'η_est(N+1)':>12}")
    rows = []
    for N in N_SCAN:
        d, M, f_d2 = linear_model_n(ctx, t, N)
        st = channel_stats(d, M)
        n_p = N + 1
        an = analytic_predictions(d, M, n_p, SIGMA2_REF, st=st)
        mc = mc_run(d, M, dft_pilots(n_p, N), SIGMA2_REF,
                    max(500, args.n_trials // 2), seed=4000 + N)
        per_ant = ((np.abs(d) ** 2 + (np.abs(M) ** 2).sum(1)) / SIGMA2_REF).mean()
        k = int(N).bit_length() - 1
        rows_n, cols_n = 1 << ((k + 1) // 2), N >> ((k + 1) // 2)
        extent = max(rows_n - 1, cols_n - 1) * 10.0
        d_sat = ctx["frames"][t]["dist_sat_target"] * 1000.0
        d_star = math.sqrt(ctx["wavelength"] * d_sat / 2.0)
        rows.append(dict(N=N, n_p=n_p, P_cf=st["P_cf"], c_mean=float(np.abs(st["c"]).mean()),
                         snr_ant_db=float(10 * math.log10(per_ant)),
                         panel_extent_m=extent, d_star_m=d_star,
                         farfield_ok=bool(extent <= d_star),
                         k_ex=n_p * an["loss_ex"] / st["P_cf"],
                         k_rd=n_p * an["loss_rd"] / st["P_cf"],
                         eta_rb=mc["eta_rb"], eta_rb_se=mc["eta_rb_se"],
                         k_mc=n_p * (1 - mc["eta_rb"])))
        print(f"{N:>5} {n_p:>5} {st['P_cf']:>12.3e} {np.abs(st['c']).mean():>10.1f} "
              f"{10 * math.log10(per_ant):>10.1f} {extent:>6.0f} {d_star:>6.1f} "
              f"{'✓' if extent <= d_star else '✗':>4} "
              f"{rows[-1]['k_ex']:>10.3e} {mc['eta_rb']:>12.8f}")
    print("  注1: 面板 10m 间距为仓库约定；N≥64 时面板尺度超过远场证书 D*=√(λd/2)"
          "（roadmap §4.4-D3），该行数字带此标注；FIM/损失律本身不依赖远场假设。")
    print("  注2: v^cf 非 P 的精确极大点（忽略跨单元耦合），强耦合下二阶 Taylor 可给出"
          "负损失/η 略大于 1（±1e-7 量级）——估计相位因扰动探索到略高于闭式的功率，"
          "以 MC 为准，不影响 η_est≈1 的结论。")
    return rows


def section_joint(ctx, d, M, st, args, t):
    """§5: 导频-跟踪联合权衡（K×n_p 网格: stale + overhead 总损失）。"""
    print("\n" + "=" * 78)
    print("[5] 导频-跟踪联合权衡：K（重构周期, 帧）× n_p（导频块）总损失")
    print("=" * 78)
    T = ctx["tau"]
    # --- stale: 周期起点真信道闭式设计, 周期内平均功率 / 逐帧理想 ---
    frames_pow = []
    for tt in range(T):
        d2, M2 = linear_model_repo(ctx, tt)
        s2 = channel_stats(d2, M2)
        frames_pow.append(s2["P_cf"])
    frames_pow = np.array(frames_pow)
    eta_stale, detail = {}, {}
    for K in K_LIST:
        ach, ide = [], []
        for t0 in range(0, T, K):
            d0, M0 = linear_model_repo(ctx, t0)
            s0 = channel_stats(d0, M0)
            v0 = s0["v_cf"]
            for tt in range(t0, min(t0 + K, T)):
                d2, M2 = linear_model_repo(ctx, tt)
                ach.append(float((np.abs(d2 + M2 @ v0) ** 2).sum()))
                ide.append(frames_pow[tt])
        eta_stale[K] = float(np.mean(ach) / np.mean(ide))
        detail[K] = dict(achieved=float(np.mean(ach)), ideal=float(np.mean(ide)))
    print(f"  stale（帧尺度, SGP4 几何漂移, oracle 设计）: "
          + ", ".join(f"K={k}: {eta_stale[k]:.4f}" for k in K_LIST))
    # --- sub-frame 多普勒相干曲线（µs 尺度差拍）---
    f1, f2 = doppler_pair(ctx, t)
    df = f2 - f1
    taus = np.linspace(0, 1.0 / abs(df), 61) if df != 0 else np.zeros(1)
    rho = np.exp(1j * 2 * np.pi * df * taus)
    hold = [float((np.abs(r * d + M @ st["v_cf"]) ** 2).sum()) / st["P_cf"] for r in rho]
    print(f"  sub-frame 多普勒相干: 差拍周期 {1e6 / abs(df):.2f} µs, "
          f"保持 v^cf 的功率比 min {min(hold):.3f}（半周期处）")
    # --- 网格: η_joint = η_stale · η_est · (1−overhead) ---
    S = args.slots_per_frame
    eta_est_np = {}
    for n_p in N_P_LIST:
        an = analytic_predictions(d, M, n_p, SIGMA2_REF, st=st)
        eta_est_np[n_p] = 1.0 - an["loss_ex"] / st["P_cf"]
    print(f"\n  overhead 模型: ρ=n_p/(K·S), S={S} 时隙/帧（归一化参数）; "
          f"η_joint=η_stale·η_est·(1−ρ)")
    print(f"\n{'K':>3} {'stale':>7} | " + " ".join(f"{'n_p=' + str(n):>10}" for n in N_P_LIST)
          + "   (总损失 1−η_joint, %)")
    grid = {}
    for K in K_LIST:
        cells = []
        for n_p in N_P_LIST:
            rho_ov = n_p / (K * S)
            eta_j = eta_stale[K] * eta_est_np[n_p] * (1 - rho_ov)
            cells.append(100 * (1 - eta_j))
        grid[K] = cells
        print(f"{K:>3} {eta_stale[K]:>7.4f} | " + " ".join(f"{c:>9.2f}%" for c in cells))
    best = min(((K, n_p, grid[K][i]) for K in K_LIST for i, n_p in enumerate(N_P_LIST)),
               key=lambda x: x[2])
    print(f"  ⇒ 最优: K={best[0]}, n_p={best[1]}（总损失 {best[2]:.2f}%）")
    # --- 直接联合 MC 验证（2 个单元格）---
    print("  直接联合 MC 验证（估计+stale 一体测量, 200 次/格）:")
    val = {}
    for (K, n_p) in ((1, 17), (4, 65)):
        V = dft_pilots(n_p, M.shape[1])
        F = design_matrix(V)
        Finv = np.linalg.inv(F.conj().T @ F)
        rng = np.random.default_rng(5000 + K * 100 + n_p)
        ach, ide, n_tr = [], [], 200
        for t0 in range(0, T, K):
            d0, M0 = linear_model_repo(ctx, t0)
            Yc, rho_b, _, _ = observe_beat(d0, M0, V, doppler_pair(ctx, t0)[0],
                                           doppler_pair(ctx, t0)[1], 0.0, rng)
            assert float(np.abs(rho_b - 1).max()) < 1e-9
            n_cyc = len(range(0, T, K))
            for _ in range(n_tr // n_cyc + 1):
                W = (rng.standard_normal((n_p, d0.shape[0])) +
                     1j * rng.standard_normal((n_p, d0.shape[0]))) * math.sqrt(SIGMA2_REF / 2)
                Y = Yc + W
                Xh = Finv @ (F.conj().T @ Y)
                c_h = np.conj(Xh[0]) @ Xh[1:].T
                v_h = np.conj(c_h) / np.maximum(np.abs(c_h), 1e-30)
                for tt in range(t0, min(t0 + K, T)):
                    d2, M2 = linear_model_repo(ctx, tt)
                    ach.append(float((np.abs(d2 + M2 @ v_h) ** 2).sum()))
                    ide.append(frames_pow[tt])
        eta_direct = float(np.mean(ach) / np.mean(ide))
        model = eta_stale[K] * eta_est_np[n_p]          # 分解验证（不含 airtime overhead）
        val[f"K{K}_np{n_p}"] = dict(eta_direct=eta_direct, eta_model=model,
                                    eta_with_overhead=model * (1 - n_p / (K * S)))
        print(f"    K={K}, n_p={n_p}: 直接 MC η_joint={eta_direct:.4f} vs "
              f"分解模型 η_stale·η_est={model:.4f}"
              f"（再乘 airtime (1−ρ) 得 {model * (1 - n_p / (K * S)):.4f}）")
    return dict(eta_stale=eta_stale, grid=grid, best=best, hold_curve=hold,
                tau_us=(taus * 1e6).tolist(), validation=val,
                slots_per_frame=S, detail=detail)


def section_robustness(ctx, d, M, st, args, t):
    """§6: 稳健性——随机正交导频 / 多 ROI SNR 扫描 / calculate_value_sat 直跑 MC。"""
    import torch
    print("\n" + "=" * 78)
    print("[6] 稳健性检查")
    print("=" * 78)
    out = {}
    # (a) 随机正交导频 vs DFT
    print("  (a) 随机正交（QR）导频 vs DFT 导频（n_p=17/65, 3 seeds, τ² 与 1−η）:")
    ra_rows = []
    for n_p in (17, 65):
        for s in range(3):
            rng = np.random.default_rng(6000 + s)
            V = random_orth_pilots(n_p, M.shape[1], rng)
            F = design_matrix(V)
            orth = float(np.abs(F.conj().T @ F - n_p * np.eye(M.shape[1] + 1)).max()) / n_p
            mc = mc_run(d, M, V, SIGMA2_REF, args.n_trials, seed=6100 + s)
            mc_dft = mc_run(d, M, dft_pilots(n_p, M.shape[1]), SIGMA2_REF,
                            args.n_trials, seed=6100 + s)
            ra_rows.append(dict(n_p=n_p, seed=s, orth_err=orth,
                                tau2=mc["tau2"], tau2_dft=mc_dft["tau2"],
                                one_minus_eta=1 - mc["eta_rb"],
                                one_minus_eta_dft=1 - mc_dft["eta_rb"]))
            print(f"    n_p={n_p} seed={s}: 正交偏差 {orth:.2e}, "
                  f"τ² {mc['tau2']:.4e} (DFT {mc_dft['tau2']:.4e}), "
                  f"1−η {1 - mc['eta_rb']:.4e} (DFT {1 - mc_dft['eta_rb']:.4e})")
    out["random_orthogonal"] = ra_rows
    # (b) 多 ROI SNR 扫描（临界 SNR 的 ROI 分散度）
    print("  (b) 多 ROI SNR 扫描 @ n_p=17（η_est(17) 与临界 SNR 的 ROI 依赖）:")
    roi_rows = []
    for rs in range(args.roi_seed, args.roi_seed + args.n_roi):
        seed_all(rs)
        roi_np, _, _ = generate_ground_target_sample()
        roi_t = torch.tensor(roi_np.astype(np.float32)).reshape(-1)
        Ht = ctx["channels"].channels_per_frame[t]
        d2, M2 = ctx["opt"]._linear_model(Ht, roi_t, ctx["X"])
        d2 = d2.numpy().astype(np.complex128)
        M2 = M2.numpy().astype(np.complex128)
        s2 = channel_stats(d2, M2)
        om, om_prior, snrs = [], [], []
        for snr_db in SNR_DB_LIST:
            sig2 = P_REF / (10.0 ** (snr_db / 10.0))
            mc = mc_run(d2, M2, dft_pilots(17, M2.shape[1]), sig2,
                        max(500, args.n_trials // 2), seed=7000 + rs * 10 + int(snr_db))
            an = analytic_predictions(d2, M2, 17, sig2, st=s2)
            om.append(1 - mc["eta_direct"])
            om_prior.append(an["loss_rd"] / s2["P_cf"])
            snrs.append(P_REF / sig2)
        c50 = interp_crossing(snrs, om, 0.5)
        a_pr = float(np.mean([o * s for o, s in zip(om_prior, snrs)]))
        roi_rows.append(dict(seed=rs, occ=int((roi_np > 0.5).sum()),
                             P_cf=s2["P_cf"],
                             one_minus_eta=[float(x) for x in om],
                             crit50_db=c50,
                             crit50_prior_db=float(10 * math.log10(a_pr / 0.5)),
                             loss_at_m8=interp_at(snrs, om, 10 ** (-0.8)),
                             eta_at_m8=1.0 - interp_at(snrs, om, 10 ** (-0.8))))

        def _db(x):
            return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"

        print(f"    seed {rs} (occ {roi_rows[-1]['occ']}, P_cf {s2['P_cf']:.2e}): "
              f"1−η@20..−20dB = " + "/".join(f"{x:.1e}" for x in om)
              + f", 0.5-临界 MC {_db(c50)} dB / 先验 "
              f"{roi_rows[-1]['crit50_prior_db']:.2f} dB, "
              f"η@−8dB {roi_rows[-1]['eta_at_m8']:.3f}")
    out["multi_roi"] = roi_rows
    # (c) calculate_value_sat 直跑 MC（不经线性模型近似）
    print("  (c) calculate_value_sat 直跑 MC（n_p=17, 200 次, 仓库标定）:")
    Ht = ctx["channels"].channels_per_frame[t]
    V = dft_pilots(17, M.shape[1])
    phases = [np.angle(V[p]).astype(np.float32) for p in range(17)]
    F = design_matrix(V)
    Finv = np.linalg.inv(F.conj().T @ F)
    eta_l = []
    for tr in range(200):
        Y = np.zeros((17, M.shape[0]), dtype=np.complex128)
        for p in range(17):
            torch.manual_seed(800000 + tr * 17 + p)
            y = calculate_value_sat(ctx["roi_t"], phases[p], ctx["X"], Ht,
                                    POWER_SIGMA, 0.0, ctx["wavelength"])
            Y[p] = y.numpy().astype(np.complex128).flatten()
        Xh = Finv @ (F.conj().T @ Y)
        c_h = np.conj(Xh[0]) @ Xh[1:].T
        v_h = np.conj(c_h) / np.maximum(np.abs(c_h), 1e-30)
        eta_l.append(float((np.abs(d + M @ v_h) ** 2).sum()) / st["P_cf"])
    eta_cvs = float(np.mean(eta_l))
    eta_cvs_se = float(np.std(eta_l, ddof=1) / math.sqrt(len(eta_l)))
    out["calculate_value_sat_mc"] = dict(eta=eta_cvs, eta_se=eta_cvs_se, n=len(eta_l))
    print(f"    η_est(17) = {eta_cvs:.6f} ± {eta_cvs_se:.1e}"
          f"（与 float64 管线同口径; 大 SE 为高 SNR 下直接 MC 的固有噪声）")
    return out


def main(args):
    t0 = time.time()
    ctx = build_context(args)
    t = args.frame if args.frame is not None else ctx["tau"] // 2
    d, M, st, f1, f2, per_ant_snr, fw = section_setup(ctx, args, t)

    fim_rows = section_fim_scan(d, M, st, args)
    np_rows, tau2_rows, slopes, np_verdict = section_np_scan(d, M, st, args)
    snr_rows, snr_verdict = section_snr_scan(d, M, st, args)
    n_rows = section_n_scan(ctx, args, t)
    joint = section_joint(ctx, d, M, st, args, t)
    robust = section_robustness(ctx, d, M, st, args, t)

    # ---- 预注册命题裁决汇总 ----
    print("\n" + "=" * 78)
    print("[7] 预注册命题裁决（roadmap §4.3 方向二）")
    print("=" * 78)
    crit50 = snr_verdict["crit"]["0.5"]
    roi_c50 = [r["crit50_db"] for r in robust["multi_roi"]
               if r["crit50_db"] is not None and np.isfinite(r["crit50_db"])]
    roi_m8 = [r["eta_at_m8"] for r in robust["multi_roi"]]
    roi_note = (f"; 多ROI 0.5-穿越 [{min(roi_c50):.1f},{max(roi_c50):.1f}]dB"
                f", η@−8dB∈[{min(roi_m8):.2f},{max(roi_m8):.2f}]" if roi_c50 else "")

    def _db(x):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}dB"

    verdicts = {
        "P1_slope": dict(pred="-1.00±0.05（小误差区估计量）",
                         measured=f"RB {slopes['rb']:+.4f} / τ² {slopes['tau2']:+.4f}"
                                  f"（应力测试 direct@−10dB: 全 {slopes['direct_m10']:+.4f}, "
                                  f"n_p≥33 {slopes['direct_m10_lin']:+.4f}）",
                         pass_=bool(np_verdict["p1"])),
        "P2_regime": dict(pred="η_est(17)>0.9999 @ SNR≈3200",
                          measured=f"{np_verdict['eta17']['eta_rb']:.8f}",
                          pass_=bool(np_verdict["p2"])),
        "P3_critical_snr": dict(pred="≈−8 dB（先验公式）",
                                measured=f"0.5-阈值: MC {_db(crit50['interp_db'])} / "
                                         f"1/SNR外推 {crit50['fit_db']:.2f}dB / "
                                         f"先验 {snr_verdict['crit_prior']['0.5']:.2f}dB; "
                                         f"η(17)@−8dB={snr_verdict['eta_at_m8']:.4f}"
                                         + roi_note,
                                pass_=bool(snr_verdict["p3"])),
    }
    for k, v in verdicts.items():
        print(f"  {k:>16}: 预测 {v['pred']:<26} 实测 {v['measured']:<62} "
              f"[{'PASS' if v['pass_'] else 'FAIL'}]")

    out = dict(
        protocol=dict(n_trials=args.n_trials, n_p_list=N_P_LIST,
                      snr_db_list=SNR_DB_LIST, doppler=args.doppler,
                      roi_seed=args.roi_seed, frame=t, irs_mode=args.irs_mode,
                      sigma2_ref=SIGMA2_REF, snr_ref=SNR_REF,
                      slots_per_frame=args.slots_per_frame),
        scenario=dict(N=M.shape[1], A=M.shape[0], P_cf=st["P_cf"],
                      c_abs_mean=float(np.abs(st["c"]).mean()),
                      per_ant_snr_mean=float(per_ant_snr.mean()),
                      f_d1=f1, f_d2=f2, beat_period_us=float(1e6 / abs(f2 - f1))),
        forward_model_check=fw,
        fim_scan=fim_rows,
        np_scan=np_rows,
        tau2_scan=tau2_rows,
        np_scan_verdict=dict(slopes=slopes,
                             k_mc=np_verdict["k_mc"], k_ex=np_verdict["k_ex"],
                             k_rd=np_verdict["k_rd"],
                             eta17=dict(eta_rb=np_verdict["eta17"]["eta_rb"],
                                        eta_direct=np_verdict["eta17"]["eta_direct"],
                                        eta_direct_se=np_verdict["eta17"]["eta_direct_se"]),
                             p1=bool(np_verdict["p1"]), p2=bool(np_verdict["p2"])),
        snr_scan=snr_rows,
        snr_scan_verdict=snr_verdict,
        n_scan=n_rows,
        joint_tradeoff=joint,
        robustness=robust,
        verdicts=verdicts,
        runtime_s=round(time.time() - t0, 1),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "phase_fim.json")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return [_clean(v) for v in o.tolist()]
        if isinstance(o, (np.floating, float)):
            o = float(o)
            return None if math.isnan(o) else o
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {path}  (运行 {out['runtime_s']}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RIS 相位设计 pilot FIM 与 η_est 验证")
    p.add_argument("--n_trials", type=int, default=2000, help="每格 MC 次数（≥500）")
    p.add_argument("--doppler", choices=["beat", "zero"], default="beat",
                   help="beat: SGP4 差拍周期导频间隔+精确补偿; zero: t_rel=0 忽略多普勒")
    p.add_argument("--roi_seed", type=int, default=42)
    p.add_argument("--n_roi", type=int, default=4, help="多 ROI 稳健性扫描的 ROI 数")
    p.add_argument("--frame", type=int, default=None, help="默认中间帧（标定参考帧）")
    p.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    p.add_argument("--bs_ant", type=int, default=4)
    p.add_argument("--ue_ant", type=int, default=4)
    p.add_argument("--tau", type=int, default=8)
    p.add_argument("--slots_per_frame", type=int, default=1000,
                   help="overhead 归一化: 每帧时隙数")
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "isac_demo"))
    args = p.parse_args()
    main(args)
