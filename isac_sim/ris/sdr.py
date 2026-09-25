"""
isac_sim/ris/sdr.py — 恒模 QCQP 的 SDR 松弛求解与最优性证书

问题：P* = max_{|v_i|=1} ‖d + Mv‖² = max_{|w_i|=1} w^H Q̄ w，
其中 w = [v; 1] ∈ T^{N+1}，Q̄ = [[M^H M, M^H d], [d^H M, ‖d‖²]] ⪰ 0（Gram）。

SDR 松弛：P_SDP = max_{W⪰0, W_ii=1} <Q̄, W> ≥ P*（松弛是合法上界）。

本模块提供（全部 numpy/torch 实现，无 cvxpy 依赖）：
  - build_augmented(d, M)      : 增广 Gram 矩阵 Q̄
  - sdr_burer_monteiro(...)    : BM 因子化 W=YY^H，行单位模（复 oblique 流形）黎曼梯度上升
  - gershgorin_upper(Q̄)       : 对偶可行上界 U = Σ_i (Q̄_ii + Σ_{j≠i}|Q̄_ij|)（**严格合法**，
                                 对角占优 ⇒ Diag(y) ⪰ Q̄ ⇒ 强对偶给 P_SDP ≤ U）
  - eigvec_rounding(W)         : W 主特征向量以常数槽为参考消全局相位 → 可行点（下界）
  - gaussian_rounding(...)     : ξ~CN(0,W)，v_i = ξ_i/ξ_N → 可行点（启发式；常数槽被钉住，
                                 So-Zhang-Ye 的 π/4 保证不覆盖本问题，只作对照）

证书结构（每帧可计算）：
  L := max(各可行点功率) ≤ P* ≤ U := Gershgorin 上界，
  且若 BM 解的 W 数值秩为 1（λ₂ ≤ ε），则其特征向量即**全局最优**（精确性证书）。
"""

import math
import numpy as np
import torch


def build_augmented(d, M):
    """d: [B] 复, M: [B, N] 复 → Q̄ = [[M^H M, M^H d], [d^H M, ‖d‖²]]（torch complex [N+1, N+1]）。"""
    d = torch.as_tensor(d).to(torch.complex64)
    M = torch.as_tensor(M).to(torch.complex64)
    n = M.shape[1]
    Q = torch.zeros((n + 1, n + 1), dtype=torch.complex64)
    Q[:n, :n] = M.conj().T @ M
    Q[:n, n] = M.conj().T @ d
    Q[n, :n] = d.conj().T @ M
    Q[n, n] = (d.conj() * d).sum().real
    return Q


def gershgorin_upper(Q):
    """严格合法的对偶上界：y_i = Q_ii + Σ_{j≠i}|Q_ij| ⇒ Diag(y) ⪰ Q（Gershgorin 圆盘）。"""
    Q = np.asarray(Q)
    n = Q.shape[0]
    y = np.real(np.diag(Q)) + np.abs(Q - np.diag(np.diag(Q))).sum(axis=1)
    return float(y.sum())


def _power_from_v(d, M, v):
    """‖d + Mv‖²，v: [N] 复单位模。"""
    d = torch.as_tensor(d).to(torch.complex64)
    M = torch.as_tensor(M).to(torch.complex64)
    y = d + M @ v.to(torch.complex64)
    return float((y.conj() * y).sum().real.item())


def sdr_burer_monteiro(Q, rank=4, iters=800, lr=0.05, n_restart=4, seed=0,
                       device="cpu"):
    """BM：W = YY^H，Y ∈ C^{(N+1)×r} 行单位模；复 oblique 流形黎曼梯度上升。

    目标 f(Y) = tr(Q YY^H) = ‖B Y^H‖_F²（Q = B^H B 时谱等价），对 Y 的
    Wirtinger 梯度 ∇ = 2QY；切空间投影后行归一化收缩。

    返回 (Y*, f*, lambda2)：
      f*     = tr(Q Y*Y*^H)（松弛问题的可行值 ⇒ SDP 最优的下估计；秩为 1 时 = P*）
      lambda2 = W* 第二大特征值（秩-1 证书：λ₂≈0 ⇒ 全局最优）
    """
    Q = torch.as_tensor(Q).to(torch.complex64).to(device)
    n1 = Q.shape[0]
    gen = torch.Generator().manual_seed(seed)
    best = (None, -1.0, None)
    for rs in range(n_restart):
        Y = torch.complex(
            torch.randn(n1, rank, generator=gen),
            torch.randn(n1, rank, generator=gen),
        ).to(device)
        Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)   # 行单位模（可行）
        for it in range(iters):
            G = 2.0 * (Q @ Y)                                   # 欧氏梯度
            # 切空间投影：g_i ← g_i - Re(y_i^H g_i) y_i
            ip = (Y.conj() * G).sum(dim=1, keepdim=True).real
            G = G - ip * Y
            Y = Y + lr * G
            Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)  # 收缩=行归一化
        W = Y @ Y.conj().T
        f = float(torch.trace(Q @ W).real.item())
        if f > best[1]:
            ev = torch.linalg.eigvalsh(W)
            lam2 = float(ev[-2].item()) if ev.numel() > 1 else 0.0
            best = (Y.detach(), f, lam2)
    return best


def eigvec_rounding(W):
    """W 主特征向量 → 可行相位（下界）。

    增广向量 w = [v; 1]：常数项在最后一个槽。整体相位自由，
    故以最后槽为参考消全局相位：v_i = w_i · conj(w_N)。
    """
    W = torch.as_tensor(W).to(torch.complex64)
    ev, V = torch.linalg.eigh(W)
    w = V[:, -1]
    v = w[:-1] * torch.conj(w[-1])
    return v / (v.abs() + 1e-12)


def gaussian_rounding(d, M, W, n_samples=64, seed=0):
    """ξ ~ CN(0, W)，v_i = e^{j∠(ξ_i/ξ_{N+1})}（对增广变量的第一分量归一化）。

    期望保证（So-Zhang-Ye，PSD 目标复 QP）：E[p(v)] ≥ (π/4)·P_SDP。
    返回 (最佳 v, 平均功率, 最佳功率)。
    """
    d = np.asarray(d); M = np.asarray(M); W = np.asarray(W)
    n = M.shape[1]
    rng = np.random.default_rng(seed)
    best_v, best_p, ps = None, -1.0, []
    for _ in range(n_samples):
        xi = rng.standard_normal(n + 1) + 1j * rng.standard_normal(n + 1)
        # 按 W 相关：xi = L z，L = chol(W)（W PSD）
        try:
            L = np.linalg.cholesky(W + 1e-9 * np.eye(n + 1))
            xi = L @ xi
        except np.linalg.LinAlgError:
            pass
        v = xi[:n] / xi[n:n + 1]           # 常数项在最后槽：以 ξ_N 为参考消全局相位
        v = v / (np.abs(v) + 1e-12)
        p = float(np.sum(np.abs(d + M @ v) ** 2))
        ps.append(p)
        if p > best_p:
            best_p, best_v = p, v
    return best_v, float(np.mean(ps)), best_p


def sdr_lagrangian_dual(Q, iters=4000, lr=0.5, device="cpu"):
    """拉格朗日对偶上界（严格合法）。

    P* = max_{|w_i|=1} w^H Q w。把可行集松弛到球面 {‖w‖² = N+1}（超集），
    对偶函数（λ ≥ 0）：
        g(λ) = (N+1)·λ_max(Q - Diag(λ)) + Σ_i λ_i  ≥  P*
    为凸函数，用投影次梯度下降（步长 lr/√k，迭代平均）最小化。
    任意 λ ≥ 0 给出合法上界。

    返回 (dual_ub, lambda)。
    """
    Q = torch.as_tensor(Q).to(torch.complex64).to(device)
    n1 = Q.shape[0]
    lam = torch.zeros(n1, device=device)
    lam_avg = torch.zeros(n1, device=device)
    best = float("inf")
    for k in range(iters):
        A = Q - torch.diag(lam)
        ev, V = torch.linalg.eigh(A)
        v = V[:, -1]
        g = n1 * float(ev[-1].real.item()) + float(lam.sum().item())
        best = min(best, g)
        sub = 1.0 - n1 * v.abs() ** 2                  # ∂g/∂λ_i
        lam = torch.clamp(lam - (lr / math.sqrt(k + 1)) * sub, min=0.0)
        lam_avg += lam
    lam_avg /= iters
    A = Q - torch.diag(lam_avg)
    g_avg = n1 * float(torch.linalg.eigvalsh(A)[-1].real.item()) + float(lam_avg.sum().item())
    return min(best, g_avg), lam_avg


def sdr_phase_bounds(d, M, rank=6, iters=1500, n_round=64, seed=0, device="cpu"):
    """一站式：返回 (certificate dict, phases dict)。

    certificate:
      p_cf_eig      : BM 特征向量舍入功率（可行点=下界）
      p_round_mean  : 高斯舍入平均功率（启发式）
      p_round_best  : 高斯舍入最佳（可行点=下界）
      p_bm          : BM 目标值（松弛可行值）
      lambda2       : BM W 的第二特征值（秩-1 证书：λ₂≈0 ⇒ 全局最优）
      p_gershgorin  : Gershgorin 对偶上界（严格合法）
      p_fw_oracle   : FW 谕言上界（严格合法，通常比 Gershgorin 紧）
      lower_bound   : max(各可行点功率)
      gap_upper     : min(合法上界) / lower_bound
    """
    d_t = torch.as_tensor(d).to(torch.complex64)
    M_t = torch.as_tensor(M).to(torch.complex64)
    Q = build_augmented(d_t, M_t)

    Y, f_bm, lam2 = sdr_burer_monteiro(Q, rank=rank, iters=iters, seed=seed, device=device)
    W = Y @ Y.conj().T

    v_eig = eigvec_rounding(W)
    p_eig = float(torch.sum(torch.abs(d_t + M_t @ v_eig) ** 2).real.item())

    v_gr, p_gr_mean, p_gr_best = gaussian_rounding(
        d_t.cpu().numpy(), M_t.cpu().numpy(), W.cpu().numpy(),
        n_samples=n_round, seed=seed + 1)

    p_dual, lam = sdr_lagrangian_dual(Q, device=device)

    cert = {
        "p_cf_eig": p_eig,
        "p_round_mean": p_gr_mean,
        "p_round_best": p_gr_best,
        "p_bm": f_bm,
        "lambda2": lam2,
        "p_gershgorin": gershgorin_upper(Q.cpu().numpy()),
        "p_dual": p_dual,
    }
    lb = max(p_eig, p_gr_best)
    cert["lower_bound"] = lb
    cert["gap_upper"] = min(cert["p_gershgorin"], p_dual) / max(lb, 1e-12)
    return cert, {"v_eig": v_eig, "v_gr": v_gr}
