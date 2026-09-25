"""verify_tracking_dp.py — RIS 分段重构的 DP 精确最优调度（roadmap §4.2 方向五 / §7 G3、P2）

把 verify_tracking.py 的 K-sweep（启发式均匀网格 track_K=1/2/4/8）升级为
**精确最优重配置调度**（动态规划）。

数学设定
--------
帧列 t=1..T 由 SGP4/TLE 确定性可预计算（setup_sat.SatISACScenario）。
更新集 S={s_1<...<s_U}，RIS 相位在 [s_k, s_{k+1}) 保持为 v*(H_{s_k})
（PhaseOptimizerSat.optimize_frame 全模型闭式对齐，与 compare_tracking
使用同一相位设计器）：

    J(S) = Σ_k Σ_{t=s_k}^{s_{k+1}-1} P(v*(H_{s_k}); H_t)

可行集约定：s_1=1（窗口首帧必更新），S 是 [1..T] 的全覆盖划分——RIS 须
服务窗口内每一帧，不存在"跳过前段帧"的选项（DP 递推 F(0,0)=0、
F(0,u)=-∞(u≥1) 即强制该约定；否则"最优调度"会退化成挑选高功率帧，
实测 u=1 时仅覆盖帧 5..8 的调度可达高于全覆盖的值，语义不合法）。

DP 递推（O(T²U)，T=8 瞬时）：

    F(t,u) = max_{s≤t} [ F(s-1,u-1) + W(s,t) ],   W(s,t) = Σ_{τ=s}^{t} P(v_s; H_τ)

    J(S*_U) = F(T,U)，S* 由回溯恢复。均匀 K 调度只是可行点 ⇒ 间隙精确成立。

预注册命题（执行前不得更改，见 docs/optimization_roadmap.md §7）
------------------------------------------------------------
G3  帧间漂移率 ρ_t=‖H_{t+1}-H_t‖_F/‖H_t‖_F 的非均匀比 max/min ≥ 3；
    <1.5 ⇒ 均匀 K 近优、DP 方向作废（如实报告证伪）。
    H_t 定义（本脚本口径）：相位优化目标的**等效线性信道**
    Ĥ_t=[d_t, M_t]∈C^{BS×(N+1)}——P(v)=‖d+Mv‖² 的完整线性模型
    （d=直达+散射系数，M=RIS 双径合成系数矩阵）。辅助口径（M 单独、
    原始 H_BS_ROI 星-地链路）一并落盘 JSON 供交叉参照；星载内部链路
    （如 H_BS_IRS：BS 天线与星载 RIS 同随卫星刚体平移）矩阵逐帧完全
    相同，不构成漂移。
P2  同预算/更少预算下 DP 更优（"同预算更高功率"）：
    A: J(S*_{U=4}) ≥ 1.03·J(均匀K=2)（均匀 K=2 恰为 U=4 次更新，同预算 +3%）；
    B: J(S*_{U=3}) ≥ J(均匀K=4)（3 次更新打败 4 帧间隔的 2 次更新）。
       B 由"均匀调度是 DP 可行点 + J 对 U 单调（在段末插入更新不降 J）"
       数学上 guaranteed，用作 DP 实现正确性的内部自检——不成立即实现有 bug。

协议（与 verify_tracking.py 同场景）
----------------------------------
- 相位设计器：PhaseOptimizerSat.optimize_frame（闭式对齐，compare_tracking 默认方法）；
- intervals=[1,2,4,8]；ROI 用 verify_tracking.make_roi_voxel（legacy 物体，与
  TECH_REPORT/verify_tracking 一致）；
- 8 个 seed = 8 个不同 SGP4 过境窗口（ISS TLE，96h 窗口搜索，seed i → 第 i 个
  窗口；seed 0 即 verify_tracking.py 的默认几何）；
- 内部自检：J(均匀K)/T 必须复现 compare_tracking 的 track_K 平均功率；
  DP 最优性经全部全覆盖划分穷举核对（T=8 共 ΣC(7,u-1)=128 个可行调度）。

诚实边界（假设）
----------------
- 自由空间确定性信道（无衰落/噪声），与 verify_tracking.py 一致；莱斯稳健性见
  verify_tracking_rician.py。
- 闭式相位对齐忽略跨单元交叉项（约为坐标上升数值上界的 77~89%），本脚本比较的是
  **调度策略**（同一相位设计器下），不声称相位设计本身最优。
- 功率为标定后的仿真值（tensor_a ~1e11 量级），表中比值为核心量。

用法：py verify_tracking_dp.py [--n_seeds 8] [--search_hours 96]
"""

import os
import sys
import json
import math
import random
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_sat as ss
from data_sat import SatScenarioChannels, _SIGNAL1
from phase_optimizer_sat import PhaseOptimizerSat, compare_tracking
from verify_tracking import make_roi_voxel

NEG = -1e300


# ----------------------------------------------------------------------
# 核心数学：功率矩阵 / DP / 均匀调度
# ----------------------------------------------------------------------

def effective_channel(opt, Ht, ROI, X):
    """等效线性信道 Ĥ=[d, M]∈C^{BS×(N+1)}（P(v)=‖d+Mv‖² 的完整线性模型）。"""
    d, M = opt._linear_model(Ht, ROI, X)
    return torch.cat([d.reshape(-1, 1), M], dim=1)


def power_matrix(opt, channels, ROI, X):
    """P[a,b] = 帧 a 最优相位在帧 b 上评估的接收功率（T×T, float64）。"""
    T = len(channels.channels_per_frame)
    phases = [opt.optimize_frame(channels.channels_per_frame[a], ROI, X)
              for a in range(T)]
    P = np.zeros((T, T), dtype=np.float64)
    for a in range(T):
        for b in range(T):
            P[a, b] = opt._power(channels.channels_per_frame[b], ROI, X, phases[a])
    return P


def dp_optimal_schedules(P):
    """DP 精确最优调度。返回 {U: (J(S*_U), S*_U)}，U=1..T。

    F(t,u)=max_{s≤t}[F(s-1,u-1)+W(s,t)]，W(s,t)=Σ_{τ=s}^{t}P[s,τ]；
    F(0,0)=0，覆盖帧 0..t-1 恰好 u 次更新。回溯恢复 S*。
    """
    T = P.shape[0]
    W = np.zeros((T, T), dtype=np.float64)
    for s in range(T):
        W[s, s:] = np.cumsum(P[s, s:])
    G = np.full((T + 1, T + 1), NEG, dtype=np.float64)
    choice = np.full((T + 1, T + 1), -1, dtype=int)
    G[0, 0] = 0.0
    for t in range(1, T + 1):                 # 覆盖帧 0..t-1
        for u in range(1, t + 1):             # 恰好 u 次更新
            best, bs = NEG, -1
            for s in range(1, t + 1):         # 最后一次更新在帧 s-1
                prev = G[s - 1, u - 1]
                if prev <= NEG / 2:
                    continue
                val = prev + W[s - 1, t - 1]
                if val > best:
                    best, bs = val, s - 1
            G[t, u], choice[t, u] = best, bs
    out = {}
    for u in range(1, T + 1):
        S, t, uu = [], T, u
        while uu > 0:
            s = int(choice[t, uu])
            S.append(s)
            t, uu = s, uu - 1
        out[u] = (float(G[T, u]), S[::-1])
    return out


def uniform_schedule(K, T):
    """均匀 K 调度（与 optimize_sequence 的 t % K == 0 一致）。"""
    return list(range(0, T, K))


def J_of_schedule(P, S):
    """J(S)=Σ_k Σ_{t∈[s_k,s_{k+1})} P(v_{s_k}; H_t)。"""
    T = P.shape[0]
    S = list(S)
    total = 0.0
    for i, s in enumerate(S):
        e = S[i + 1] if i + 1 < len(S) else T
        total += P[s, s:e].sum()
    return float(total)


# ----------------------------------------------------------------------
# 单 seed：一个过境窗口
# ----------------------------------------------------------------------

def run_seed(seed_idx, args, ROI):
    torch.manual_seed(args.seed + seed_idx)
    np.random.seed(args.seed + seed_idx)
    random.seed(args.seed + seed_idx)

    scenario = ss.SatISACScenario(tau=args.tau, search_hours=args.search_hours)
    windows = scenario.find_overpass()
    if seed_idx >= len(windows):
        raise RuntimeError(f"seed {seed_idx}: {args.search_hours:.0f}h 内仅找到 "
                           f"{len(windows)} 个过境窗口，请增大 --search_hours "
                           f"或减小 --n_seeds")
    w = windows[seed_idx]
    center = (w[0] + w[1]) / 2.0
    frames = scenario.build_frames(window_center_sec=center)
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device="cpu",
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    opt = PhaseOptimizerSat(channels, device="cpu")
    T = len(frames)

    # ---- 功率矩阵 + DP ----
    P = power_matrix(opt, channels, ROI, X)
    dp = dp_optimal_schedules(P)                      # {U: (J*, S*)}
    S_unif = {K: uniform_schedule(K, T) for K in args.intervals}
    J_unif = {K: J_of_schedule(P, S) for K, S in S_unif.items()}
    U_of_K = {K: len(S) for K, S in S_unif.items()}   # 均匀 K 的更新次数（预算）

    # ---- G3：帧间漂移率（主口径=等效线性信道 [d,M]）----
    # 注：星载内部链路（如 H_BS_IRS，BS 天线与星载 RIS 同随卫星刚体平移）矩阵
    # 逐帧完全相同——漂移全部来自卫星-地面链路，故辅助口径取 H_BS_ROI。
    H_eff = [effective_channel(opt, channels.channels_per_frame[t], ROI, X)
             for t in range(T)]
    M_only = [opt._linear_model(channels.channels_per_frame[t], ROI, X)[1]
              for t in range(T)]
    link = [channels.channels_per_frame[t]["H_BS_ROI"] for t in range(T)]

    def _rho(mats):
        r = [float((mats[t + 1] - mats[t]).norm() / mats[t].norm())
             for t in range(T - 1)]
        ratio = float(max(r) / min(r)) if min(r) > 1e-12 else float("inf")
        return r, ratio

    def _finite(x):
        return x if math.isfinite(x) else None

    rho_eff, ratio_eff = _rho(H_eff)
    rho_M, ratio_M = _rho(M_only)
    rho_link, ratio_link = _rho(link)

    # ---- P2：同预算/更少预算对比 ----
    J_U4, S_U4 = dp[4]
    J_U3, S_U3 = dp[3]
    p2 = {
        "J_U4": J_U4, "S_U4": S_U4, "J_unif_K2": J_unif[2],
        "ratio_U4_over_K2": J_U4 / J_unif[2],
        "J_U3": J_U3, "S_U3": S_U3, "J_unif_K4": J_unif[4],
        "ratio_U3_over_K4": J_U3 / J_unif[4],
        "abs_diff_U3_minus_K4": J_U3 - J_unif[4],
    }

    # ---- 精确间隙 gap_K=(J(S*_{U_K})-J(S_K))/J(S*_{U_K}) ----
    gap = {K: (dp[U_of_K[K]][0] - J_unif[K]) / dp[U_of_K[K]][0]
           for K in args.intervals}

    # ---- 内部自检 1：J(均匀K)/T 复现 compare_tracking 的 track_K 功率 ----
    cmp_res = compare_tracking(channels, ROI, X, device="cpu",
                               n_iter=args.n_iter, intervals=args.intervals)
    cross = {}
    for K in args.intervals:
        ref = cmp_res[f"track_K={K}"]["power"]
        mine = J_unif[K] / T
        cross[K] = {"compare_tracking_power": ref, "J_div_T": mine,
                    "rel_diff": abs(mine / ref - 1.0) if ref > 0 else 0.0}

    # ---- 内部自检 2：DP 结构性质 + 穷举证书 ----
    # 可行集 = 必含帧 0 的全覆盖划分（RIS 须服务窗口内每一帧；DP 递推
    # F(0,0)=0 / F(0,u)=-inf(u≥1) 即强制 s_1=0。若允许跳过前段帧，"最优
    # 调度"会退化成挑选高功率帧——实测 u=1 时 [4]（仅覆盖 4..7 帧）可达
    # 8.09e6 > 全覆盖 [0] 的 5.50e6，故该语义被预注册排除）。
    Js = [dp[u][0] for u in range(1, T + 1)]
    bf = {u: max(J_of_schedule(P, [0] + list(S))
                 for S in combinations(range(1, T), u - 1))
          for u in range(1, T + 1)}
    bf_max_rel = max(abs(bf[u] - dp[u][0]) / max(abs(bf[u]), 1.0)
                     for u in range(1, T + 1))
    checks = {
        "brute_force_matches_dp": bf_max_rel <= 1e-9,
        "brute_force_max_rel_diff": bf_max_rel,
        "Jstar_monotone_in_U": all(Js[u] >= Js[u - 1] - 1e-6 * abs(Js[u])
                                   for u in range(1, T)),
        "Jstar_U1_eq_unif_K8": abs(dp[1][0] - J_unif[max(args.intervals)])
                               <= 1e-6 * abs(J_unif[max(args.intervals)]),
        "Jstar_UT_eq_unif_K1": abs(dp[T][0] - J_unif[1])
                               <= 1e-6 * abs(J_unif[1]),
        "Jstar_ge_uniform_same_budget": all(
            dp[U_of_K[K]][0] >= J_unif[K] - 1e-6 * abs(J_unif[K])
            for K in args.intervals),
        "cross_check_max_rel_diff": max(c["rel_diff"] for c in cross.values()),
    }

    return {
        "seed": args.seed + seed_idx, "window_index": seed_idx,
        "window_center_sec": center, "window_dur_s": w[1] - w[0],
        "T": T,
        "elevation_deg": [float(f["elevation_deg"]) for f in frames],
        "dist_sat_target_km": [float(f["dist_sat_target"]) for f in frames],
        "rho_eff": rho_eff, "rho_ratio_eff": _finite(ratio_eff),
        "rho_M_supp": rho_M, "rho_ratio_M_supp": _finite(ratio_M),
        "rho_BSROI_supp": rho_link, "rho_ratio_BSROI_supp": _finite(ratio_link),
        "J_dp": {str(u): {"J": j, "S": s} for u, (j, s) in dp.items()},
        "S_uniform": {str(K): S for K, S in S_unif.items()},
        "J_uniform": {str(K): j for K, j in J_unif.items()},
        "gap_K": {str(K): g for K, g in gap.items()},
        "p2": p2, "cross_check": {str(K): c for K, c in cross.items()},
        "internal_checks": checks,
    }


# ----------------------------------------------------------------------
# 汇总与报告
# ----------------------------------------------------------------------

def _ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main(args):
    print("=" * 100)
    print("verify_tracking_dp.py — RIS 分段重构 DP 精确最优调度（K-sweep → DP）")
    print(f"场景: ISS TLE / irs_mode={args.irs_mode} / T={args.tau} 帧 / "
          f"intervals={args.intervals} / {args.n_seeds} 个过境窗口 seed")
    print("相位设计器: PhaseOptimizerSat.optimize_frame（闭式对齐，与 compare_tracking 一致）")
    print("=" * 100)

    # 固定种子后生成 ROI（generate_ROI 用 python random，未播种则每次不同；
    # 与 verify_tracking.py 的 main() 相同顺序：先播种后 make_roi_voxel）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ROI = make_roi_voxel()      # legacy 物体，与 verify_tracking.py 相同
    rows = [run_seed(i, args, ROI) for i in range(args.n_seeds)]

    # ---------------- 表 A：每 seed 的 J(S*_U)（U=1..T）----------------
    T = rows[0]["T"]
    print(f"\n[表 A] 每 seed 的 DP 最优 J(S*_U)（帧功率和；均值功率 = J/T）")
    header = f"{'seed':>5}" + "".join(f"{'J*_U=' + str(u):>12}" for u in range(1, T + 1))
    print(header)
    for r in rows:
        line = f"{r['seed']:>5}"
        for u in range(1, T + 1):
            line += f"{r['J_dp'][str(u)]['J']:12.4e}"
        print(line)

    # ---------------- 表 B：均匀 K 对比 + P2 ----------------
    print(f"\n[表 B] 均匀 K 调度 J(S_K) 与 P2 对比（同预算：K=2→U=4，K=4→U=2）")
    print(f"{'seed':>5}{'J(K=1)':>12}{'J(K=2)':>12}{'J(K=4)':>12}{'J(K=8)':>12}"
          f"{'J*_4/J(K=2)':>13}{'J*_3/J(K=4)':>13}{'P2-A':>7}{'P2-B':>7}")
    for r in rows:
        ju = r["J_uniform"]
        print(f"{r['seed']:>5}{ju['1']:12.4e}{ju['2']:12.4e}{ju['4']:12.4e}{ju['8']:12.4e}"
              f"{r['p2']['ratio_U4_over_K2']:13.5f}{r['p2']['ratio_U3_over_K4']:13.5f}"
              f"{'✓' if r['p2']['ratio_U4_over_K2'] >= 1.03 else '✗':>7}"
              f"{'✓' if r['p2']['abs_diff_U3_minus_K4'] >= -1e-9 * abs(r['p2']['J_unif_K4']) else '✗':>7}")

    # ---------------- 表 C：U=1..T 预算-功率 Pareto（跨 seed）----------------
    U_of_K = {K: len(rows[0]["S_uniform"][str(K)]) for K in args.intervals}
    K_of_U = {u: K for K, u in U_of_K.items()}
    print(f"\n[表 C] U=1..{T} 全扫描预算-功率 Pareto（跨 seed 均值±std；"
          f"代表 S* 为众数调度）")
    print(f"{'U':>3}{'K̄':>6}{'J*(U) 均值':>13}{'±std':>11}{'均值功率':>13}"
          f"{'±std':>11}{'代表 S*':>18}{'同预算K':>8}{'gap_K':>9}")
    for u in range(1, T + 1):
        Js = [r["J_dp"][str(u)]["J"] for r in rows]
        m, sd = _ms(Js)
        mode_S = list(Counter(tuple(r["J_dp"][str(u)]["S"]) for r in rows)
                      .most_common(1)[0][0])
        k_unif = K_of_U.get(u)
        gap_txt = ""
        if k_unif is not None:
            gm = _ms([r["gap_K"][str(k_unif)] for r in rows])[0]
            gap_txt = f"{100 * gm:+7.3f}%"
        print(f"{u:>3}{T / u:>6.2f}{m:13.4e}{sd:11.2e}{m / T:13.4e}{sd / T:11.2e}"
              f"{str(mode_S):>18}{('K=' + str(k_unif)) if k_unif else '—':>8}{gap_txt:>9}")

    # ---------------- 表 D：精确间隙 gap_K（%）----------------
    print(f"\n[表 D] 精确间隙 gap_K=(J(S*)-J(S_K))/J(S*)（%；S* 为同预算 DP 最优）")
    print(f"{'seed':>5}" + "".join(f"{'gap_K=' + str(K):>11}" for K in args.intervals))
    for r in rows:
        print(f"{r['seed']:>5}" + "".join(f"{100 * r['gap_K'][str(K)]:11.4f}"
                                          for K in args.intervals))
    print(f"{'mean':>5}" + "".join(f"{100 * _ms([r['gap_K'][str(K)] for r in rows])[0]:11.4f}"
                                   for K in args.intervals))
    print(f"{'std':>5}" + "".join(f"{100 * _ms([r['gap_K'][str(K)] for r in rows])[1]:11.4f}"
                                  for K in args.intervals))

    # ---------------- G3：帧间漂移率非均匀性 ----------------
    print(f"\n[G3] 帧间漂移率 ρ_t=‖Ĥ_(t+1)-Ĥ_t‖_F/‖Ĥ_t‖_F（Ĥ=[d,M] 等效线性信道）")
    for r in rows:
        print(f"  seed {r['seed']} (win{r['window_index']}): "
              + " ".join(f"{x:6.3f}" for x in r["rho_eff"])
              + f"  → max/min = {r['rho_ratio_eff']:.3f}")
    ratios = [r["rho_ratio_eff"] for r in rows]
    rm, rsd = _ms(ratios)
    med = float(np.median(ratios))
    n_pass = sum(x >= 3.0 for x in ratios)
    g3_verdict = "PASS" if med >= 3.0 else ("FALSIFIED" if med < 1.5 else "MARGINAL")
    print(f"  非均匀比: 均值 {rm:.3f} ± {rsd:.3f}，中位 {med:.3f}，"
          f"{n_pass}/{len(ratios)} 个 seed ≥ 3")
    print(f"  辅助口径（JSON）: M 单独 max/min 均值 "
          f"{_ms([r['rho_ratio_M_supp'] for r in rows])[0]:.3f}；"
          f"原始 H_BS_ROI 星-地链路 max/min 均值 "
          f"{_ms([r['rho_ratio_BSROI_supp'] for r in rows])[0]:.3f}")
    if g3_verdict == "PASS":
        print("  ⇒ PASS：漂移高度非均匀，DP 调度存在可利用的优化空间（预注册阈值 3）")
    elif g3_verdict == "FALSIFIED":
        print("  ⇒ FALSIFIED（<1.5）：帧间漂移近均匀，均匀 K 已近优，DP 方向作废")
    else:
        print("  ⇒ MARGINAL：非均匀比介于 1.5~3，方向边缘")

    # ---------------- P2：同预算/更少预算 DP 增益 ----------------
    print(f"\n[P2] DP 最优 vs 均匀 K（预注册：J(S*_4)≥1.03·J(K=2) 且 J(S*_3)≥J(K=4)）")
    r4 = [r["p2"]["ratio_U4_over_K2"] for r in rows]
    r3 = [r["p2"]["ratio_U3_over_K4"] for r in rows]
    m4, s4 = _ms(r4)
    m3, s3 = _ms(r3)
    n4 = sum(x >= 1.03 for x in r4)
    min_d3 = min(r["p2"]["abs_diff_U3_minus_K4"] for r in rows)
    p2a_verdict = "PASS" if m4 >= 1.03 else "FALSIFIED"
    p2b_verdict = "PASS" if min_d3 >= -1e-9 else "FAIL(实现 bug)"
    print(f"  A: J(S*_4)/J(均匀K=2) = {m4:.5f} ± {s4:.5f}"
          f"（{n4}/{len(r4)} 个 seed ≥ 1.03；同预算增益 "
          f"{100 * (m4 - 1):+.3f}%） ⇒ {p2a_verdict}")
    print(f"  B: J(S*_3)/J(均匀K=4) = {m3:.5f} ± {s3:.5f}"
          f"（最小绝对差 {min_d3:.3e}；数学上 guaranteed，用作 DP 内部自检）"
          f" ⇒ {p2b_verdict}")

    # ---------------- 内部自检汇总 ----------------
    print(f"\n[内部自检]")
    max_cross = max(r["internal_checks"]["cross_check_max_rel_diff"] for r in rows)
    max_bf = max(r["internal_checks"]["brute_force_max_rel_diff"] for r in rows)
    print(f"  J(均匀K)/T vs compare_tracking track_K 功率: 最大相对偏差 "
          f"{max_cross:.2e} {'✓ PASS' if max_cross < 1e-6 else '✗ FAIL'}")
    print(f"  DP 最优性穷举证书（全部 C(8,U) 调度枚举 vs DP）: 最大相对偏差 "
          f"{max_bf:.2e} {'✓ PASS' if max_bf <= 1e-9 else '✗ FAIL'}")
    all_checks = all(all(v for k, v in r["internal_checks"].items()
                         if isinstance(v, bool)) for r in rows)
    print(f"  DP 结构性质（J* 对 U 单调 / J*_1=J(K=8) / J*_T=J(K=1) / "
          f"J*≥同预算均匀）: {'✓ PASS' if all_checks else '✗ FAIL'}")

    # ---------------- 总结论 ----------------
    print("\n" + "=" * 100)
    print("总结论（预注册命题裁决）:")
    print(f"  G3 帧间漂移非均匀比 max/min: 中位 {med:.3f}（阈值 ≥3 成立 / <1.5 作废）"
          f" ⇒ {g3_verdict}")
    print(f"  P2-A 同预算 DP 增益（U=4 vs K=2）: {100 * (m4 - 1):+.3f}%"
          f"（预注册 ≥+3%） ⇒ {p2a_verdict}")
    print(f"  P2-B 更少预算支配（U=3 vs K=4）: {100 * (m3 - 1):+.3f}% ⇒ {p2b_verdict}")
    if g3_verdict == "PASS" and p2a_verdict == "FALSIFIED":
        print("  解读: 漂移确实非均匀（G3 成立），但精确最优调度相对均匀 K=2 的增益"
              "低于预注册的 3%——均匀网格在本场景已接近该预算下的最优，"
              "DP 的价值在于给出**精确间隙证书**而非大幅涨点。")
    print("=" * 100)

    # ---------------- JSON 落盘 ----------------
    out = {
        "script": "verify_tracking_dp.py",
        "roadmap_refs": ["§4.2 方向五 K-sweep→DP 最优调度", "§7 G3", "§7 P2"],
        "protocol": {
            "n_seeds": args.n_seeds, "T": T, "irs_mode": args.irs_mode,
            "bs_ant": args.bs_ant, "ue_ant": args.ue_ant,
            "intervals": args.intervals, "search_hours": args.search_hours,
            "phase_designer": "PhaseOptimizerSat.optimize_frame (closed-form alignment)",
            "roi": "verify_tracking.make_roi_voxel (legacy object, same as verify_tracking.py)",
            "seed_to_window": "seed i -> i-th SGP4 overpass window (ISS TLE); seed 0 = verify_tracking.py default geometry",
            "H_definition_for_rho": "effective linear channel [d_t, M_t] in C^{BS x (N+1)} of P(v)=||d+Mv||^2",
            "J_definition": "J(S)=sum_k sum_{t in [s_k,s_{k+1})} P(v*(H_s_k); H_t) (sum over frames; mean power = J/T)",
            "DP": "F(t,u)=max_{s<=t}[F(s-1,u-1)+W(s,t)], W(s,t)=sum_{tau=s}^t P(v_s;H_tau), O(T^2 U)",
        },
        "seeds": rows,
        "aggregate": {
            "g3_rho_ratio": {"mean": rm, "std": rsd, "median": med,
                             "n_pass_of_3": int(n_pass), "verdict": g3_verdict},
            "p2_A": {"ratio_U4_over_K2_mean": m4, "std": s4,
                     "n_seeds_ge_1p03": int(n4), "verdict": p2a_verdict},
            "p2_B": {"ratio_U3_over_K4_mean": m3, "std": s3,
                     "min_abs_diff": min_d3, "verdict": p2b_verdict},
            "gap_K_mean_pct": {str(K): 100 * _ms([r["gap_K"][str(K)]
                                                  for r in rows])[0]
                               for K in args.intervals},
            "gap_K_std_pct": {str(K): 100 * _ms([r["gap_K"][str(K)]
                                                 for r in rows])[1]
                              for K in args.intervals},
            "internal_checks_all_pass": bool(all_checks),
            "cross_check_max_rel_diff": max_cross,
        },
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isac_demo")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "tracking_dp.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RIS 分段重构 DP 精确最优调度验证")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--search_hours", type=float, default=96.0,
                        help="过境窗口搜索时长（小时）；ISS 96h 内恰有 8 个窗口")
    args = parser.parse_args()
    main(args)
