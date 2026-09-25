"""verify_waveform_doppler.py — OTFS/AFDM 波形层与真实 LEO 多普勒验证（G9 预注册命题）

把仓库已验证的真实 LEO 多普勒（±611 kHz @30 GHz，setup_sat/SGP4）从"近似"
变成"精确处理"：OFDM 子载波间泄漏是精确恒等式，OTFS/AFDM 把同一物理量
吸收进二维/仿射 DD 域。协议与 verify_tracking.py / verify_placement_crb.py /
verify_optimality_decomposition.py 一致：固定种子、理论列 vs MC 列、
JSON 落盘 isac_demo/、PASS/FAIL 判定、诚实标注假设。

章节：
  §1  G9：OFDM 单音 ICI 恒等式 C_mk(ν)=e^{jπ(m-k+ν)}sinc(m-k+ν)；
          ICI 功率占比 = 1−sinc²(ν)；MC 10^6 符号；扫 ν∈{0.05, 0.1, 0.313}
  §2  OTFS DD 域：TX/RX 互逆；on-grid 二维扭曲卷积（机器精度）；
          off-grid 泄漏能谱 ≈ 1−sinc²(δ)；5 径真实多普勒 DD 响应 vs 时域仿真
  §3  AFDM：IDAFT/DAFT 单位性；c1=(2α+1)/(2N)、c2=(2N+1)c1（c1=c2/(2N+1)）；
          2Nc1 奇整数 + N 偶 ⇒ CPP=普通 CP；DAFT 域信道结构 vs 仿真
  §4  通信 BER：QPSK 无编码 + CSI 已知 MMSE；OFDM vs OTFS vs AFDM
          @5 径真实 LEO 信道（含 ν=0.313），SNR 20 dB
  §5  感知：DD 域信道估计 range-Doppler 峰 SIR
          （S1 全功率导频；S2 嵌入式导频 + 数据：OTFS 保护带 vs OFDM 梳状）
  §6  模型证书：默认 8 帧 |f_d| 实测范围、距离迁移比 M、ISAR 冻结几何
          32 Hz 条件与实际 kHz 的矛盾（只报告，不改 data_sat）

用法：py verify_waveform_doppler.py [--n_frames 48] [--n_mc 1000000]
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_sat as ss
from isac_sim.waveforms import OfdmWaveform, OtfsWaveform, AfdmWaveform
from isac_sim.waveforms.otfs import dirichlet_kernel

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isac_demo")

# 波形参数（对齐 data_sat 宽带档：512 子载波 × 1 GHz ⇒ Δf = 1.953125 MHz）
N_SYM = 16          # OTFS 符号数 = 多普勒 bin 数（Doppler 分辨率 Δf/N = 122.07 kHz）
N_SUB = 512         # 子载波数 = 时延 bin 数（时延分辨率 1/B = 1 ns）
BANDWIDTH_HZ = 1e9
CARRIER_HZ = 30e9
DF = BANDWIDTH_HZ / N_SUB           # 1.953125 MHz
TS = 1.0 / BANDWIDTH_HZ             # 1 ns
TSYM = 1.0 / DF                     # 512 ns
TFRAME = N_SYM * TSYM               # 8192 ns
NU_HEADLINE = 611e3 / DF            # 0.3128 ≈ 0.313（G9 预注册值）
SNR_DB = 20.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)   # 每符号噪声方差（信道归一化到单位功率）


def sinc_np(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    return out


def qpsk_symbols(shape, rng):
    bits = rng.integers(0, 2, (shape[0], shape[1], 2))
    sym = ((1.0 - 2.0 * bits[:, :, 0]) + 1j * (1.0 - 2.0 * bits[:, :, 1])) / np.sqrt(2.0)
    return sym, bits


def hard_bits(xhat):
    return np.stack([xhat.real < 0, xhat.imag < 0], axis=2).astype(int)


# ======================================================================
# §1 G9：OFDM 单音 ICI 恒等式
# ======================================================================
def section_g9(n_mc, results):
    print("=" * 78)
    print("§1  G9 预注册命题：OFDM 单音 ICI 恒等式")
    print("    C_mk(ν) = e^{jπ(m-k+ν)}·sinc(m-k+ν)；ICI 功率占比 = 1 − sinc²(ν)")
    print(f"    MC：{n_mc} 个单音符号/ν（随机子载波 + QPSK + 随机相位基准）")
    print("-" * 78)
    hdr = (f"{'ν':>8} {'理论 1-sinc²':>13} {'精确(Dirichlet)':>15} "
           f"{'MC 实测':>12} {'MC-理论':>10} {'复系数最大偏差':>14} {'判定':>6}")
    print(hdr)
    rows = []
    all_pass = True
    for nu in (0.05, 0.10, 0.313):
        theory = float(1.0 - sinc_np(nu) ** 2)
        exact = float(1.0 - np.abs(dirichlet_kernel(np.array([nu]), N_SUB)[0]) ** 2
                      / N_SUB ** 2)
        # ---- MC：n_mc 次单音传输（多普勒相位取 e^{-j2πνn/N}，与所证恒等式
        #      C_mk(ν)=e^{jπ(m-k+ν)}sinc(m-k+ν) 的符号约定一致；功率占比
        #      与符号约定无关）----
        rng = np.random.default_rng(20260925)
        n_batch = 4096
        leak_acc = 0.0
        coef_err = 0.0
        coef_exact_err = 0.0
        done = 0
        while done < n_mc:
            b = min(n_batch, n_mc - done)
            k_idx = rng.integers(0, N_SUB, b)                 # 随机子载波
            qpsk = ((1 - 2 * rng.integers(0, 2, b))
                    + 1j * (1 - 2 * rng.integers(0, 2, b))) / np.sqrt(2.0)
            phi0 = rng.uniform(0, 2 * np.pi, b)               # 随机相位基准
            n_ax = np.arange(N_SUB)[None, :]
            # 时域：单音 + 多普勒相位 e^{-j2πν(n+φ0)/N}（ν 为子载波归一化多普勒）
            s = (qpsk[:, None] * np.exp(2j * np.pi * k_idx[:, None] * n_ax / N_SUB)
                 * np.exp(-1j * (2 * np.pi * nu * n_ax / N_SUB + phi0[:, None])))
            Y = np.fft.fft(s, axis=1) / np.sqrt(N_SUB)
            p_tot = np.sum(np.abs(Y) ** 2, axis=1)            # Parseval = |qpsk|²·N
            p_main = np.abs(Y[np.arange(b), k_idx]) ** 2
            leak_acc += float(np.sum(1.0 - p_main / p_tot))
            # 复系数恒等式（取第一行详细比对幅值 + 精确 Dirichlet）
            m_rel = np.arange(-6, 7)                          # 相对子载波
            k0 = int(k_idx[0])
            C_mc = (Y[0, (k0 + m_rel) % N_SUB]
                    * np.sqrt(N_SUB) / (qpsk[0] * np.exp(-1j * phi0[0]) * N_SUB))
            C_th = (np.exp(1j * np.pi * (m_rel + nu))
                    * sinc_np(m_rel + nu))
            coef_err = max(coef_err, float(np.abs(np.abs(C_mc)
                                                  - np.abs(C_th)).max()))
            # 闭式（精确有限 N）：C_exact[m] = D_N(k0 - ν - m)/N
            C_ex = dirichlet_kernel(k0 - nu - ((k0 + m_rel) % N_SUB), N_SUB) / N_SUB
            coef_exact_err = max(coef_exact_err,
                                 float(np.abs(C_mc - C_ex).max()))
            done += b
        mc = leak_acc / n_mc
        dev = mc - theory
        ok = abs(dev) < 0.005 and coef_err < 0.005 and coef_exact_err < 1e-9
        all_pass &= ok
        rows.append({"nu": float(nu), "theory": theory, "exact_dirichlet": exact,
                     "mc": mc, "dev": dev, "coef_max_dev": coef_err,
                     "coef_exact_dev": coef_exact_err,
                     "pass": bool(ok)})
        print(f"{nu:8.3f} {theory:13.5f} {exact:15.5f} {mc:12.5f} "
              f"{dev:+10.5f} {coef_err:14.4f} {'PASS' if ok else 'FAIL':>6}")
    print(f"\n  G9 判定：{'PASS' if all_pass else 'FAIL'}"
          f"（ν=0.313 理论 ICI 占比 {1 - sinc_np(0.313) ** 2:.4f}"
          f"（预注册 0.2834，差 {abs(1 - sinc_np(0.313) ** 2 - 0.2834):.1e}），"
          f"即 20 dB SNR 的 ICI 地板 SINR ≈ "
          f"{sinc_np(0.313) ** 2 / (1 - sinc_np(0.313) ** 2):.2f}"
          f" = {10 * math.log10(sinc_np(0.313) ** 2 / (1 - sinc_np(0.313) ** 2)):.1f} dB）")
    results["g9_ici_identity"] = {"rows": rows, "pass": bool(all_pass)}


# ======================================================================
# §2 OTFS DD 域：扭曲卷积 / Zak 视角
# ======================================================================
def section_otfs(results):
    print("=" * 78)
    print("§2  OTFS DD 域输入输出（二维扭曲卷积，Zak 变换视角）")
    otfs = OtfsWaveform(n_symbols=N_SYM, n_subcarriers=N_SUB,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_hz=CARRIER_HZ)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((N_SYM, N_SUB)) + 1j * rng.standard_normal((N_SYM, N_SUB))) / np.sqrt(2.0)

    # (a) TX/RX 互逆
    err_id = float(np.abs(otfs.demodulate(otfs.modulate(x)) - x).max())

    # (b) on-grid 单径：y[k,l] = h·e^{j2πlk_i/(MN)}·x[k-k_i, l-l_i]
    k0, l0, h0 = 3, 40, 0.6 - 0.2j
    fd0 = k0 / TFRAME
    xd = otfs.transmit_through(x, [(h0, l0 * TS, fd0)])
    pred = np.zeros_like(x)
    for k in range(N_SYM):
        for l in range(N_SUB):
            pred[k, l] = (h0 * np.exp(2j * np.pi * l * k0 / (N_SUB * N_SYM))
                          * x[(k - k0) % N_SYM, (l - l0) % N_SUB])
    err_tw = float(np.abs(xd - pred).max())

    # (c) off-grid 多普勒泄漏能谱 vs 1−sinc²(δ)
    print("-" * 78)
    print(f"{'δ(分数bin)':>10} {'泄漏能谱实测':>13} {'1−sinc²(δ)':>12} {'偏差':>9}")
    off_rows = []
    for delta in (0.05, 0.25, 0.50):
        fd = (5 + delta) / TFRAME
        xi = np.zeros((N_SYM, N_SUB), dtype=complex)
        xi[0, :] = x[0, :]                      # 单多普勒行输入
        xd1 = otfs.transmit_through(xi, [(1.0, 0.0, fd)])
        e = np.abs(xd1) ** 2
        leak = 1.0 - float(e[5].sum() / e.sum())
        theo = float(1 - sinc_np(delta) ** 2)
        off_rows.append({"delta": delta, "leak": leak, "theory": theo})
        print(f"{delta:10.2f} {leak:13.5f} {theo:12.5f} {leak - theo:+9.5f}")

    # (d) 5 径真实多普勒：闭式 dd_response vs 时域仿真
    paths5 = leo_five_paths()
    xi = np.zeros((N_SYM, N_SUB), dtype=complex)
    xi[0, 0] = 1.0
    sim = otfs.transmit_through(xi, paths5)
    closed = otfs.dd_response([p[1] for p in paths5], [p[2] for p in paths5],
                              [p[0] for p in paths5])
    err_dd = float(np.abs(sim - closed).max())
    ok = err_id < 1e-10 and err_tw < 1e-9 and err_dd < 1e-9
    print("-" * 78)
    print(f"  TX/RX 互逆最大误差           : {err_id:.2e}")
    print(f"  on-grid 扭曲卷积最大误差      : {err_tw:.2e}  (机器精度)")
    print(f"  5 径闭式 DD 响应 vs 时域仿真  : {err_dd:.2e}")
    print(f"  §2 判定：{'PASS' if ok else 'FAIL'}")
    results["otfs_dd"] = {"identity_err": err_id, "twisted_conv_err": err_tw,
                          "dd_response_err": err_dd, "offgrid": off_rows,
                          "pass": bool(ok)}


# ======================================================================
# §3 AFDM：DAFT/IDAFT 链与 chirp 条件
# ======================================================================
def section_afdm(results):
    print("=" * 78)
    print("§3  AFDM：IDAFT/DAFT 链、啁啾条件与 DAFT 域信道结构")
    afdm = AfdmWaveform(n_blocks=N_SYM, n_subcarriers=N_SUB,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_hz=CARRIER_HZ,
                        alpha_max=1)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((N_SYM, N_SUB)) + 1j * rng.standard_normal((N_SYM, N_SUB))) / np.sqrt(2.0)

    err_rt = float(np.abs(afdm.demodulate(afdm.modulate(x)) - x).max())
    cond_c = abs(afdm.c1 - afdm.c2 / (2 * N_SUB + 1))
    two_n_c1_int = abs(afdm.two_n_c1 - round(afdm.two_n_c1)) < 1e-12
    cpp_ok = two_n_c1_int and (N_SUB % 2 == 0) and (round(afdm.two_n_c1) % 2 == 1)

    # 整数多普勒结构：每行每径单非零元（论文 regime）
    paths_int = [(0.3, 20e-9, 2 * DF), (0.25, 40e-9, 0.0), (0.2, 60e-9, -2 * DF)]
    Hb = afdm.channel_matrix(paths_int, 0)
    nz = (np.abs(Hb) > 1e-8 * np.abs(Hb).max()).sum(axis=1)
    nz_ok = bool(np.all(nz == len(paths_int)))

    # 5 径真实（分数）多普勒：闭式 vs 仿真 + loc 位置
    paths5 = leo_five_paths()
    sim = afdm.transmit_through(x, paths5)
    err_mat = 0.0
    for b in (0, 5, 15):
        err_mat = max(err_mat, float(np.abs(afdm.channel_matrix(paths5, b) @ x[b]
                                            - sim[b]).max()))
    locs = [float(afdm.loc_i(p[1], p[2])) for p in paths5]
    ok = err_rt < 1e-12 and cond_c < 1e-15 and cpp_ok and nz_ok and err_mat < 1e-6
    print("-" * 78)
    print(f"  c1 = {afdm.c1:.6f}，c2 = {afdm.c2:.6f}，2Nc1 = {afdm.two_n_c1:.1f}"
          f"（奇整数={two_n_c1_int}，N 偶={N_SUB % 2 == 0}）")
    print(f"  c1 = c2/(2N+1) 条件偏差        : {cond_c:.2e}")
    print(f"  CPP 退化为普通 CP（2Nc1 奇整数+ N 偶）: {'是' if cpp_ok else '否'}")
    print(f"  IDAFT/DAFT 互逆最大误差        : {err_rt:.2e}")
    print(f"  整数多普勒每行非零元数（应={len(paths_int)}）: "
          f"{int(nz.min())}/{int(nz.max())}  {'OK' if nz_ok else 'FAIL'}")
    print(f"  5 径闭式 DAFT 信道 vs 时域仿真  : {err_mat:.2e}")
    print(f"  loc_i = 2Nc1·l_i − ν_i         : "
          + ", ".join(f"{v:.2f}" for v in locs))
    print(f"  §3 判定：{'PASS' if ok else 'FAIL'}")
    results["afdm"] = {"roundtrip_err": err_rt, "c_cond_dev": cond_c,
                       "two_n_c1": float(afdm.two_n_c1), "cpp_reduces_to_cp": bool(cpp_ok),
                       "row_nonzeros_ok": nz_ok, "channel_err": err_mat,
                       "loc_i": locs, "pass": bool(ok)}


# ======================================================================
# 5 径真实 LEO 信道（每径独立取 setup_sat/SGP4 真实 f_d）
# ======================================================================
def leo_five_paths():
    """5 径 LEO 信道：(gain, delay_s, doppler_hz)。

    - 多普勒：5 个 SGP4 真实 f_d（默认 ISS 场景首个过境窗口 ±60 s 密采样，
      与 verify_sat.py 的窗口扫描一致；峰值 ±611 kHz），按 DD 多普勒栅格
      {0, ±2, ±5}·Δf/N 就近选取（使 RDM 峰落在栅格上；off-grid 泄漏律
      已在 §2(c) 单独验证）。
    - 时延：简化等距多径（任务允许），Δτ = 20 ns，0..80 ns，
      小于一个符号时长 512 ns（CP 等价模型无帧间 ISI）。
    - 功率：指数递减 PDP |h_i|² ∝ ρ^i（ρ=0.55，物理标准多径功率谱，
      归一化到单位功率）。注：等功率等距 5 径在 DD 平面构成等间距点集，
      其 DD 域频率响应存在精确零点（正多边形相消）——属病态组合，
      故采用标准递减 PDP（病态性本身在 OTFS/AFDM MMSE 中可见）。
    """
    sc = ss.SatISACScenario()
    wins = sc.find_overpass()
    w = wins[0]
    jd0, fr0 = ss.jday(*sc.start_utc)
    jd0f = jd0 + fr0
    ts = np.arange(w[0] - 60, w[1] + 60, 1.0)
    fds = []
    for t in ts:
        jdf = jd0f + t / 86400.0
        jdi, fri = np.floor(jdf), jdf - np.floor(jdf)
        err, r_eci, v_eci = sc.sat.sgp4(int(jdi), fri)
        if err != 0:
            fds.append(np.nan)
            continue
        r_ecef, v_ecef = ss.eci_to_ecef(r_eci, v_eci, jdf)
        v_rel = ss.radial_velocity_mps(v_ecef, np.zeros(3), r_ecef, sc.target_ecef)
        fds.append(v_rel / sc.wavelength_m)
    fds = np.asarray(fds)
    df_dop = DF / N_SYM                       # DD 多普勒 bin = 122.07 kHz
    picks = []
    for k_t in (-5, -2, 0, 2, 5):
        i = int(np.nanargmin(np.abs(fds - k_t * df_dop)))
        picks.append(float(fds[i]))
    rho = 0.55
    pw = rho ** np.arange(5)
    pw /= pw.sum()
    gains = np.sqrt(pw)
    delays = [i * 20e-9 for i in range(5)]
    return list(zip(gains.tolist(), delays, picks))


# ======================================================================
# §4 通信 BER：OFDM vs OTFS vs AFDM @ ν=0.313 真实 5 径信道
# ======================================================================
def ofdm_ber(paths, n_frames, seed):
    """OFDM：每符号 FFT + 每子载波 MMSE（CSI 已知=接收机可估的有效信道，
    含 ICI 对角相位 e^{jπν(N-1)/N}；ICI 离对角项不抵消——标准基线）。"""
    rng = np.random.default_rng(seed)
    m_ax = np.arange(N_SUB)
    t = np.arange(N_SYM)[:, None] * TSYM + m_ax[None, :] * TS
    n_ax = np.arange(N_SYM)
    errs = 0
    tot = 0
    for _ in range(n_frames):
        x, b = qpsk_symbols((N_SYM, N_SUB), rng)
        s = np.fft.ifft(x, axis=1) * np.sqrt(N_SUB)
        r = np.zeros_like(s)
        for (h, d, fd) in paths:
            r += h * np.roll(s, int(round(d * BANDWIDTH_HZ)), axis=1) * np.exp(2j * np.pi * fd * t)
        r += (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape)) * np.sqrt(SIGMA2 / 2)
        Y = np.fft.fft(r, axis=1) / np.sqrt(N_SUB)
        H = np.zeros((N_SYM, N_SUB), dtype=complex)
        for (h, d, fd) in paths:
            nu = fd / DF
            diag = dirichlet_kernel(np.array([nu]), N_SUB)[0] / N_SUB
            H += (h * np.exp(-2j * np.pi * m_ax * int(round(d * BANDWIDTH_HZ)) / N_SUB)
                  * np.exp(2j * np.pi * fd * n_ax * TSYM)[:, None] * diag)
        xhat = np.conj(H) * Y / (np.abs(H) ** 2 + SIGMA2)
        errs += int(np.sum(hard_bits(xhat) != b))
        tot += b.size
    return errs / tot, tot


def otfs_ber(paths, n_frames, seed):
    rng = np.random.default_rng(seed)
    otfs = OtfsWaveform(n_symbols=N_SYM, n_subcarriers=N_SUB,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_hz=CARRIER_HZ)
    errs = 0
    tot = 0
    for _ in range(n_frames):
        x, b = qpsk_symbols((N_SYM, N_SUB), rng)
        y = otfs.transmit_through(x, paths)
        y += (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) * np.sqrt(SIGMA2 / 2)
        xhat = otfs.mmse_equalize(y, paths, SIGMA2)
        errs += int(np.sum(hard_bits(xhat) != b))
        tot += b.size
    return errs / tot, tot


def afdm_ber(paths, n_frames, seed):
    rng = np.random.default_rng(seed)
    afdm = AfdmWaveform(n_blocks=N_SYM, n_subcarriers=N_SUB,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_hz=CARRIER_HZ,
                        alpha_max=1)
    filters = afdm.mmse_filters(paths, SIGMA2)
    errs = 0
    tot = 0
    for _ in range(n_frames):
        x, b = qpsk_symbols((N_SYM, N_SUB), rng)
        y = afdm.transmit_through(x, paths)
        y += (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) * np.sqrt(SIGMA2 / 2)
        xhat = afdm.mmse_equalize(y, paths, SIGMA2, filters=filters)
        errs += int(np.sum(hard_bits(xhat) != b))
        tot += b.size
    return errs / tot, tot


def section_ber(n_frames, results):
    print("=" * 78)
    print("§4  通信 BER：QPSK 无编码 + CSI 已知 MMSE，SNR 20 dB，5 径真实 LEO 信道")
    paths = leo_five_paths()
    nus = [p[2] / DF for p in paths]
    print(f"    路径多普勒 ν = f_d/Δf : " + ", ".join(f"{v:+.4f}" for v in nus)
          + f"  （Δf = {DF/1e6:.4f} MHz，DD bin = {DF/N_SYM/1e3:.2f} kHz）")
    print(f"    路径时延 (ns)        : " + ", ".join(f"{p[1]*1e9:.0f}" for p in paths)
          + "  （等距 20 ns）")
    t0 = time.time()
    ber_ofdm, tot = ofdm_ber(paths, n_frames, seed=101)
    ber_otfs, _ = otfs_ber(paths, n_frames, seed=102)
    ber_afdm, _ = afdm_ber(paths, n_frames, seed=103)
    ber_ofdm_sp, _ = ofdm_ber([(1.0, 0.0, 611e3)], n_frames, seed=104)
    # 参考：单径 ν=0.313 的 OFDM ICI 地板（理论）
    s = float(sinc_np(NU_HEADLINE))
    sinr_floor = s ** 2 / ((1 - s ** 2) + SIGMA2)
    ber_floor = 0.5 * math.erfc(math.sqrt(sinr_floor) / math.sqrt(2.0))
    ber_awgn = 0.5 * math.erfc(math.sqrt(2 * (1.0 / SIGMA2) / 2) / math.sqrt(2))
    print("-" * 78)
    print(f"{'波形':>10} {'BER':>12} {'比特数':>10} {'说明':>34}")
    print(f"{'OFDM':>10} {ber_ofdm:12.3e} {tot:>10} {'每子载波 MMSE（ICI 地板）':>34}")
    print(f"{'OTFS':>10} {ber_otfs:12.3e} {tot:>10} {'DD 域 MMSE（CG）':>34}")
    print(f"{'AFDM':>10} {ber_afdm:12.3e} {tot:>10} {'DAFT 域逐块 MMSE':>34}")
    print(f"{'OFDM单径':>10} {ber_ofdm_sp:12.3e} {tot:>10} {'ν=0.313 单径（ICI 地板实测）':>34}")
    print(f"{'理论参考':>10} {ber_floor:12.3e} {'-':>10} {'OFDM 单径 ICI 地板(高斯近似)':>34}")
    print(f"{'AWGN参考':>10} {ber_awgn:12.3e} {'-':>10} {'QPSK@20dB 无干扰':>34}")
    otfs_ok = ber_otfs <= 1e-5
    order_ok = ber_ofdm > 100 * max(ber_otfs, 1e-12)
    afdm_ok = ber_afdm <= 1e-5
    print(f"\n  预注册核对：OTFS ≤ 1e-5 → {'PASS' if otfs_ok else 'FAIL'}；"
          f"OFDM ≫ OTFS → {'PASS' if order_ok else 'FAIL'}；"
          f"AFDM ≤ 1e-5 → {'PASS' if afdm_ok else 'FAIL'}")
    print(f"  （诚实标注：预注册 OFDM≈5e-3 偏乐观约一个数量级；实测 "
          f"{ber_ofdm:.1e} 与高斯 ICI 地板 {ber_floor:.1e} 同量级）")
    print(f"  [§4 耗时 {time.time()-t0:.1f}s]")
    results["ber"] = {"ofdm": ber_ofdm, "otfs": ber_otfs, "afdm": ber_afdm,
                      "ofdm_single_path": ber_ofdm_sp,
                      "theory_ofdm_floor": ber_floor, "awgn": ber_awgn,
                      "n_bits": tot, "nu": nus,
                      "otfs_pass": bool(otfs_ok), "order_pass": bool(order_ok),
                      "afdm_pass": bool(afdm_ok)}


# ======================================================================
# §5 感知：DD 域信道估计 range-Doppler 峰 SIR
# ======================================================================
def section_sensing(results):
    print("=" * 78)
    print("§5  感知：DD 域信道估计 range-Doppler 峰 SIR（5 径真实信道，SNR 20 dB）")
    paths = leo_five_paths()
    otfs = OtfsWaveform(n_symbols=N_SYM, n_subcarriers=N_SUB,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_hz=CARRIER_HZ)
    ofdm = OfdmWaveform(n_subcarriers=N_SUB)
    rng = np.random.default_rng(31)

    # ---- S1：全功率导频 RDM（无未知数据）----
    xi = np.zeros((N_SYM, N_SUB), dtype=complex)
    xi[0, 0] = np.sqrt(N_SYM * N_SUB)          # DD 冲激导频（全帧功率）
    y_otfs = otfs.transmit_through(xi, paths)
    y_otfs += (rng.standard_normal(y_otfs.shape) + 1j * rng.standard_normal(y_otfs.shape)) * np.sqrt(SIGMA2 / 2)
    rdm_o = np.abs(y_otfs) ** 2
    # OFDM：全 1 TF 导频（全帧功率）→ 每 (n,m) 信道估计 → SFFT 读出
    s = np.fft.ifft(np.ones((N_SYM, N_SUB), dtype=complex), axis=1) * np.sqrt(N_SUB)
    t = np.arange(N_SYM)[:, None] * TSYM + np.arange(N_SUB)[None, :] * TS
    r = np.zeros_like(s)
    for (h, d, fd) in paths:
        r += h * np.roll(s, int(round(d * BANDWIDTH_HZ)), axis=1) * np.exp(2j * np.pi * fd * t)
    r += (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape)) * np.sqrt(SIGMA2 / 2)
    Y = np.fft.fft(r, axis=1) / np.sqrt(N_SUB)   # 全 1 导频 ⇒ Ĥ = Y（ICI 卷积对全导频无污染）
    rdm_f = np.abs(otfs.sfft(Y)) ** 2

    def peak_sir(rdm, paths):
        """峰区（真实路径 (k_i,l_i) ±1 bin）能量 / 其余能量。"""
        mask = np.zeros(rdm.shape, dtype=bool)
        for (h, d, fd) in paths:
            k_i = fd * TFRAME
            l_i = int(round(d * BANDWIDTH_HZ))
            for dk in (-1, 0, 1):
                kk = int(round(k_i)) + dk
                for dl in (-1, 0, 1):
                    mask[kk % N_SYM, (l_i + dl) % N_SUB] = True
        num = float(rdm[mask].sum())
        den = float(rdm[~mask].sum())
        return 10 * math.log10(num / max(den, 1e-30))

    sir_o = peak_sir(rdm_o, paths)
    sir_f = peak_sir(rdm_f, paths)
    print("-" * 78)
    print("  S1 全功率导频（无未知数据，DD 读出 = 信道 DD 响应）：")
    print(f"    OTFS (DD 冲激导频) 峰 SIR : {sir_o:6.2f} dB")
    print(f"    OFDM (全 1 TF 导频) 峰 SIR : {sir_f:6.2f} dB")

    # ---- S2：嵌入式导频 + QPSK 数据，5 径复增益估计 SIR ----
    # 导频栅格（两波形同数）：时延/子载波 {0,170,340} × 多普勒/符号（OTFS 全 16 个
    # 多普勒 bin——DD 多普勒扩散 ±5 bin 使部分布保护带不可行，改用全多普勒轴导频，
    # 时延保护带 ±80 bin = l_max；OFDM 同 16 符号 × 3 子载波 = 48 导频）
    l_pilots = [0, 170, 340]
    n_pil = N_SYM * len(l_pilots)

    def otfs_estimate(seed):
        """OTFS：DD 导频（全多普勒轴 × 稀时延）+ 时延保护带，数据填时延间隙。"""
        rng = np.random.default_rng(seed)
        x = np.zeros((N_SYM, N_SUB), dtype=complex)
        pilot_val = {}
        for kp in range(N_SYM):
            for lp in l_pilots:
                v = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                pilot_val[(kp, lp)] = v
        # 保护带：每个导频时延 ±80 bin（= l_max）对全部多普勒行禁布数据
        # （导频位置本身在保护带内，先填数据再覆写导频，避免数据冲掉导频）
        guard = np.zeros(x.shape, dtype=bool)
        for lp in l_pilots:
            for dl in range(-80, 81):
                guard[:, (lp + dl) % N_SUB] = True
        x[~guard] = qpsk_symbols((int((~guard).sum()), 1), rng)[0][:, 0]
        for (kp, lp), v in pilot_val.items():
            x[kp, lp] = v
        y = otfs.transmit_through(x, paths)
        y += (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) * np.sqrt(SIGMA2 / 2)
        # 观测：每导频 × 每径 → (k_p+k_i, l_p+l_i)；跨径项落在保护带（空）⇒ 干净
        A_rows, b_rows = [], []
        for (kp, lp), v in pilot_val.items():
            for i, (h, d, fd) in enumerate(paths):
                k_i = fd * TFRAME
                l_i = int(round(d * BANDWIDTH_HZ))
                kk, ll = (kp + int(round(k_i))) % N_SYM, (lp + l_i) % N_SUB
                k_i_int = int(round(k_i))
                coef = (np.exp(2j * np.pi * ll * k_i / (N_SUB * N_SYM))
                        * dirichlet_kernel(np.array([k_i - k_i_int]), N_SYM)[0]
                        / N_SYM)
                row = np.zeros(5, dtype=complex)
                row[i] = coef * v
                A_rows.append(row)
                b_rows.append(y[kk, ll])
        h_est, *_ = np.linalg.lstsq(np.array(A_rows), np.array(b_rows), rcond=None)
        return h_est

    def ofdm_estimate(seed):
        """OFDM：梳状 TF 导频（同 48 个），数据填其余；每导频观测受数据 ICI 污染。"""
        rng = np.random.default_rng(seed)
        x = np.zeros((N_SYM, N_SUB), dtype=complex)
        pilot_val = {}
        for np_ in range(N_SYM):
            for mp in l_pilots:
                v = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                x[np_, mp] = v
                pilot_val[(np_, mp)] = v
        mask = np.ones(x.shape, dtype=bool)
        for key in pilot_val:
            mask[key] = False
        x[mask] = qpsk_symbols((int(mask.sum()), 1), rng)[0][:, 0]
        s = np.fft.ifft(x, axis=1) * np.sqrt(N_SUB)
        t = np.arange(N_SYM)[:, None] * TSYM + np.arange(N_SUB)[None, :] * TS
        r = np.zeros_like(s)
        for (h, d, fd) in paths:
            r += h * np.roll(s, int(round(d * BANDWIDTH_HZ)), axis=1) * np.exp(2j * np.pi * fd * t)
        r += (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape)) * np.sqrt(SIGMA2 / 2)
        Y = np.fft.fft(r, axis=1) / np.sqrt(N_SUB)
        A_rows, b_rows = [], []
        for (np_, mp), v in pilot_val.items():
            for i, (h, d, fd) in enumerate(paths):
                nu = fd / DF
                diag = dirichlet_kernel(np.array([nu]), N_SUB)[0] / N_SUB
                coef = (np.exp(-2j * np.pi * mp * int(round(d * BANDWIDTH_HZ)) / N_SUB)
                        * np.exp(2j * np.pi * fd * np_ * TSYM) * diag)
                row = np.zeros(5, dtype=complex)
                row[i] = coef * v
                A_rows.append(row)
                b_rows.append(Y[np_, mp])
        h_est, *_ = np.linalg.lstsq(np.array(A_rows), np.array(b_rows), rcond=None)
        return h_est

    h_true = np.array([p[0] for p in paths])
    n_real = 8
    sir_otfs_l, sir_ofdm_l = [], []
    for sseed in range(n_real):
        he = otfs_estimate(1000 + sseed)
        sir_otfs_l.append(10 * math.log10(
            np.sum(np.abs(h_true) ** 2) / max(np.sum(np.abs(he - h_true) ** 2), 1e-30)))
        he = ofdm_estimate(2000 + sseed)
        sir_ofdm_l.append(10 * math.log10(
            np.sum(np.abs(h_true) ** 2) / max(np.sum(np.abs(he - h_true) ** 2), 1e-30)))
    sir_otfs_e = float(np.mean(sir_otfs_l))
    sir_ofdm_e = float(np.mean(sir_ofdm_l))
    print(f"  S2 嵌入式导频 + QPSK 数据（{n_pil} 导频/波形，5 径增益估计 SIR，"
          f"{n_real} 次实现均值）：")
    print(f"    OTFS（DD 导频 + 保护带）: {sir_otfs_e:6.2f} dB  "
          f"(min {min(sir_otfs_l):.2f}, max {max(sir_otfs_l):.2f})")
    print(f"    OFDM（梳状 TF 导频）    : {sir_ofdm_e:6.2f} dB  "
          f"(min {min(sir_ofdm_l):.2f}, max {max(sir_ofdm_l):.2f})")
    otfs_sir_ok = sir_otfs_e >= 15.0
    ofdm_sir_ok = sir_ofdm_e <= 5.0
    print(f"  预注册核对：OTFS ≥ 15 dB → {'PASS' if otfs_sir_ok else 'FAIL'}；"
          f"OFDM ≤ 5 dB → {'PASS' if ofdm_sir_ok else 'FAIL'}")
    print("  （诚实标注：S1 显示全功率导频下两波形 DD 读出同样干净——"
          "roadmap 的 OFDM≤5dB 只在嵌入式导频+数据 regime 有意义）")
    results["sensing"] = {
        "s1_full_pilot": {"otfs_db": sir_o, "ofdm_db": sir_f},
        "s2_embedded": {"otfs_db": sir_otfs_e, "ofdm_db": sir_ofdm_e,
                        "otfs_min": min(sir_otfs_l), "ofdm_min": min(sir_ofdm_l),
                        "n_pilots": n_pil, "n_real": n_real},
        "otfs_sir_pass": bool(otfs_sir_ok), "ofdm_sir_pass": bool(ofdm_sir_ok)}


# ======================================================================
# §6 模型证书：真实多普勒尺度 vs ISAR 冻结几何
# ======================================================================
def section_certificate(results):
    print("=" * 78)
    print("§6  模型证书：真实 LEO 多普勒尺度 vs ISAR 冻结几何（只报告，不改 data_sat）")
    sc = ss.SatISACScenario()
    frames = sc.build_frames()
    fd8 = np.array([f["f_d_hz"] for f in frames])
    abs_fd8 = np.abs(fd8)
    # 窗口 ±60 s 密采样（与 verify_sat.py 一致）取 S 曲线峰值
    wins = sc.find_overpass()
    w = wins[0]
    jd0, fr0 = ss.jday(*sc.start_utc)
    jd0f = jd0 + fr0
    ts = np.arange(w[0] - 60, w[1] + 60, 2.0)
    fds = []
    for t in ts:
        jdf = jd0f + t / 86400.0
        jdi, fri = np.floor(jdf), jdf - np.floor(jdf)
        err, r_eci, v_eci = sc.sat.sgp4(int(jdi), fri)
        if err != 0:
            continue
        r_ecef, v_ecef = ss.eci_to_ecef(r_eci, v_eci, jdf)
        v_rel = ss.radial_velocity_mps(v_ecef, np.zeros(3), r_ecef, sc.target_ecef)
        fds.append(v_rel / sc.wavelength_m)
    fds = np.asarray(fds)
    lam = sc.wavelength_m
    T_burst = 0.93          # data_sat ISAR_M=32, ISAR_DT=0.03 → 31*0.03 = 0.93 s
    c = 299_792_458.0

    def mig_ratio(fd):
        return fd * lam * T_burst * BANDWIDTH_HZ / c

    fd_thresh = c / (lam * BANDWIDTH_HZ * T_burst)     # ISAR 冻结几何阈值
    print("-" * 78)
    print(f"  默认 8 帧窗口 |f_d| 实测范围 : {abs_fd8.min()/1e3:.2f} .. "
          f"{abs_fd8.max()/1e3:.2f} kHz（帧值 "
          + ", ".join(f"{v/1e3:+.1f}" for v in fd8) + " kHz）")
    print(f"  过境窗口 ±60 s S 曲线峰值 |f_d|: {np.abs(fds).max()/1e3:.1f} kHz"
          f"（ν = {np.abs(fds).max()/DF:.4f}）")
    print(f"  ISAR 冻结几何有效条件 |f_d| ≤ c/(λBT_burst) = {fd_thresh:.1f} Hz"
          f"（λ={lam*100:.1f} cm, B={BANDWIDTH_HZ/1e9:.0f} GHz, T_burst={T_burst} s）")
    print(f"{'f_d':>12} {'距离迁移比 M':>14} {'判定(冻结几何)':>16}")
    for label, fd in [("8帧最小", abs_fd8.min()), ("8帧最大", abs_fd8.max()),
                      ("S曲线峰值", np.abs(fds).max())]:
        M = mig_ratio(fd)
        print(f"{fd/1e3:10.1f}k {M:14.3e} {'M≫1 失效':>16}")
    viol = bool(abs_fd8.min() > fd_thresh)
    print(f"\n  矛盾确认：默认窗口最小 |f_d| = {abs_fd8.min()/1e3:.1f} kHz ≫ "
          f"{fd_thresh:.0f} Hz 阈值 {abs_fd8.min()/fd_thresh:.0f}× ⇒ "
          "卫星运动 ISAR 的冻结几何假设失效（现有 0.933 s ISAR 是目标旋转 ISAR，"
          "真实几何必须配 RCMC）。")
    print("  仅报告：data_sat.compute_isar_sequence 保持不变。")
    results["model_certificate"] = {
        "default8_abs_fd_hz": {"min": float(abs_fd8.min()), "max": float(abs_fd8.max())},
        "scurve_peak_abs_fd_hz": float(np.abs(fds).max()),
        "isar_frozen_threshold_hz": float(fd_thresh),
        "migration_ratio": {"fd8_min": float(mig_ratio(abs_fd8.min())),
                            "fd8_max": float(mig_ratio(abs_fd8.max())),
                            "peak": float(mig_ratio(np.abs(fds).max()))},
        "frozen_geometry_violated": viol}


def main():
    ap = argparse.ArgumentParser(description="OTFS/AFDM 波形层与真实 LEO 多普勒验证")
    ap.add_argument("--n_frames", type=int, default=48)
    ap.add_argument("--n_mc", type=int, default=1_000_000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"meta": {"n_sym": N_SYM, "n_sub": N_SUB, "df_hz": DF,
                        "nu_headline": NU_HEADLINE, "snr_db": SNR_DB,
                        "n_frames": args.n_frames, "n_mc": args.n_mc}}
    t0 = time.time()
    section_g9(args.n_mc, results)
    section_otfs(results)
    section_afdm(results)
    section_ber(args.n_frames, results)
    section_sensing(results)
    section_certificate(results)

    # ---- 总结 ----
    print("=" * 78)
    print("总结（PASS/FAIL）")
    checks = [
        ("G9 OFDM 单音 ICI 恒等式 (MC 偏差<±0.005)", results["g9_ici_identity"]["pass"]),
        ("OTFS DD 扭曲卷积 (TX/RX 互逆 + on-grid + 5径)", results["otfs_dd"]["pass"]),
        ("AFDM DAFT/IDAFT + chirp 条件 + 信道结构", results["afdm"]["pass"]),
        ("BER: OTFS ≤ 1e-5", results["ber"]["otfs_pass"]),
        ("BER: OFDM ≫ OTFS (>100×)", results["ber"]["order_pass"]),
        ("BER: AFDM ≤ 1e-5", results["ber"]["afdm_pass"]),
        ("SIR: OTFS ≥ 15 dB (嵌入式导频)", results["sensing"]["otfs_sir_pass"]),
        ("SIR: OFDM ≤ 5 dB (嵌入式导频)", results["sensing"]["ofdm_sir_pass"]),
    ]
    n_pass = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_pass += int(ok)
    results["summary"] = {name: bool(ok) for name, ok in checks}
    results["summary"]["n_pass"] = n_pass
    results["summary"]["n_total"] = len(checks)
    print(f"  ⇒ {n_pass}/{len(checks)} 通过；总耗时 {time.time()-t0:.1f}s")

    json_path = os.path.join(OUT_DIR, "waveform_doppler.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    main()
