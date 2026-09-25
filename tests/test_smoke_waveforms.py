"""isac_sim 波形层冒烟测试（OTFS / AFDM / OFDM 真实多普勒）。

运行：python tests/test_smoke_waveforms.py  （或 pytest tests/）
原则：小样本、秒级、numpy-only；每层一个物理 sanity 断言。
覆盖 G9 预注册命题的快速版：OFDM 单音 ICI 恒等式（ν=0.313）。
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isac_sim.waveforms import OfdmWaveform, OtfsWaveform, AfdmWaveform

N_SYM, N_SUB = 16, 512
DF = 1e9 / N_SUB                 # 1.953125 MHz
TS = 1e-9
TSYM = 1.0 / DF
TFRAME = N_SYM * TSYM


def _sinc(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    return out


def test_otfs_roundtrip_and_twisted_convolution():
    """OTFS：TX/RX 互逆 + on-grid 单径 DD 响应 = 二维扭曲卷积（机器精度）。"""
    otfs = OtfsWaveform(n_symbols=N_SYM, n_subcarriers=N_SUB)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((N_SYM, N_SUB))
         + 1j * rng.standard_normal((N_SYM, N_SUB))) / np.sqrt(2.0)
    xr = otfs.demodulate(otfs.modulate(x))
    assert np.abs(xr - x).max() < 1e-10, np.abs(xr - x).max()

    k0, l0, h0 = 3, 40, 0.6 - 0.2j
    xd = otfs.transmit_through(x, [(h0, l0 * TS, k0 / TFRAME)])
    pred = np.zeros_like(x)
    for k in range(N_SYM):
        for l in range(N_SUB):
            pred[k, l] = (h0 * np.exp(2j * np.pi * l * k0 / (N_SUB * N_SYM))
                          * x[(k - k0) % N_SYM, (l - l0) % N_SUB])
    assert np.abs(xd - pred).max() < 1e-9, np.abs(xd - pred).max()


def test_afdm_roundtrip_and_chirp_condition():
    """AFDM：IDAFT/DAFT 互逆；c1 = c2/(2N+1)；2Nc1 奇整数 + N 偶 ⇒ CPP=普通CP。"""
    afdm = AfdmWaveform(n_blocks=N_SYM, n_subcarriers=N_SUB, alpha_max=1)
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((N_SYM, N_SUB))
         + 1j * rng.standard_normal((N_SYM, N_SUB))) / np.sqrt(2.0)
    xr = afdm.demodulate(afdm.modulate(x))
    assert np.abs(xr - x).max() < 1e-10, np.abs(xr - x).max()
    assert abs(afdm.c1 - afdm.c2 / (2 * N_SUB + 1)) < 1e-15
    two_n_c1 = afdm.two_n_c1
    assert abs(two_n_c1 - round(two_n_c1)) < 1e-12 and round(two_n_c1) % 2 == 1
    assert N_SUB % 2 == 0
    # DAFT 域信道：整数多普勒下每行每径单非零元（路径分离）
    paths = [(0.3, 20e-9, 2 * DF), (0.25, 40e-9, 0.0), (0.2, 60e-9, -2 * DF)]
    Hb = afdm.channel_matrix(paths, 0)
    nz = (np.abs(Hb) > 1e-8 * np.abs(Hb).max()).sum(axis=1)
    assert np.all(nz == len(paths)), (int(nz.min()), int(nz.max()))


def test_ici_identity_physics():
    """物理 sanity（G9 快速版）：OFDM 单音 ICI 功率占比 = 1 − sinc²(ν)。

    ν=0.313（真实 LEO 多普勒 611 kHz @ Δf=1.953 MHz）：理论 0.2835，
    MC（2×10^5 单音符号）偏差 < ±0.005。
    """
    nu = 611e3 / DF
    theory = float(1.0 - _sinc(nu) ** 2)
    rng = np.random.default_rng(42)
    n_mc, batch = 200_000, 2048
    leak = 0.0
    done = 0
    n_ax = np.arange(N_SUB)[None, :]
    while done < n_mc:
        b = min(batch, n_mc - done)
        k_idx = rng.integers(0, N_SUB, b)
        qpsk = ((1 - 2 * rng.integers(0, 2, b))
                + 1j * (1 - 2 * rng.integers(0, 2, b))) / np.sqrt(2.0)
        phi0 = rng.uniform(0, 2 * np.pi, b)
        s = (qpsk[:, None] * np.exp(2j * np.pi * k_idx[:, None] * n_ax / N_SUB)
             * np.exp(-1j * (2 * np.pi * nu * n_ax / N_SUB + phi0[:, None])))
        Y = np.fft.fft(s, axis=1) / np.sqrt(N_SUB)
        p_tot = np.sum(np.abs(Y) ** 2, axis=1)
        p_main = np.abs(Y[np.arange(b), k_idx]) ** 2
        leak += float(np.sum(1.0 - p_main / p_tot))
        done += b
    mc = leak / n_mc
    assert abs(mc - theory) < 0.005, (mc, theory)


def test_otfs_doppler_concentration_vs_ofdm():
    """物理 sanity：ν=0.313 时 OTFS DD 响应能量集中（>98%）于真实 bin，
    而 OFDM 单音仅 sinc²(ν)≈71.7% 留在本子载波（28.3% 泄漏为 ICI）。"""
    otfs = OtfsWaveform(n_symbols=N_SYM, n_subcarriers=N_SUB)
    ofdm = OfdmWaveform(n_subcarriers=N_SUB)
    fd = 611e3
    # OTFS：DD 冲激 → 5 径中的单径响应
    xi = np.zeros((N_SYM, N_SUB), dtype=complex)
    xi[0, 0] = 1.0
    y = otfs.transmit_through(xi, [(1.0, 0.0, fd)])
    k_i = fd * TFRAME
    e = np.abs(y) ** 2
    conc = float(e[int(round(k_i)) % N_SYM].sum() / e.sum())
    assert conc > 0.98, conc
    # OFDM：单音泄漏比例 = 1 − sinc²(ν)
    rng = np.random.default_rng(7)
    n_ax = np.arange(N_SUB)[None, :]
    s = (np.exp(2j * np.pi * 100 * n_ax / N_SUB)
         * np.exp(-1j * 2 * np.pi * (fd / DF) * n_ax / N_SUB))[0]
    Y = np.fft.fft(s) / np.sqrt(N_SUB)
    leak = 1.0 - float(np.abs(Y[100]) ** 2 / np.sum(np.abs(Y) ** 2))
    assert abs(leak - (1.0 - float(_sinc(fd / DF) ** 2))) < 0.005, leak


TESTS = [test_otfs_roundtrip_and_twisted_convolution,
         test_afdm_roundtrip_and_chirp_condition,
         test_ici_identity_physics,
         test_otfs_doppler_concentration_vs_ofdm]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        sys.exit(f"{failed} test(s) failed")
    print(f"ALL PASS ({len(TESTS)} waveform smoke tests)")
