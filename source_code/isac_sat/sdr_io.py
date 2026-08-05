"""
sdr_io.py — SDR IQ 数据格式与 IO（硬件预留）

工程目标：让 ISAC 感知管线能吃 SDR 采集的真实接收信号。
当前无硬件，先定义数据格式 + 模拟 IQ 生成 + 回放接口；
有硬件后 `capture_live()` 接 pyrtlsdr / uhd 即可。

IQ 文件格式（.npz）：
  iq       : complex64 数组（基带 IQ）
  fs_hz    : 采样率
  fc_hz    : 中心频率
  utc      : 采集时间（ISO 字符串）
  meta     : 额外元数据（dict）

用法：
  # 生成模拟 IQ（从 ISAC 仿真场景）
  save_iq(iq, path, fs_hz=2e6, fc_hz=30e9, utc="2026-08-05T12:00:00Z")

  # 回放
  iq, meta = load_iq(path)

  # 硬件（预留，需 pip install pyrtlsdr）
  # from sdr_io import capture_live; iq, meta = capture_live(device="rtlsdr", fs=2e6, fc=915e6, dur=1.0)
"""

import os
import time
import numpy as np


def save_iq(iq, path, fs_hz=2e6, fc_hz=30e9, utc=None, meta=None):
    """保存 IQ 数据（.npz）。iq: complex64 数组。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if utc is None:
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    np.savez_compressed(
        path,
        iq=np.asarray(iq, dtype=np.complex64),
        fs_hz=np.float64(fs_hz),
        fc_hz=np.float64(fc_hz),
        utc=utc,
        meta=meta or {},
    )
    return path


def load_iq(path):
    """加载 IQ 数据，返回 (iq complex64, meta dict)。"""
    d = np.load(path, allow_pickle=True)
    meta = {
        "fs_hz": float(d["fs_hz"]),
        "fc_hz": float(d["fc_hz"]),
        "utc": str(d["utc"]),
    }
    m = d["meta"]
    if m.ndim > 0 and m.shape == ():
        meta.update(m.item() if isinstance(m.item(), dict) else {})
    elif isinstance(m, dict):
        meta.update(m)
    return d["iq"], meta


def simulate_isac_iq(channels, frame_idx, roi_voxel, X, phase=None,
                     n_ue=4, noise_scale=1.0, seed=0):
    """从 ISAC 仿真生成"模拟 SDR IQ"（基带复信号）。

    用现有信道模型计算单帧接收信号，作为 SDR 采集的基带 IQ 内容。
    这是无硬件时的保真验证：IQ 应能经 sdr_ingest 恢复出相同的接收特征。
    """
    import setup_sat as ss
    from data_sat import calculate_value_sat

    Ht = channels.channels_per_frame[frame_idx]
    t_rel = channels.frames[frame_idx]["t_sec"]
    if phase is None:
        n_irs = Ht["H_ROI_IRS"].shape[1] if "H_ROI_IRS" in Ht else 0
        phase = (np.random.RandomState(seed).rand(n_irs) * 2 * np.pi
                 if n_irs > 0 else np.array([]))
    Y = calculate_value_sat(roi_voxel, torch.tensor(phase, dtype=torch.float32),
                            X, Ht, channels.power_sigma if hasattr(channels, "power_sigma") else 0.01,
                            t_rel, channels.wavelength_m)
    # Y: [UE_ant, 1] 复数 → 展平为 IQ 序列
    iq = Y.squeeze(-1).detach().cpu().numpy().astype(np.complex64)
    return iq


def simulate_isac_iq_wideband(ROI_np, target_ecef, ground_ecef, wavelength_m,
                               k=512, bw_hz=1e9, snr_db=None, seed=0):
    """生成宽带时域 IQ（模拟 SDR 采集的时域回波）。

    物理：目标体素散射 → 频域响应 H(f) → 时域回波 = IFFT(H)。
    SDR 采集的是时域 IQ，sdr_ingest 用 FFT 恢复频域响应。
    """
    import setup_sat as ss
    from data_sat import make_roi_local
    C_MS = ss.C_LIGHT_KM * 1000.0
    local = make_roi_local()
    occ = np.argwhere(ROI_np > 0.5)
    if len(occ) == 0:
        return np.zeros(k, dtype=np.complex64)
    p = target_ecef[None, :] + local[occ[:, 0] * 256 + occ[:, 1] * 16 + occ[:, 2], :] / 1000.0
    rel = p - p.mean(axis=0)
    u = ground_ecef - target_ecef
    u = u / (np.linalg.norm(u) + 1e-12)
    d_proj = (rel @ u) * 1000.0
    # 与 data_sat.compute_range_profile 保持完全一致的时延模型
    d_bs = 695e3 + d_proj * 0.1
    d_ue = 50e3 + d_proj
    tau = (d_bs + d_ue) / C_MS
    f = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, k)
    H = np.exp(-2j * np.pi * f[:, None] * tau[None, :]).sum(axis=1)
    if snr_db is not None:
        rng = np.random.RandomState(seed)
        sig_pow = np.mean(np.abs(H) ** 2)
        n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
    return np.fft.ifft(H).astype(np.complex64)  # 时域 IQ


def capture_live(device="rtlsdr", fs_hz=2e6, fc_hz=915e6, duration_s=1.0,
                 gain=40, **kwargs):
    """实时采集（硬件预留）。接入 pyrtlsdr / uhd 后启用。

    返回 (iq, meta)。
    """
    if device == "rtlsdr":
        try:
            from rtlsdr import RtlSdr
        except ImportError:
            raise RuntimeError("需要 pip install pyrtlsdr 才能使用 RTL-SDR")
        sdr = RtlSdr()
        sdr.sample_rate = fs_hz
        sdr.center_freq = fc_hz
        sdr.gain = gain
        n = int(fs_hz * duration_s)
        iq = sdr.read_samples(n)
        sdr.close()
        return np.asarray(iq, dtype=np.complex64), {
            "fs_hz": fs_hz, "fc_hz": fc_hz,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
        }
    raise ValueError(f"未支持的设备: {device}")
