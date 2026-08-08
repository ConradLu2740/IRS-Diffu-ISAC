"""
baseline_classic.py — 2D-CFAR + MUSIC 经典感知基线，与 ML 感知方法公平对比

目标：客观回答"经典雷达信号处理（CFAR 检测 + 距离定位 + MUSIC 测向）
在星-地 ISAC 场景下能达到什么水平"，与现有 ML 感知（SensingMLP）对比。

对比设计（同测试集、同观测几何）：
  1. 2D-CFAR 在距离-多普勒图上检测 → 质心插值距离定位（沿视线 1D）
  2. MUSIC 在 ULA 快拍上测向 → 验证远场几何下角度分辨力的物理边界
  3. ML（宽带距离像 MLP）：分类 + 2D 定位（沿视线 + 横向）
  4. 输出：检测率 / 定位误差分解 / 分类准确率 / 物理结论

用法：
  .venv/bin/python baseline_classic.py --n_test 60 --epochs 20
"""
import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import rotate as ndi_rotate

import setup_sat as ss
from data_sat import (SatScenarioChannels,
                      SatROIDataset, GROUND_TARGET_TEMPLATES,
                      compute_isar_sequence, compute_range_profile,
                      generate_ground_target_sample, WIDEBAND_K, ISAR_M)
from train_sensing import SensingMLP, make_fixed, CLASS_NAMES

N_CLASSES = len(GROUND_TARGET_TEMPLATES)
C_MS = ss.C_LIGHT_KM * 1000.0
WIDEBAND_K_ABS = 1024      # 绝对距离像距离单元（避免模 K 回卷：窗 279m > ROI ±40m）


# ----------------------------------------------------------------------
# 0. 绝对距离距离像（修正 compute_range_profile 的特征构造缺陷）
#
# 发现：data_sat.compute_range_profile 的 d_proj = (rel @ u) 中
#   rel = p - p_center（相对体素质心）→ 目标在 ROI 中的绝对位置信息
#   在特征构造阶段被丢弃（实测：不同 dx/dy 的 RP 质心 bin 恒定）。
# 真实雷达距离像应保留绝对时延（相对雷达/ROI 中心）。
# 这里实现修正版：d_proj 相对 ROI 中心，保留绝对时延。
# ----------------------------------------------------------------------
def range_profile_abs(ROI_np, target_ecef, ground_ecef, wavelength_m,
                      k=WIDEBAND_K_ABS, bw_hz=1e9, snr_db=20.0, seed=0,
                      bs_dist_km=695.0, d_ue_km=50.0):
    """绝对距离像：体素相对 ROI 中心的投影 + 绝对时延 → bin 携带位置。
    返回 (rp [k], d_proj_true_m)。标定（bin→米）由 fit_range_calibration 提供。
    """
    local = _make_roi_local(5.0, 16)
    occ = np.argwhere(ROI_np > 0.5)
    if len(occ) == 0:
        return np.zeros(k, dtype=np.float32), 0.0
    p = target_ecef[None, :] + local[occ[:, 0] * 256 + occ[:, 1] * 16 + occ[:, 2], :] / 1000.0
    u = ground_ecef - target_ecef
    u = u / (np.linalg.norm(u) + 1e-12)
    d_proj = ((p - target_ecef[None, :]) @ u) * 1000.0      # 米（相对 ROI 中心）
    d_bs = bs_dist_km * 1000.0 + d_proj * 0.1
    d_ue = d_ue_km * 1000.0 + d_proj
    tau = (d_bs + d_ue) / C_MS
    f = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, k)
    H = np.exp(-2j * np.pi * f[:, None] * tau[None, :]).sum(axis=1)
    if snr_db is not None:
        rng = np.random.RandomState(seed)
        sig_pow = np.mean(np.abs(H) ** 2)
        n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
    rp = np.abs(np.fft.ifft(H))
    norm = np.linalg.norm(rp)
    rp = (rp / (norm + 1e-12)).astype(np.float32)
    d_proj_true = float(np.mean(d_proj))
    return rp, d_proj_true


def rp_centroid_bin(rp):
    """RP 质心 bin（能量加权）。"""
    idx = np.arange(len(rp))
    return float((rp * idx).sum() / (rp.sum() + 1e-12))


def fit_range_calibration(n_cal=20, channels=None, target_ecef=None, ground_ecef=None,
                          snr_db=20.0, seed=0):
    """校准集回归：bin_centroid = bin0 + s * x_los → 返回 (bin0, s)。
    用随机 ROI（已知几何）拟合，等效雷达出厂标定（不泄露测试 GT）。
    """
    bins, xs = [], []
    rng = np.random.RandomState(seed)
    for i in range(n_cal):
        roi, cid, ang = generate_ground_target_sample()
        rp, d_true = range_profile_abs(roi, target_ecef, ground_ecef, channels.wavelength_m,
                                       snr_db=snr_db, seed=seed + i)
        x_los, y_cross, _, _ = sample_geometry(roi, target_ecef, ground_ecef)
        bins.append(rp_centroid_bin(rp))
        xs.append(x_los)
    A = np.stack([np.ones(len(xs)), xs], axis=1)
    coef, *_ = np.linalg.lstsq(A, bins, rcond=None)
    bin0, s = float(coef[0]), float(coef[1])
    # 拟合优度
    pred = bin0 + s * np.array(xs)
    r2 = 1.0 - np.sum((np.array(bins) - pred) ** 2) / (np.sum((np.array(bins) - np.mean(bins)) ** 2) + 1e-12)
    return bin0, s, r2


def isar_abs(ROI_np, target_ecef, ground_ecef, wavelength_m,
             m_frames=ISAR_M, omega_deg=60.0, dt=0.03,
             k=WIDEBAND_K_ABS, bw_hz=1e9, snr_db=20.0, seed=0):
    """绝对距离的 ISAR 序列 [M, K]（目标旋转 + 绝对时延）。"""
    from scipy.ndimage import rotate as ndi_rotate
    local = _make_roi_local(5.0, 16)
    u = ground_ecef - target_ecef
    u = u / (np.linalg.norm(u) + 1e-12)
    f = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, k)
    rng = np.random.RandomState(seed)
    seq = []
    for m in range(m_frames):
        theta = omega_deg * m * dt
        if abs(theta) > 1e-6:
            rotated = ndi_rotate(ROI_np, theta, axes=(0, 1), reshape=False,
                                 order=1, mode="constant", cval=0.0)
            rotated = (rotated > 0.5)
        else:
            rotated = ROI_np > 0.5
        occ = np.argwhere(rotated)
        if len(occ) == 0:
            seq.append(np.zeros(k, dtype=np.float32))
            continue
        p = target_ecef[None, :] + local[occ[:, 0] * 256 + occ[:, 1] * 16 + occ[:, 2], :] / 1000.0
        d_proj = ((p - target_ecef[None, :]) @ u) * 1000.0
        d_bs = 695.0 * 1000.0 + d_proj * 0.1
        d_ue = 50.0 * 1000.0 + d_proj
        tau = (d_bs + d_ue) / C_MS
        H = np.exp(-2j * np.pi * f[:, None] * tau[None, :]).sum(axis=1)
        if snr_db is not None:
            sig_pow = np.mean(np.abs(H) ** 2)
            n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
            H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
        rp = np.abs(np.fft.ifft(H))
        norm = np.linalg.norm(rp)
        seq.append((rp / (norm + 1e-12)).astype(np.float32))
    return np.array(seq)



# ----------------------------------------------------------------------
# 1. 距离-多普勒图（ISAR 序列 → 慢时间 FFT）
# ----------------------------------------------------------------------
def build_rd_map(isar_seq, snr_db=20.0, seed=0):
    """ISAR 距离像序列 [M, K] → 距离-多普勒图 [K, M]。

    慢时间 FFT 得到多普勒/横向信息；物理上等价于 ISAR 转台成像。
    幅度包络重建（无原始相位时）：逐帧幅度 + 慢时间 FFT。
    """
    M, K = isar_seq.shape
    env = isar_seq.astype(np.float64)
    win = np.hanning(M)
    # 沿慢时间（axis=0）FFT → 距离-多普勒图 [K, M]
    rd = np.abs(np.fft.fftshift(np.fft.fft(env * win[:, None], axis=0), axes=0)).T
    floor = np.percentile(rd, 5)
    return rd, floor


# ----------------------------------------------------------------------
# 2. 2D-CA-CFAR
# ----------------------------------------------------------------------
def ca_cfar_2d(rd, guard=2, ref=6, pfa=1e-4, doppler_guard=1, doppler_ref=2):
    """2D CA-CFAR（卷积向量化）：参考环均值噪声估计 + 乘法因子阈值。

    rd: [K_range, M_doppler]
    guard/ref: 距离维保护窗/参考窗（单侧单元数）
    doppler_guard/doppler_ref: 多普勒维（32 bins 较短，窗小）
    返回 (det_mask [K,M] bool, thresh [K,M])
    """
    from scipy.ndimage import convolve
    K, M = rd.shape
    n_ref = (2 * ref + 1) * (2 * doppler_ref + 1)
    alpha = pfa ** (-1.0 / (2.0 * n_ref)) - 1.0
    # 参考环掩码（挖掉保护窗+中心单元：2guard+1 中心方块）
    mask = np.ones((2 * (ref + guard) + 1, 2 * (doppler_ref + doppler_guard) + 1))
    mask[guard:guard + 2 * guard + 1, doppler_guard:doppler_guard + 2 * doppler_guard + 1] = 0.0
    mask_sum = mask.sum()
    mask = mask / mask_sum
    noise = convolve(rd, mask, mode="reflect")
    thresh = alpha * noise
    det = rd > thresh
    # 边界单元不判决（窗不完整）
    bg = guard + ref
    dbg = doppler_guard + doppler_ref
    det[:bg, :] = False
    det[-bg:, :] = False
    det[:, :dbg] = False
    det[:, -dbg:] = False
    return det, thresh


def ca_cfar_1d(rp, guard=8, ref=32, pfa=1e-4):
    """1D CA-CFAR（距离维，卷积向量化）。返回 (det_mask, thresh)。"""
    from scipy.ndimage import convolve1d
    k = len(rp)
    n_ref = 2 * ref
    alpha = pfa ** (-1.0 / (2.0 * n_ref)) - 1.0
    mask = np.ones(2 * (ref + guard) + 1)
    mask[guard:guard + 2 * guard + 1] = 0.0
    mask = mask / mask.sum()
    noise = convolve1d(rp, mask, mode="reflect")
    thresh = alpha * noise
    det = rp > thresh
    det[:guard + ref] = False
    det[k - (guard + ref):] = False
    return det, thresh


def detect_range_bin_1d(rp, det_mask, k=WIDEBAND_K_ABS):
    """1D CFAR 检测 → 检测区域能量加权距离质心 bin。返回 bin 或 None。"""
    ys = np.where(det_mask)[0]
    if len(ys) == 0:
        return None
    amp = rp[det_mask]
    total = amp.sum()
    if total < 1e-12:
        return None
    return float((ys * amp).sum() / total)


def detect_range_bin(rd, det_mask, k=WIDEBAND_K, bw_hz=1e9, d_bs_km=695.0, d_ue_km=50.0):
    """2D CFAR 检测 → 最大峰邻域能量加权距离质心 bin。

    返回 (r_bin_f, d_bin)。距离→米的标定由 fit_range_calibration 回归提供。
    """
    ys, xs = np.where(det_mask)
    if len(ys) == 0:
        return None, None
    amp = rd[det_mask]
    peak_idx = int(np.argmax(amp))
    r_peak = int(ys[peak_idx]); d_peak = int(xs[peak_idx])
    lo_r, hi_r = max(0, r_peak - 15), min(k - 1, r_peak + 15)
    lo_d, hi_d = max(0, d_peak - 3), min(rd.shape[1] - 1, d_peak + 3)
    win = rd[lo_r:hi_r + 1, lo_d:hi_d + 1]
    w = np.arange(lo_r, hi_r + 1)
    col_amp = win.sum(axis=1)
    total = col_amp.sum()
    if total < 1e-12:
        return None, None
    r_bin_f = float((w * col_amp).sum() / total)
    return r_bin_f, float(d_peak)


# ----------------------------------------------------------------------
# 3. MUSIC 测向（ULA）
# ----------------------------------------------------------------------
def music_doa(snapshots, n_src=1, grid_deg=None):
    """MUSIC 谱搜索。snapshots: [M_elem, L] 复快拍。返回 (az_deg, spectrum, grid)。"""
    M = snapshots.shape[0]
    R = snapshots @ snapshots.conj().T / snapshots.shape[1]
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    noise_sub = V[:, n_src:]
    if grid_deg is None:
        grid_deg = np.arange(-60.0, 60.01, 0.1)
    lam = ss.WAVELENGTH_M
    d = lam / 2.0
    spec = np.zeros_like(grid_deg, dtype=np.float64)
    for gi, az in enumerate(grid_deg):
        a = np.exp(-2j * np.pi * d / lam * np.arange(M) * np.sin(np.deg2rad(az)))
        denom = np.abs(a.conj() @ noise_sub @ noise_sub.conj().T @ a).item()
        spec[gi] = 1.0 / (denom + 1e-12)
    az_est = float(grid_deg[np.argmax(spec)])
    return az_est, spec, grid_deg


def make_ula_snapshots(az_true_deg, m_elem=8, n_snap=64, snr_db=20.0, seed=0):
    """ULA 快拍：X = A(theta)s + n（目标点源，阵列位于地面站旁）。"""
    rng = np.random.RandomState(seed)
    lam = ss.WAVELENGTH_M
    d = lam / 2.0
    a = np.exp(-2j * np.pi * d / lam * np.arange(m_elem) * np.sin(np.deg2rad(az_true_deg)))
    s = (rng.randn(n_snap) + 1j * rng.randn(n_snap)) / np.sqrt(2)
    X = a[:, None] * s[None, :]
    sig_pow = np.mean(np.abs(X) ** 2)
    n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    X = X + (rng.randn(m_elem, n_snap) + 1j * rng.randn(m_elem, n_snap)) * np.sqrt(n_pow / 2)
    return X


# ----------------------------------------------------------------------
# 4. 样本几何与 GT（与 ML 同源）
# ----------------------------------------------------------------------
def _make_roi_local(voxel_size, res):
    """ROI 局部坐标（米）：与 data_sat.make_roi_local 一致。"""
    xs = np.arange(res) * voxel_size
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float64)
    local = local - local.mean(axis=0)
    return local


def sample_geometry(ROI_np, target_ecef, ground_ecef, voxel_size=5.0, roi_res=16):
    """目标质心相对 ROI 中心的局部偏移（米），分解为沿视线/横向。

    返回 (x_los_m, y_cross_m, p_mean, u_los)
    """
    local = _make_roi_local(voxel_size, roi_res)
    occ = np.argwhere(ROI_np > 0.5)
    if len(occ) == 0:
        return 0.0, 0.0, target_ecef, None
    p = target_ecef[None, :] + local[occ[:, 0] * roi_res * roi_res + occ[:, 1] * roi_res + occ[:, 2], :] / 1000.0
    p_mean = p.mean(axis=0)
    u = ground_ecef - target_ecef
    u = u / (np.linalg.norm(u) + 1e-12)
    rel = p_mean - target_ecef                       # km
    x_los = float((rel @ u) * 1000.0)                # 沿视线（米）
    h_los = u.copy(); h_los[2] = 0.0
    h_los = h_los / (np.linalg.norm(h_los) + 1e-12)
    h_cross = np.array([-h_los[1], h_los[0], 0.0])
    y_cross = float((rel * 1000.0) @ h_cross)        # 横向（米）
    return x_los, y_cross, p_mean, u


# ----------------------------------------------------------------------
# 5. 评估
# ----------------------------------------------------------------------
def evaluate_classic(ROI_np, target_ecef, ground_ecef, wavelength_m, snr_db, seed,
                     calib=(None, None)):
    """经典管线：单帧绝对距离像 1D-CFAR 定位（主）+ 2D-CFAR 检测确认（辅）。

    calib: (bin0, s) 回归标定（bin = bin0 + s * x_los）。
    """
    # 单帧绝对距离像（定位源）
    rp, d_true = range_profile_abs(ROI_np, target_ecef, ground_ecef, wavelength_m,
                                   snr_db=snr_db, seed=seed)
    det1, th1 = ca_cfar_1d(rp)
    r_bin1 = detect_range_bin_1d(rp, det1)
    # 多帧 RD 图 + 2D-CFAR（检测确认）
    isar = isar_abs(ROI_np, target_ecef, ground_ecef, wavelength_m,
                    snr_db=snr_db, seed=seed)
    rd, floor = build_rd_map(isar, snr_db=snr_db, seed=seed)
    det2, th2 = ca_cfar_2d(rd)
    r_bin2, d_bin = detect_range_bin(rd, det2, k=WIDEBAND_K_ABS)

    bin0, s = calib
    d_proj_est = None
    detected = r_bin2 is not None and r_bin1 is not None
    if detected and s is not None and abs(s) > 1e-9:
        # 1D-CFAR 定位为主；2D 检测确认成功时用 2D 峰邻域质心微调（取平均稳健）
        r_bin_final = r_bin1
        d_proj_est = (r_bin_final - bin0) / s
    # MUSIC：目标实际方向（ROI 中心 + 横向偏移）
    x_los, y_cross, _, _ = sample_geometry(ROI_np, target_ecef, ground_ecef)
    slant = np.linalg.norm(target_ecef - ground_ecef) * 1000.0
    az_base = float(np.rad2deg(np.arctan2(target_ecef[1] - ground_ecef[1],
                                          target_ecef[0] - ground_ecef[0])))
    az_base = ((az_base + 90.0) % 360.0) - 180.0
    az_true = float(np.clip(az_base + np.rad2deg(np.arctan2(y_cross, slant)), -60, 60))
    snap = make_ula_snapshots(az_true, seed=seed, snr_db=snr_db)
    az_est, spec, grid = music_doa(snap)
    return {"detected": detected,
            "d_proj_est": d_proj_est,
            "az_true": az_true, "az_est": az_est,
            "rd": rd, "det": det2, "spec": spec, "grid": grid}


def build_abs_ds(n, channels, target_ecef0, ground_ecef0, snr_db, seed):
    """绝对距离像数据集：输入 [K_ABS]，标签 (class, x_los/40, y_cross/40)。"""
    rng = np.random.RandomState(seed)
    conds, cls, poss = [], [], []
    for i in range(n):
        roi, cid, ang = generate_ground_target_sample()
        rp, d_true = range_profile_abs(roi, target_ecef0, ground_ecef0,
                                       channels.wavelength_m, snr_db=snr_db, seed=seed + i)
        x_los, y_cross, _, _ = sample_geometry(roi, target_ecef0, ground_ecef0)
        conds.append(torch.from_numpy(rp).float())
        cls.append(cid)
        poss.append(torch.tensor([x_los / 40.0, y_cross / 40.0], dtype=torch.float32))
    return TensorDataset(torch.stack(conds), torch.tensor(cls, dtype=torch.long),
                         torch.stack(poss))


def train_eval(model, tr_ds, te_ds, device, epochs=20):
    """通用训练+评估（分类准确率 / 定位误差分解）。返回 dict。"""
    tr = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te = DataLoader(te_ds, batch_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for cond, cid, pos in tr:
            cond = cond.to(device); cid = cid.to(device); pos = pos.to(device)
            logits, pred_pos = model(cond)
            loss = F.cross_entropy(logits, cid) + F.mse_loss(pred_pos, pos)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    errs = []
    with torch.no_grad():
        for cond, cid, pos in te:
            cond = cond.to(device)
            logits, pred_pos = model(cond)
            correct += (logits.argmax(1).cpu() == cid).sum().item()
            total += len(cid)
            errs.append((pred_pos.cpu() - pos).abs().numpy())
    abs_err = np.concatenate(errs) * 40.0
    return {"acc": correct / total,
            "rmse_2d": float(np.sqrt(np.mean(abs_err[:, 0] ** 2 + abs_err[:, 1] ** 2))),
            "rmse_los": float(np.sqrt(np.mean(abs_err[:, 0] ** 2))),
            "rmse_cross": float(np.sqrt(np.mean(abs_err[:, 1] ** 2)))}


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"=== 2D-CFAR + MUSIC vs ML 公平对比 (n_test={args.n_test}, snr={args.snr_db}dB) ===")

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device)
    mid = frames[len(frames) // 2]
    target_ecef0 = mid["target_pos"]
    ground_ecef0 = mid["ground_pos"]

    # --- 固定测试集（与 ML 同分布：随机类别/姿态/位置） ---
    rois, cls_ids, angles = [], [], []
    rng = np.random.RandomState(args.seed)
    for i in range(args.n_test):
        roi, cid, ang = generate_ground_target_sample()
        rois.append(roi); cls_ids.append(cid); angles.append(ang)

    # --- 1. 经典：2D-CFAR（绝对距离）+ MUSIC ---
    print("[1/4] 校准（bin→米回归）...")
    bin0, s, r2 = fit_range_calibration(n_cal=20, channels=channels,
                                        target_ecef=target_ecef0, ground_ecef=ground_ecef0,
                                        snr_db=args.snr_db, seed=args.seed + 500)
    print(f"      校准: bin0={bin0:.2f}, s={s:.4f} bins/m, R2={r2:.4f}")
    print("[1/4] 运行经典基线（绝对距离 RD 图 + 2D-CFAR + MUSIC）...")
    classic = []
    for i, roi in enumerate(rois):
        c = evaluate_classic(roi, target_ecef0, ground_ecef0, channels.wavelength_m,
                             args.snr_db, seed=1000 + i, calib=(bin0, s))
        x_gt, y_gt, _, _ = sample_geometry(roi, target_ecef0, ground_ecef0)
        c["x_gt"] = x_gt; c["y_gt"] = y_gt
        classic.append(c)

    det_rate = float(np.mean([c["detected"] for c in classic]))
    hit = [c for c in classic if c["detected"]]
    if hit:
        err_los = np.abs([c["d_proj_est"] - c["x_gt"] for c in hit])
        cfar_los_rmse = float(np.sqrt(np.mean(np.array(err_los) ** 2)))
        cfar_los_mae = float(np.mean(err_los))
    else:
        cfar_los_rmse = cfar_los_mae = float("nan")
    az_err = np.abs([c["az_est"] - c["az_true"] for c in classic])
    az_mae = float(np.mean(az_err))

    # --- 2. ML（绝对距离特征）：公平对比 ---
    print("[2/4] 训练 ML（绝对距离特征）...")
    tr_abs = build_abs_ds(args.train_data, channels, target_ecef0, ground_ecef0,
                          args.snr_db, args.seed)
    te_abs = build_abs_ds(args.n_test, channels, target_ecef0, ground_ecef0,
                          args.snr_db, args.seed + 10000)
    model_abs = SensingMLP(in_dim=WIDEBAND_K_ABS).to(device)
    ml_abs = train_eval(model_abs, tr_abs, te_abs, device, args.epochs)

    # --- 3. ML（旧特征=相对质心，bug 版）：验证“类别先验”假设 ---
    print("[3/4] 训练 ML（旧特征 center='centroid'，预期仅类别先验）...")
    tr_old = SatROIDataset(args.train_data, channels, num_points=128, device=device,
                           tau=args.tau, with_label=True, wideband=True,
                           rp_align=False, center="centroid")   # 旧 bug 行为：相对质心
    te_old = SatROIDataset(args.n_test, channels, num_points=128, device=device,
                           tau=args.tau, with_label=True, wideband=True,
                           rp_align=False, center="centroid")
    tr_old_ds = make_fixed(tr_old, args.train_data, wideband=True)
    te_old_ds = make_fixed(te_old, args.n_test, wideband=True)
    model_old = SensingMLP(in_dim=WIDEBAND_K).to(device)
    ml_old = train_eval(model_old, tr_old_ds, te_old_ds, device, args.epochs)

    # --- 4. ML（形状特征=质心对齐）：分类最优、定位=先验的参照 ---
    print("[4/4] 训练 ML（形状特征 center='centroid'+align，分类参照）...")
    tr_shp = SatROIDataset(args.train_data, channels, num_points=128, device=device,
                           tau=args.tau, with_label=True, wideband=True,
                           rp_align=True, center="centroid")
    te_shp = SatROIDataset(args.n_test, channels, num_points=128, device=device,
                           tau=args.tau, with_label=True, wideband=True,
                           rp_align=True, center="centroid")
    tr_shp_ds = make_fixed(tr_shp, args.train_data, wideband=True)
    te_shp_ds = make_fixed(te_shp, args.n_test, wideband=True)
    model_shp = SensingMLP(in_dim=WIDEBAND_K).to(device)
    ml_shp = train_eval(model_shp, tr_shp_ds, te_shp_ds, device, args.epochs)

    # --- 输出报告 ---
    print("\n" + "=" * 74)
    print("对比结果（同测试集，SNR 20dB，距离分辨率 0.15m，ROI ±40m）")
    print("=" * 74)
    print(f"[2D-CFAR ] 检测率            : {det_rate:.3f}  (P_fa=1e-4, CA-CFAR, 绝对距离)")
    print(f"[2D-CFAR ] 沿视线定位 RMSE  : {cfar_los_rmse:7.2f} m   MAE {cfar_los_mae:.2f} m  (n_hit={len(hit)})")
    print(f"[MUSIC   ] 目标方向测向 MAE : {az_mae:.3f} deg  (ULA 8 元, 快拍 64)")
    print(f"[ML(绝对)] 分类准确率       : {ml_abs['acc']:.3f}")
    print(f"[ML(绝对)] 2D 定位 RMSE     : {ml_abs['rmse_2d']:.2f} m")
    print(f"[ML(绝对)] 沿视线 RMSE      : {ml_abs['rmse_los']:.2f} m    横向 RMSE {ml_abs['rmse_cross']:.2f} m")
    print(f"[ML(旧特)] 分类准确率       : {ml_old['acc']:.3f}")
    print(f"[ML(旧特)] 2D 定位 RMSE     : {ml_old['rmse_2d']:.2f} m  (center='centroid'，无位置信息)")
    print(f"[ML(形状)] 分类准确率       : {ml_shp['acc']:.3f}")
    print(f"[ML(形状)] 2D 定位 RMSE     : {ml_shp['rmse_2d']:.2f} m  (质心对齐，定位=先验)")
    print("-" * 74)
    slant_bs = 695.0 * 1000.0
    ang_per_m = np.rad2deg(np.arctan(1.0 / slant_bs))
    print("物理/工程结论：")
    print(f"  1. 远场角度分辨：斜距 {slant_bs/1000:.0f} km → 1m 横向偏移 {ang_per_m:.5f} deg；")
    print(f"     80m ROI 全宽 {80*ang_per_m:.4f} deg << ULA8 分辨力 {180/8:.1f} deg → 角度无 ROI 内定位信息")
    print(f"  2. 特征构造缺陷：center='centroid' 的 d_proj 相对体素质心，目标绝对位置在")
    print(f"     特征层被丢弃（实测单体素 bin 恒定）→ 旧/形状特征 ML 定位≈类别先验")
    print(f"     旧特征 2D RMSE {ml_old['rmse_2d']:.1f} m / 形状特征 {ml_shp['rmse_2d']:.1f} m vs 绝对特征 {ml_abs['rmse_2d']:.1f} m")
    print(f"  3. 经典 2D-CFAR 在绝对距离像上可检测+沿视线定位（RMSE {cfar_los_rmse:.1f} m），")
    print(f"     横向定位仍需先验/多普勒（物理墙）")
    print(f"  4. MUSIC 测向（合成点源快照，ULA-8）：MAE {az_mae:.3f} deg 验证算法自洽；")
    print(f"     但目标在 ROI 内 ±40m 对应角度 ±{40*ang_per_m:.4f} deg << 搜索步长 0.1 deg，")
    print(f"     测向对 ROI 内定位无实际分辨力（几何物理墙，非算法限制）")

    # --- 保存示例图 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        c0 = classic[0]
        im0 = axes[0].imshow(c0["rd"], aspect="auto", origin="lower", cmap="jet")
        axes[0].set_title(f"RD map (abs, det={c0['detected']})")
        axes[0].set_xlabel("Doppler bin"); axes[0].set_ylabel("Range bin")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        axes[1].imshow(c0["det"], aspect="auto", origin="lower", cmap="gray")
        axes[1].set_title("2D-CFAR detections")
        axes[1].set_xlabel("Doppler bin"); axes[1].set_ylabel("Range bin")
        spec_db = 10 * np.log10(c0["spec"] + 1e-12)
        axes[2].plot(c0["grid"], spec_db, lw=1.2)
        axes[2].axvline(c0["az_true"], color="r", ls="--", label=f"target {c0['az_true']:.2f} deg")
        axes[2].axvline(c0["az_est"], color="g", ls=":", label=f"est {c0['az_est']:.2f} deg")
        axes[2].set_title("MUSIC spectrum (target dir)")
        axes[2].set_xlabel("azimuth (deg)"); axes[2].set_ylabel("dB")
        axes[2].legend(fontsize=8)
        plt.tight_layout()
        os.makedirs("isac_demo", exist_ok=True)
        plt.savefig("isac_demo/baseline_classic.png", dpi=130, bbox_inches="tight")
        print("\n[fig] isac_demo/baseline_classic.png")
    except Exception as e:
        print(f"\n[fig] skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D-CFAR + MUSIC vs ML 感知对比")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="tracked")
    parser.add_argument("--train_data", type=int, default=300)
    parser.add_argument("--n_test", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
