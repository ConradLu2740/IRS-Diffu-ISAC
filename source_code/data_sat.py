"""
data_sat.py — 星-地 ISAC 动态数据生成

在 setup_sat.py 的轨道/信道物理层之上：
  - 预计算每帧的 5 路径动态信道（BS=卫星 → ROI=地面目标 → UE=地面站，可选星载/地面 RIS）
  - 生成 (点云, 动态条件) 样本对，条件包含 幅值/相位/IRS相位 + 动态特征（多普勒/时延/距离/仰角）
  - 支持 irs_mode: 'none'（无RIS） / 'sat'（星载RIS） / 'ground'（地面RIS）

复用：data.py 的目标模板与点云提取、data_progress_amp_phase 特征提取。
依赖：setup_sat.py
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset

import setup
import setup_sat as ss
from data import generate_ROI, extract_point_cloud_from_voxel, data_progress_amp_phase
from phase_optimizer_sat import PhaseOptimizerSat

# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------
IRS_ELEMENTS = 16              # 单块 RIS 面板（4×4）
IRS_GAIN = 100.0               # RIS 每元素增益（线性，~20dB/元素）；大孔径面板的孔径增益补偿路径损耗
P_SNR = 20                     # 参考 SNR dB（与 setup.py 一致）
POWER_SIGMA = 0.01             # 噪声功率
ROI_RES = 16                   # ROI 体素分辨率（16³=4096，与 setup.ROI_Length 对齐）
VOXEL_SIZE_M = 5.0             # 体素边长（米）→ 目标区域 80m×80m×80m
BS_ANT = 4                     # 卫星（BS）天线数
UE_ANT = 4                     # 地面站（UE）天线数

# 导频星座（16QAM，与 data.py 一致）
_SIGNAL1 = np.array([
    -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
    -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
     3 - 3j,  3 - 1j,  3 + 3j,  3 + 1j,
     1 - 3j,  1 - 1j,  1 + 3j,  1 + 1j,
], dtype=np.complex64)


def make_roi_local(res=ROI_RES, voxel_size=VOXEL_SIZE_M):
    """ROI 体素中心局部坐标（米），原点在目标区域中心。返回 [res³, 3]"""
    half = res * voxel_size / 2.0
    centers = (np.arange(res) + 0.5) * voxel_size - half
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing="ij")
    return np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T.astype(np.float64)


def _ant_positions(center_km, n, offset_m=0.5):
    """围绕中心点生成 n 个天线位置（ECEF km）。"""
    off = offset_m / 1000.0
    offs = np.zeros((n, 3))
    for i in range(n):
        offs[i, 1] = (i % 2 - 0.5) * 2 * off
        offs[i, 2] = (i // 2 - 0.5) * 2 * off
    return np.asarray(center_km, dtype=float) + offs


def _irs_panel_positions(center_km, n=IRS_ELEMENTS, panel_m=10.0, axis=(0, 1, 0)):
    """RIS 面板元素位置（ECEF km）。星载面板可 10m 量级，地面面板 1m 量级。"""
    side = int(math.isqrt(n))
    off = panel_m / 1000.0
    pos = []
    for i in range(side):
        for j in range(side):
            d = np.zeros(3, dtype=float)
            d[axis[0]] = (i - (side - 1) / 2.0) * off
            d[axis[1]] = (j - (side - 1) / 2.0) * off
            pos.append(np.asarray(center_km, dtype=float) + d)
    return np.array(pos)


def get_channel_mat(a_pos, b_pos, wavelength_m, b_gain=1.0, a_gain=1.0):
    """自由空间信道矩阵：H[i,j] = sqrt(0.1) exp(j2πd/λ)/d，a→b 方向。

    a_pos: [Na,3], b_pos: [Nb,3] (km) → H: [Na, Nb]
    b_gain / a_gain: 端点增益（线性），乘 sqrt(gain)（等效接收/发射端天线增益）
    """
    d = np.linalg.norm(a_pos[:, None, :] - b_pos[None, :, :], axis=-1)  # km
    d_m = d * 1000.0
    H = np.sqrt(0.1) * np.exp(1j * 2.0 * np.pi * d_m / wavelength_m) / np.maximum(d_m, 1e-6)
    H = H * np.sqrt(b_gain) * np.sqrt(a_gain)
    return H.astype(np.complex64)


class SatScenarioChannels:
    """每帧预计算 5 路径信道矩阵（结构与 setup.py 的 H_dict 对齐）。"""

    def __init__(self, frames, irs_mode="sat", roi_res=ROI_RES,
                 voxel_size=VOXEL_SIZE_M, device="cpu"):
        self.frames = frames
        self.irs_mode = irs_mode
        self.device = device
        self.roi_res = roi_res
        self.voxel_size = voxel_size
        self.wavelength_m = ss.C_LIGHT_KM * 1000.0 / ss.FC_HZ
        self.roi_local = make_roi_local(roi_res, voxel_size)  # [R,3] 米

        self.channels_per_frame = [self._precompute_frame(f) for f in frames]
        # 信道矩阵统一转为 torch tensor（complex64）
        for ht in self.channels_per_frame:
            for k in list(ht.keys()):
                if k.startswith("H_"):
                    ht[k] = torch.from_numpy(ht[k]).to(self.device)
        self.tensor_a = self._calibrate()

    # ------------------------------------------------------------------
    def _frame_geometry(self, frame):
        """从帧数据构建本帧几何（ECEF km）。"""
        target_center = frame["target_pos"]
        roi_ecef = target_center[None, :] + self.roi_local / 1000.0   # [R,3]

        bs_ecef = _ant_positions(frame["sat_pos"], BS_ANT)
        ue_ecef = _ant_positions(frame["ground_pos"], UE_ANT)
        irs_ecef = None
        if self.irs_mode == "sat":
            irs_ecef = _irs_panel_positions(frame["sat_ris_pos"], IRS_ELEMENTS, panel_m=10.0)
        elif self.irs_mode == "ground":
            irs_ecef = _irs_panel_positions(frame["ground_ris_pos"], IRS_ELEMENTS, panel_m=1.0)
        return bs_ecef, ue_ecef, roi_ecef, irs_ecef

    def _precompute_frame(self, frame):
        wl = self.wavelength_m
        bs, ue, roi, irs = self._frame_geometry(frame)
        H = {}
        H["H_BS_ROI"] = get_channel_mat(bs, roi, wl)      # [4, R]
        H["H_ROI_UE"] = get_channel_mat(roi, ue, wl)      # [R, 4]
        if irs is not None:
            H["H_BS_IRS"] = get_channel_mat(bs, irs, wl, b_gain=IRS_GAIN)  # [4, N]
            H["H_ROI_IRS"] = get_channel_mat(roi, irs, wl, b_gain=IRS_GAIN)  # [R, N]
            H["H_IRS_ROI"] = get_channel_mat(irs, roi, wl, a_gain=IRS_GAIN)  # [N, R]
            H["H_IRS_UE"] = get_channel_mat(irs, ue, wl, a_gain=IRS_GAIN)    # [N, 4]
        H["f_d_bs_roi"] = frame["f_d_hz"]
        # BS→IRS 段多普勒（若 IRS 存在且随卫星运动：星载IRS运动近似等于卫星）
        if irs is not None and self.irs_mode == "sat":
            u = frame["sat_ris_pos"] - frame["sat_pos"]
            u = u / (np.linalg.norm(u) + 1e-12)
            v_rel = np.dot(frame["sat_vel"], u) * 1000.0
            H["f_d_bs_irs"] = v_rel / self.wavelength_m
        else:
            H["f_d_bs_irs"] = 0.0
        H["d_sat_target"] = frame["dist_sat_target"]
        H["d_target_ground"] = frame["dist_target_ground"]
        H["elevation_deg"] = frame["elevation_deg"]
        H["delay_s"] = frame["delay_s"]
        return H

    # ------------------------------------------------------------------
    def _calibrate(self):
        """以中心帧参考链路的自由空间损耗做功率标定（使 SNR≈P_SNR）。"""
        mid = self.channels_per_frame[len(self.channels_per_frame) // 2]
        d1 = mid["d_sat_target"] * 1000.0   # m
        d2 = mid["d_target_ground"] * 1000.0
        # 参考：单路径幅度 |H_ref| = sqrt(0.1)^2 / (d1*d2)（两项各乘一次 sqrt0.1）
        H_ref2 = 0.1 / (d1 * d1 * d2 * d2)
        Power = 10 ** (P_SNR / 10.0) * POWER_SIGMA * 64
        a = float(np.sqrt(Power / (H_ref2 * 10.0)))
        return a

    # ------------------------------------------------------------------
    def to_device(self, d):
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    def frame_cond_dim(self):
        """每帧条件维度：12(X) + 12(Y) + 32(IRS) + 4(动态) 或去掉 IRS 项。"""
        irs_part = 2 * IRS_ELEMENTS if self.irs_mode != "none" else 0
        return 12 + 12 + irs_part + 4


# ----------------------------------------------------------------------
# 信号计算
# ----------------------------------------------------------------------

def calculate_value_sat(ROI_voxel, phase, X, Ht, Power_sigma, t_rel, wavelength_m):
    """单帧接收信号（5 路径结构 + 多普勒注入）。

    ROI_voxel: [R] 体素占据（0/1）
    phase:     [N] 或 []（无IRS）RIS 相位
    X:         [4,1] 发射导频
    Ht:        该帧信道 dict
    t_rel:     相对帧0时间（秒），多普勒相位累积用
    """
    S = ROI_voxel.reshape(-1)
    S_f = S.to(torch.float32)
    S_c = torch.complex(S_f, torch.zeros_like(S_f))
    N = X.shape[0]

    H_BS_ROI = Ht["H_BS_ROI"]; H_ROI_UE = Ht["H_ROI_UE"]
    f_d1 = Ht["f_d_bs_roi"]
    dop1 = torch.exp(torch.tensor(1j * 2 * math.pi * f_d1 * t_rel, dtype=torch.complex64))

    Bmat = S_c[:, None] * H_ROI_UE                       # [R, 4]
    H_BS_ROI_UE = H_BS_ROI.matmul(Bmat) * dop1           # [4, 4]
    H_total = H_BS_ROI_UE                               # [4, 4]

    if "H_ROI_IRS" in Ht:
        v = torch.exp(1j * torch.tensor(phase, dtype=torch.float32)).to(torch.complex64)  # [N]
        f_d2 = Ht["f_d_bs_irs"]
        dop2 = torch.exp(torch.tensor(1j * 2 * math.pi * f_d2 * t_rel, dtype=torch.complex64))

        C = H_BS_ROI * S_c[None, :]                      # [4, R]
        C1 = C.matmul(Ht["H_ROI_IRS"])                   # [4, N]
        H_BS_ROI_IRS_UE = ((C1 * v[None, :]).matmul(Ht["H_IRS_UE"])) * dop2

        IR1 = Ht["H_IRS_ROI"] * S_c[None, :]             # [N, R]
        B1 = IR1.matmul(H_ROI_UE)                        # [N, 4]
        H_BS_IRS_ROI_UE = ((Ht["H_BS_IRS"] * v[None, :]).matmul(B1)) * dop2

        H_total = H_total + H_BS_ROI_IRS_UE + H_BS_IRS_ROI_UE

    receive = H_total.matmul(X)                          # [4, 1]
    std = math.sqrt(Power_sigma)
    noise = torch.complex(torch.randn_like(receive.real) * std, torch.randn_like(receive.imag) * std)
    return receive + noise


# ----------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------

class SatROIDataset(Dataset):
    """星-地 ISAC 样本集：每个样本 = 地面目标点云 + 动态条件 [Tau, cond_dim]。

    phase_mode:
      'random'  - 每帧随机 IRS 相位（基线）
      'tracked' - 每帧解析优化相位（动态跟踪，Phase 2）
    """

    def __init__(self, n_samples, channels, num_points=2048, device="cpu",
                 tau=ss.TAU, p_snr=P_SNR, power_sigma=POWER_SIGMA,
                 phase_mode="random"):
        self.n = n_samples
        self.ch = channels
        self.device = device
        self.num_points = num_points
        self.tau = tau
        self.p_snr = p_snr
        self.power_sigma = power_sigma
        self.a = channels.tensor_a
        self.phase_mode = phase_mode
        if phase_mode == "tracked" and channels.irs_mode == "none":
            self.phase_mode = "random"  # 无 IRS 时退回随机
        self._opt = PhaseOptimizerSat(channels, device=device) if self.phase_mode == "tracked" else None

        # 导频 X（16QAM 前4符号 × 标定系数）
        pilot = torch.tensor(_SIGNAL1[:4], dtype=torch.complex64).view(4, 1)
        self.X_fixed = (self.a * pilot).to(device)

        # 每帧条件特征
        self.cond_dim = channels.frame_cond_dim()

    def __len__(self):
        return self.n

    def _frame_phases(self, ROI_voxel):
        """返回 [Tau] 相位序列（每帧 [N] 或空）。"""
        seq = []
        for t, ht in enumerate(self.ch.channels_per_frame):
            if "H_ROI_IRS" not in ht:
                seq.append(torch.tensor([]))
                continue
            if self.phase_mode == "tracked":
                ph = self._opt.optimize_frame(ht, ROI_voxel, self.X_fixed)
            else:
                N = ht["H_ROI_IRS"].shape[1]
                ph = torch.rand(N).float() * 2 * np.pi
            seq.append(ph)
        return seq

    def __getitem__(self, idx):
        ROI_np = generate_ROI().astype("float32")
        # 点云（地面目标区域，物理坐标 → [-1,1] 归一化）
        point_cloud = extract_point_cloud_from_voxel(
            ROI_np, num_points=self.num_points, voxel_size=self.ch.voxel_size)
        max_extent = self.ch.roi_res * self.ch.voxel_size
        point_cloud = (point_cloud / max_extent) * 2.0 - 1.0
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        ROI_voxel = torch.tensor(ROI_np).reshape(-1)
        # X 特征（12 维）
        X_cpu = self.X_fixed.detach().cpu()
        X_amp = torch.abs(X_cpu).reshape(-1)
        X_phase = torch.angle(X_cpu).reshape(-1)
        X_feat = torch.cat([X_amp, torch.sin(X_phase), torch.cos(X_phase)], dim=0).float()

        cond_list = []
        phases_seq = self._frame_phases(ROI_voxel)
        for t in range(self.tau):
            frame = self.ch.channels_per_frame[t]
            t_rel = self.ch.frames[t]["t_sec"]
            phases = phases_seq[t]

            Y_t = calculate_value_sat(
                ROI_voxel, phases, self.X_fixed, frame, self.power_sigma,
                t_rel, self.ch.wavelength_m)
            Y_feat = data_progress_amp_phase(Y_t.detach().cpu())     # 12 维

            if len(phases) > 0:
                IRS_feat = torch.cat([torch.sin(phases), torch.cos(phases)], dim=0).float()  # 2N
            else:
                IRS_feat = torch.tensor([])

            # 动态特征：多普勒(kHz) / 时延(ms) / 距离(百km) / 仰角(归一化)
            dyn = torch.tensor([
                frame["f_d_bs_roi"] / 1e3,          # kHz（±700）
                frame["delay_s"] * 1e3,             # ms（~2.5）
                frame["d_sat_target"] / 1e2,        # 百 km（~7）
                frame["elevation_deg"] / 90.0,      # [0,1]
            ], dtype=torch.float32)

            cond_t = torch.cat([X_feat, Y_feat, IRS_feat, dyn], dim=0).float()
            cond_list.append(cond_t)

        cond = torch.stack(cond_list).float()        # [Tau, cond_dim]
        return point_cloud.float(), cond


# ----------------------------------------------------------------------
# 便捷入口
# ----------------------------------------------------------------------

def build_sat_dataset(n_samples=16, irs_mode="sat", num_points=512,
                      tau=ss.TAU, device="cpu", seed=42):
    """构建星-地 ISAC 数据集（smoke 用），返回 (dataset, scenario, channels)。"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    scenario = ss.SatISACScenario(tau=tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=irs_mode, device=device)
    dataset = SatROIDataset(n_samples=n_samples, channels=channels,
                            num_points=num_points, device=device, tau=tau)
    return dataset, scenario, channels


if __name__ == "__main__":
    for mode in ["none", "sat", "ground"]:
        ds, _, ch = build_sat_dataset(n_samples=2, irs_mode=mode, num_points=256)
        pc, cond = ds[0]
        print(f"[{mode:>6}] point_cloud={tuple(pc.shape)} cond={tuple(cond.shape)} "
              f"cond_dim={ch.frame_cond_dim()} a={ch.tensor_a:.3e}")
