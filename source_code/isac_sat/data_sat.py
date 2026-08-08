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
import sys, os
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)
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
                 voxel_size=VOXEL_SIZE_M, device="cpu", bs_ant=BS_ANT, ue_ant=UE_ANT):
        self.frames = frames
        self.irs_mode = irs_mode
        self.device = device
        self.roi_res = roi_res
        self.voxel_size = voxel_size
        self.bs_ant = bs_ant
        self.ue_ant = ue_ant
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

        bs_ecef = _ant_positions(frame["sat_pos"], self.bs_ant)
        ue_ecef = _ant_positions(frame["ground_pos"], self.ue_ant)
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
        """每帧条件维度：3·BS_ANT(X) + 3·UE_ANT(Y) + 2·IRS + 4(动态) + 1(RCS)。"""
        irs_part = 2 * IRS_ELEMENTS if self.irs_mode != "none" else 0
        return 3 * self.bs_ant + 3 * self.ue_ant + irs_part + 4 + 1


# ----------------------------------------------------------------------
# 地面目标模板（星-地 ISAC 感知目标）
# 在 8×8×8 体素内定义，放置到 16³ ROI 空间
# ----------------------------------------------------------------------


def _template_car():
    """车辆：车身 + 4 轮"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[2:6, 1:7, 2:4] = 1      # 车身
    obj[1:3, 1:3, 0:2] = 1      # 前左轮
    obj[1:3, 5:7, 0:2] = 1      # 前右轮
    obj[5:7, 1:3, 0:2] = 1      # 后左轮
    obj[5:7, 5:7, 0:2] = 1      # 后右轮
    return obj


def _template_uav():
    """无人机：中心体 + 十字臂 + 4 旋翼"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[3:5, 3:5, 4:6] = 1      # 中心体
    obj[2:6, 3:5, 5:6] = 1      # 横臂
    obj[3:5, 2:6, 5:6] = 1      # 纵臂
    obj[1:3, 1:3, 6:7] = 1      # 旋翼 1
    obj[1:3, 5:7, 6:7] = 1      # 旋翼 2
    obj[5:7, 1:3, 6:7] = 1      # 旋翼 3
    obj[5:7, 5:7, 6:7] = 1      # 旋翼 4
    return obj


def _template_building():
    """建筑：底座 + 楼身"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[0:8, 0:8, 0:2] = 1      # 底座
    obj[2:6, 2:6, 2:8] = 1      # 楼身
    return obj


def _template_tank():
    """储油罐：圆柱体（近似）"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    cx, cy, r = 3.5, 3.5, 2.5
    for x in range(8):
        for y in range(8):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                obj[x, y, 0:5] = 1
    return obj


def _template_tower():
    """天线塔：细高塔身 + 横杆"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[3:5, 3:5, 0:8] = 1      # 塔身
    obj[0:8, 3:4, 3:4] = 1      # 横杆 1
    obj[0:8, 4:5, 5:6] = 1      # 横杆 2
    return obj


def _template_cubesat():
    """立方星：卫星体 + 太阳能板"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[3:5, 3:5, 4:6] = 1      # 卫星体
    obj[1:3, 3:5, 4:5] = 1      # 左板
    obj[5:7, 3:5, 4:5] = 1      # 右板
    obj[3:5, 1:3, 4:5] = 1      # 前板
    obj[3:5, 5:7, 4:5] = 1      # 后板
    return obj


def _template_bicycle():
    """自行车：两轮 + 车架 + 车把"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[1:2, 1:3, 0:2] = 1      # 前轮
    obj[6:7, 1:3, 0:2] = 1      # 后轮
    obj[2:6, 1:2, 0:1] = 1      # 车架下梁
    obj[2:6, 2:3, 1:2] = 1      # 车架上梁
    obj[1:3, 3:4, 1:3] = 1      # 前叉/车把
    return obj


def _template_pedestrian():
    """行人：头 + 躯干 + 双腿"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[3:5, 3:5, 6:8] = 1      # 头
    obj[2:6, 2:6, 4:6] = 1      # 躯干
    obj[2:3, 2:3, 2:4] = 1      # 左腿
    obj[2:3, 5:6, 2:4] = 1      # 右腿
    return obj


def _template_train():
    """火车（车厢）：长条车身 + 车顶"""
    obj = np.zeros((8, 8, 8), dtype=np.uint8)
    obj[0:8, 0:8, 2:5] = 1      # 车身（拉长）
    obj[0:8, 2:6, 5:6] = 1      # 车顶
    obj[0:8, 0:1, 0:2] = 1      # 轮子带
    obj[0:8, 7:8, 0:2] = 1
    return obj


GROUND_TARGET_TEMPLATES = [
    ("car", _template_car),
    ("uav", _template_uav),
    ("bicycle", _template_bicycle),
    ("pedestrian", _template_pedestrian),
    ("train", _template_train),
]

# 移动目标类别（供多目标追踪场景使用）
MOBILE_CLASSES = ["car", "uav", "bicycle", "pedestrian", "train"]

# 微多普勒特征（雷达目标识别 RATR 物理基础）：运动部件对回波的频率调制
# 无人机旋翼高速旋转 → 大扩展；车辆移动 → 中；静止目标 → 0
MICRO_DOPPLER_HZ = {
    "car": 800.0,
    "uav": 3000.0,
    "building": 0.0,
    "tank": 0.0,
    "tower": 0.0,
    "cubesat": 0.0,
}

# ----------------------------------------------------------------------
# 宽带距离像（HRRP 物理基础）
# ----------------------------------------------------------------------
WIDEBAND_K = 512           # 子载波数（距离单元）；512@1GHz → 距离窗 ~154m 覆盖 80m ROI
WIDEBAND_BW_HZ = 1e9       # 带宽 1GHz → 距离分辨率 ~0.15m

# ISAR 参数（目标转动 → 距离-时间序列，合成孔径积累）
# 物理规律：总转角 = ω·dt·(M-1) 决定姿态可估性（转角越大横向分辨率越高）
ISAR_M = 32                # 观测帧数
ISAR_OMEGA_DEG = 60.0      # 目标自转角速度（°/s）
ISAR_DT = 0.03             # 帧间隔（s）→ 总转角 ≈ 56°


def compute_isar_sequence(ROI_np, target_ecef, ground_ecef, wavelength_m,
                          m_frames=ISAR_M, omega_deg=ISAR_OMEGA_DEG, dt=ISAR_DT,
                          k=WIDEBAND_K, bw_hz=WIDEBAND_BW_HZ, snr_db=20.0, seed=0):
    """ISAR 距离-时间序列：目标自转 ω，M 帧宽带距离像。

    物理：目标自转（如卫星自旋/无人机盘旋）产生不同视角的距离像，
    转角越大横向分辨率越高（ISAR 合成孔径），姿态可估性越强。
    返回 [M, K]（每帧距离像，L2 归一化）。
    """
    from scipy.ndimage import rotate as ndi_rotate
    C_MS = ss.C_LIGHT_KM * 1000.0
    local = make_roi_local()
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
        rel = p - p.mean(axis=0)
        d_proj = (rel @ u) * 1000.0                     # 米
        tau = (d_proj + 50e3 + 695e3) / C_MS
        H = np.exp(-2j * np.pi * f[:, None] * tau[None, :]).sum(axis=1)
        if snr_db is not None:
            sig_pow = np.mean(np.abs(H) ** 2)
            n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
            H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
        rp = np.abs(np.fft.ifft(H))
        norm = np.linalg.norm(rp)
        seq.append((rp / (norm + 1e-12)).astype(np.float32))
    return np.array(seq)


def compute_range_profile(ROI_np, target_ecef, ground_ecef, wavelength_m,
                          k=WIDEBAND_K, bw_hz=WIDEBAND_BW_HZ, snr_db=20.0,
                          seed=0, bs_dist_km=695.0, align=True):
    """宽带距离像：目标体素沿观测视线的散射分布（HRRP）。

    align=True：质心对齐（提取形状/姿态信息，位置无关）
    align=False：保留原始距离像（包含目标位置信息，用于定位）
    返回 [k] 距离像幅度（归一化）。
    """
    C_MS = ss.C_LIGHT_KM * 1000.0
    local = make_roi_local()                       # [4096, 3] 米
    occ = np.argwhere(ROI_np > 0.5)
    if len(occ) == 0:
        return np.zeros(k, dtype=np.float32)
    # 体素 ECEF 位置（km）
    p = target_ecef[None, :] + local[occ[:, 0] * 256 + occ[:, 1] * 16 + occ[:, 2], :] / 1000.0
    # 三维质心对齐：提取目标形状沿视线的分布（与位置无关）
    p_center = p.mean(axis=0)
    rel = p - p_center                              # km
    u = ground_ecef - target_ecef
    u = u / (np.linalg.norm(u) + 1e-12)             # 视线方向单位矢量
    d_proj = (rel @ u) * 1000.0                     # 沿视线投影（米，相对质心）
    # 总时延（常数偏移对距离像形状无影响，只平移——质心对齐后无影响）
    d_bs = bs_dist_km * 1000.0 + d_proj * 0.1       # BS 端差异小（远场近似）
    d_ue = 50e3 + d_proj                            # 体素到 UE 差异 ≈ 视线投影
    tau = (d_bs + d_ue) / C_MS                        # 秒

    f = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, k)     # 基带子载波
    H = np.exp(-2j * np.pi * f[:, None] * tau[None, :]).sum(axis=1)  # [k]
    if snr_db is not None:
        rng = np.random.RandomState(seed)
        sig_pow = np.mean(np.abs(H) ** 2)
        n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        H = H + rng.randn(k) * np.sqrt(n_pow / 2) + 1j * rng.randn(k) * np.sqrt(n_pow / 2)
    rp = np.abs(np.fft.ifft(H))
    if align:
        # 质心对齐（HRRP 经典处理）：去掉目标位置影响，提取形状/姿态信息
        idx = np.arange(k)
        total = np.sum(rp)
        if total > 1e-12:
            centroid = int(np.clip(np.sum(rp * idx) / total, 0, k - 1))
            rp = np.roll(rp, k // 2 - centroid)
    # 归一化（L2）
    norm = np.linalg.norm(rp)
    return (rp / (norm + 1e-12)).astype(np.float32)


def generate_ground_target_sample(pose_angle_deg=None):
    """生成含 1 个随机地面目标的 ROI（16³ 体素），带类别与姿态标签。

    物理：目标绕竖直轴（z）随机旋转 → 体素分布/散射中心改变 →
    接收信号的幅度/相位模式随姿态变化（雷达目标姿态敏感性）。
    返回 (ROI float32, class_id int, angle_deg float)
    """
    import random as _random
    from scipy.ndimage import rotate as ndi_rotate

    name, maker = _random.choice(GROUND_TARGET_TEMPLATES)
    class_id = [n for n, _ in GROUND_TARGET_TEMPLATES].index(name)
    angle = pose_angle_deg if pose_angle_deg is not None else _random.uniform(0.0, 360.0)

    space = np.zeros((setup.ROI_Length, setup.ROI_Length, setup.ROI_Length), dtype=np.float32)
    obj = maker().astype(np.float32)
    placed = False
    attempts = 0
    while not placed and attempts < 100:
        x = _random.randint(0, setup.ROI_Length - 8)
        y = _random.randint(0, setup.ROI_Length - 8)
        z = _random.randint(0, setup.ROI_Length - 8)
        if np.all(space[x:x + 8, y:y + 8, z:z + 8] == 0):
            space[x:x + 8, y:y + 8, z:z + 8] = obj
            placed = True
        attempts += 1

    # 绕 z 轴（竖直轴）旋转整个 ROI：姿态改变散射体分布
    if abs(angle) > 1e-6:
        space = ndi_rotate(space, angle, axes=(0, 1), reshape=False,
                           order=1, mode="constant", cval=0.0)
        space = (space > 0.5).astype(np.float32)
    return space, class_id, float(angle)


def generate_ground_roi():
    """兼容旧接口：只返回 ROI（默认随机姿态）。"""
    roi, _, _ = generate_ground_target_sample()
    return roi


def generate_multi_target_sample(n_max=2, max_tries=40):
    """生成含 1-n_max 个地面目标的 ROI（16³ 体素），返回 (ROI, targets)。

    targets: [(class_id, (cx, cy)), ...]，cx/cy 为归一化质心 [-1,1]。
    工程用途：多目标感知（分类+定位每个目标）。
    """
    import random as _random
    n_obj = _random.randint(1, n_max)
    space = np.zeros((setup.ROI_Length, setup.ROI_Length, setup.ROI_Length), dtype=np.float32)
    targets = []
    for _ in range(n_obj):
        for _ in range(max_tries):
            roi, cid, _ = generate_ground_target_sample()
            if not np.any((space > 0.5) & (roi > 0.5)):
                space = np.maximum(space, roi)
                occ = np.argwhere(roi > 0.5)
                cx = occ[:, 0].mean() / setup.ROI_Length * 2.0 - 1.0
                cy = occ[:, 1].mean() / setup.ROI_Length * 2.0 - 1.0
                targets.append((cid, (float(cx), float(cy))))
                break
    return space, targets


# ----------------------------------------------------------------------
def data_progress_amp_phase_db(data_complex: torch.Tensor) -> torch.Tensor:
    """复数信号 → [幅值(dB), sin(相位), cos(相位)] 特征。

    dB 尺度是通信工程标准（幅度/功率用对数表示），避免星-地场景
    功率标定系数（~1e11）造成特征尺度爆炸。
    """
    amp = torch.abs(data_complex).reshape(-1)
    phase = torch.angle(data_complex).reshape(-1)
    return torch.cat([20.0 * torch.log10(amp + 1e-12),
                      torch.sin(phase), torch.cos(phase)], dim=0).float()


# 信号计算
# ----------------------------------------------------------------------

def calculate_value_sat(ROI_voxel, phase, X, Ht, Power_sigma, t_rel, wavelength_m,
                        micro_fd_hz=0.0):
    """单帧接收信号（5 路径结构 + 多普勒注入 + 可选微多普勒）。

    ROI_voxel: [R] 体素占据（0/1）
    phase:     [N] 或 []（无IRS）RIS 相位
    X:         [4,1] 发射导频
    Ht:        该帧信道 dict
    t_rel:     相对帧0时间（秒），多普勒相位累积用
    micro_fd_hz: 目标微动多普勒（RATR 微多普勒特征），0 表示静止目标
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
    # 目标微动对回波的整体频率调制（微多普勒，物理：运动部件旋转/振动）
    if micro_fd_hz != 0.0:
        receive = receive * torch.exp(torch.tensor(
            1j * 2 * math.pi * micro_fd_hz * t_rel, dtype=torch.complex64))
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
                 phase_mode="random", target_source="ground", with_label=False,
                 wideband=False, wideband_snr_db=20.0, isar=False, rp_align=True,
                 multi=False):
        self.n = n_samples
        self.ch = channels
        self.device = device
        self.num_points = num_points
        self.tau = tau
        self.p_snr = p_snr
        self.power_sigma = power_sigma
        self.a = channels.tensor_a
        self.phase_mode = phase_mode
        self.target_source = target_source
        self.with_label = with_label
        self.wideband = wideband or isar   # ISAR 隐含宽带
        self.wideband_snr_db = wideband_snr_db
        self.isar = isar
        self.rp_align = rp_align
        self.multi = multi
        if phase_mode == "tracked" and channels.irs_mode == "none":
            self.phase_mode = "random"  # 无 IRS 时退回随机
        self._opt = PhaseOptimizerSat(channels, device=device) if self.phase_mode == "tracked" else None
        # 宽带距离像几何（取中帧）
        if self.wideband:
            mid = channels.frames[len(channels.frames) // 2]
            self._target_ecef = mid["target_pos"]
            self._ground_ecef = mid["ground_pos"]

        # 导频 X（16QAM 前 BS_ANT 个符号 × 标定系数）
        n_bs = channels.bs_ant
        pilot = torch.tensor(_SIGNAL1[:n_bs], dtype=torch.complex64).view(n_bs, 1)
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

    def _make_roi(self):
        """生成 ROI 体素，返回 (ROI, class_id, angle, targets)。"""
        if self.target_source == "indoor":
            return generate_ROI().astype("float32"), None, 0.0, None
        if self.multi:
            roi, targets = generate_multi_target_sample()
            return roi, None, 0.0, targets
        roi, cid, ang = generate_ground_target_sample()
        return roi, cid, ang, None

    def __getitem__(self, idx):
        ROI_np, class_id, angle, targets = self._make_roi()
        # 点云（地面目标区域，物理坐标 → [-1,1] 归一化）
        point_cloud = extract_point_cloud_from_voxel(
            ROI_np, num_points=self.num_points, voxel_size=self.ch.voxel_size)
        max_extent = self.ch.roi_res * self.ch.voxel_size
        point_cloud = (point_cloud / max_extent) * 2.0 - 1.0
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        ROI_voxel = torch.tensor(ROI_np).reshape(-1)
        # X 特征（12 维）
        X_cpu = self.X_fixed.detach().cpu()
        X_amp_db = 20.0 * torch.log10(torch.abs(X_cpu).reshape(-1) + 1e-12)
        X_phase = torch.angle(X_cpu).reshape(-1)
        X_feat = torch.cat([X_amp_db, torch.sin(X_phase), torch.cos(X_phase)], dim=0).float()

        cond_list = []
        phases_seq = self._frame_phases(ROI_voxel)
        for t in range(self.tau):
            frame = self.ch.channels_per_frame[t]
            t_rel = self.ch.frames[t]["t_sec"]
            phases = phases_seq[t]

            Y_t = calculate_value_sat(
                ROI_voxel, phases, self.X_fixed, frame, self.power_sigma,
                t_rel, self.ch.wavelength_m)
            Y_feat = data_progress_amp_phase_db(Y_t.detach().cpu())     # 12 维（dB）

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

            # RCS 功率特征（物理：目标散射截面 → 回波功率，类别可分主特征）
            y_pow = torch.log10(Y_t.abs().pow(2).mean() + 1e-12)

            cond_t = torch.cat([X_feat, Y_feat, IRS_feat, dyn, y_pow.view(1)], dim=0).float()
            cond_list.append(cond_t)

        cond = torch.stack(cond_list).float()        # [Tau, cond_dim]
        if self.wideband:
            if self.isar:
                feat = torch.from_numpy(compute_isar_sequence(
                    ROI_np, self._target_ecef, self._ground_ecef, self.ch.wavelength_m,
                    snr_db=self.wideband_snr_db, seed=idx)).float()  # [M, K]
            else:
                feat = torch.from_numpy(compute_range_profile(
                    ROI_np, self._target_ecef, self._ground_ecef, self.ch.wavelength_m,
                    snr_db=self.wideband_snr_db, seed=idx, align=self.rp_align)).float()  # [K]
            if self.with_label:
                if self.multi:
                    return point_cloud.float(), cond, feat, targets
                return point_cloud.float(), cond, feat, class_id, float(angle)
            return point_cloud.float(), cond, feat
        if self.with_label:
            if self.multi:
                return point_cloud.float(), cond, targets
            return point_cloud.float(), cond, class_id, float(angle)
        return point_cloud.float(), cond


# ----------------------------------------------------------------------
# 便捷入口
# ----------------------------------------------------------------------

def build_sat_dataset(n_samples=16, irs_mode="sat", num_points=512,
                      tau=ss.TAU, device="cpu", seed=42, phase_mode="random",
                      target_source="ground"):
    """构建星-地 ISAC 数据集（smoke 用），返回 (dataset, scenario, channels)。"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    scenario = ss.SatISACScenario(tau=tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=irs_mode, device=device)
    dataset = SatROIDataset(n_samples=n_samples, channels=channels,
                            num_points=num_points, device=device, tau=tau,
                            phase_mode=phase_mode, target_source=target_source)
    return dataset, scenario, channels


if __name__ == "__main__":
    for mode in ["none", "sat", "ground"]:
        ds, _, ch = build_sat_dataset(n_samples=2, irs_mode=mode, num_points=256)
        pc, cond = ds[0]
        print(f"[{mode:>6}] point_cloud={tuple(pc.shape)} cond={tuple(cond.shape)} "
              f"cond_dim={ch.frame_cond_dim()} a={ch.tensor_a:.3e}")
