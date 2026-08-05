import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import setup

Scatter_x_1 = 1
Scatter_x_2 = 1
Scatter_x_3 = 1
Scatter_x_4 = 1


# =========================================================
# ROI 生成
# =========================================================
def generate_ROI():
    def initialize_space():
        return np.zeros(
            (setup.ROI_Length, setup.ROI_Length, setup.ROI_Length),
            dtype=np.float32
        )

    def place_object_1():
        obj = np.zeros((8, 8, 8), dtype=np.float32)
        obj[0:2, 0:2, 0:3] = 1
        obj[0:2, 6:8, 0:3] = 1
        obj[6:8, 0:2, 0:3] = 1
        obj[6:8, 6:8, 0:3] = 1
        obj[:, :, 3:5] = 1
        obj[7:, :, 5:8] = 1
        obj *= Scatter_x_1
        return obj

    def place_object_2():
        obj = np.zeros((8, 8, 8), dtype=np.float32)
        obj[2:6, 2:6, 0:5] = 1
        obj[:, :, 5:8] = 1
        obj *= Scatter_x_2
        return obj

    def place_object_3():
        obj = np.zeros((8, 8, 8), dtype=np.float32)
        obj[:, :, 5:8] = 1
        leg_coords = [(0, 0), (0, 6), (6, 0), (6, 6)]
        for lx, ly in leg_coords:
            obj[lx:lx + 2, ly:ly + 2, 0:5] = 1
        obj *= Scatter_x_3
        return obj

    def place_object_4():
        obj = np.ones((8, 8, 8), dtype=np.float32)
        obj *= Scatter_x_4
        return obj

    def check_space(space, x, y, z, obj_size=8):
        return np.all(space[x:x + obj_size, y:y + obj_size, z:z + obj_size] == 0)

    def randomly_place_objects(space):
        object_placers = [place_object_1, place_object_2, place_object_3, place_object_4]
        random.shuffle(object_placers)

        num_objects = 1

        for i in range(num_objects):
            obj = object_placers[i]()
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                x = random.randint(0, setup.ROI_Length - 8)
                y = random.randint(0, setup.ROI_Length - 8)
                z = random.randint(0, setup.ROI_Length - 8)
                if check_space(space, x, y, z, obj_size=8):
                    space[x:x + 8, y:y + 8, z:z + 8] = obj
                    placed = True
                attempts += 1
        return space

    space = initialize_space()
    space = randomly_place_objects(space)
    return space.astype(np.float32)


# =========================================================
# 点云生成
# =========================================================
def extract_point_cloud_from_voxel(ROI_np, num_points=2048, voxel_size=0.1):
    ROI_np = ROI_np.astype(np.float32)
    occ_indices = np.argwhere(ROI_np > 0.0)
    num_occ = len(occ_indices)

    if num_occ == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    centers = occ_indices.astype(np.float32) * voxel_size + (voxel_size / 2.0)

    base_count = num_points // num_occ
    remainder = num_points % num_occ

    if base_count > 0:
        idx_base = np.repeat(np.arange(num_occ), base_count)
    else:
        idx_base = np.array([], dtype=int)

    if remainder > 0:
        idx_rem = np.random.choice(num_occ, remainder, replace=False)
    else:
        idx_rem = np.array([], dtype=int)

    idx = np.concatenate([idx_base, idx_rem]).astype(int)
    sampled_centers = centers[idx]

    noise_range = voxel_size * 0.25
    jitter = np.random.uniform(
        low=-noise_range,
        high=noise_range,
        size=(num_points, 3)
    )

    point_cloud = sampled_centers + jitter
    return point_cloud.astype(np.float32)


def normalize_point_cloud_global(point_cloud, roi_length=16, voxel_size=0.1):
    physical_max = roi_length * voxel_size
    roi_center = physical_max / 2.0
    point_cloud_norm = (point_cloud - roi_center) / roi_center
    return point_cloud_norm.astype(np.float32)


def denormalize_point_cloud_global(point_cloud_norm, roi_length=16, voxel_size=0.1):
    physical_max = roi_length * voxel_size
    roi_center = physical_max / 2.0
    point_cloud_original = (point_cloud_norm * roi_center) + roi_center
    return point_cloud_original.astype(np.float32)


# =========================================================
# 无反射面遮挡：仅使用 BS 多视角
# =========================================================
def apply_occlusion_to_roi(ROI_raw, ld=0.05):
    L = ROI_raw.shape[0]

    pBS1 = setup.pBS[0:4]
    pBS2 = setup.pBS[4:8]
    pBS3 = setup.pBS[8:12]
    pBS4 = setup.pBS[12:16]

    indices = np.argwhere(ROI_raw != 0)
    if len(indices) == 0:
        return ROI_raw.astype(np.float32)

    roi_points = indices.astype(np.float32) * 0.1

    def visible_from_sources(sources):
        occ = np.ones((sources.shape[0], L, L, L), dtype=np.float32)

        for idx in indices:
            x, y, z = idx
            p1 = np.array([0.1 * x, 0.1 * y, 0.1 * z], dtype=np.float32)

            c = sources[:, None, :] - roi_points[None, :, :]
            b = sources[:, None, :] - p1[None, None, :]
            b = np.broadcast_to(b, c.shape)

            c_norm = np.linalg.norm(c, axis=-1)
            b_norm = np.linalg.norm(b, axis=-1) + 1e-8
            dot = np.sum(b * c, axis=-1)
            cos = np.clip(dot / (b_norm * c_norm + 1e-8), -1.0, 1.0)

            dis = b_norm * np.sqrt(np.maximum(0.0, 1.0 - cos ** 2))
            mask = np.logical_and(dis < ld, c_norm < b_norm)

            occ[:, x, y, z] = (np.count_nonzero(mask, axis=1) == 0).astype(np.float32)

        return occ

    def safe_visible_max(sources):
        vis = visible_from_sources(sources)
        if vis.shape[0] == 0:
            return np.zeros((L, L, L), dtype=np.float32)
        return np.max(vis, axis=0)

    r_bs1 = safe_visible_max(pBS1)
    r_bs2 = safe_visible_max(pBS2)
    r_bs3 = safe_visible_max(pBS3)
    r_bs4 = safe_visible_max(pBS4)

    result = ((r_bs1 + r_bs2 + r_bs3 + r_bs4) > 0).astype(np.float32)
    ROI_occ = ROI_raw * result
    return ROI_occ.astype(np.float32)


# =========================================================
# 16QAM：单帧固定导频向量 [16, 1]
# =========================================================
def build_fixed_16qam_pilot_vector(num_tx, device="cpu", normalize_power=True):
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device)

    real_idx = torch.arange(num_tx, device=device) % 4
    imag_idx = (torch.arange(num_tx, device=device) // 4) % 4

    real_part = levels[real_idx]
    imag_part = levels[imag_idx]

    x = torch.complex(real_part, imag_part).to(torch.complex64).unsqueeze(1)

    if normalize_power:
        x = x / math.sqrt(10.0)

    return x


# =========================================================
# 无反射面单帧信号生成：仅保留 BS-ROI-BS
# =========================================================
def simulate_received_signal(ROI_voxel, phase, X, H_dict, device):
    """
    ROI_voxel: [L, L, L]
    phase:     [IRS_total]，这里保留接口但不使用
    X:         [16, 1] complex
    返回:
        Y:      [16, 1] complex
    """
    H_BS_ROI = H_dict["H_BS_ROI"]
    H_ROI_BS = H_dict["H_ROI_BS"]

    S = ROI_voxel.reshape(-1).to(device)
    S_c = torch.complex(S.float(), torch.zeros_like(S.float()))

    Bmat = S_c[:, None] * H_ROI_BS.T
    H_BS_ROI_BS = H_BS_ROI.matmul(Bmat)  # [16,16]

    H_total = H_BS_ROI_BS
    Y = H_total.matmul(X)

    std = math.sqrt(setup.Power_sigma)
    noise = torch.complex(
        torch.randn_like(Y.real) * std,
        torch.randn_like(Y.imag) * std
    )
    return Y + noise


# =========================================================
# VAE 数据集
# =========================================================
class ROIVAEDataset(Dataset):
    def __init__(self, n_samples, num_points=2048):
        self.n = n_samples
        self.num_points = num_points

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ROI_raw = generate_ROI().astype(np.float32)
        roi_raw_t = torch.tensor(ROI_raw, dtype=torch.float32)

        point_cloud = extract_point_cloud_from_voxel(
            ROI_raw,
            num_points=self.num_points,
            voxel_size=0.1
        )
        point_cloud = normalize_point_cloud_global(
            point_cloud,
            roi_length=setup.ROI_Length,
            voxel_size=0.1
        )
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        return point_cloud, roi_raw_t


# =========================================================
# LDM 数据集
# =========================================================
class ROILDMDataset(Dataset):
    def __init__(self, n_samples, H_dict, num_points=2048, device="cpu",
                 irs_mode="none", phase_optimizer=None, save_dir=None):
        self.n = n_samples
        self.H_dict = H_dict
        self.device = device
        self.num_points = num_points
        self.irs_mode = irs_mode
        self.phase_optimizer = phase_optimizer
        self.samples = []

        X_fixed_cpu = build_fixed_16qam_pilot_vector(
            num_tx=setup.BS_Number,
            device="cpu",
            normalize_power=True
        ).to(torch.complex64)

        for i in range(self.n):
            if i % 1000 == 0:
                print(f"[ROILDMDataset-{irs_mode}] building sample {i}/{self.n}")

            ROI_raw = generate_ROI().astype(np.float32)
            ROI_occ = apply_occlusion_to_roi(ROI_raw).astype(np.float32)

            roi_raw_t = torch.tensor(ROI_raw, dtype=torch.float32)
            roi_occ_t = torch.tensor(ROI_occ, dtype=torch.float32)

            point_cloud = extract_point_cloud_from_voxel(
                ROI_raw,
                num_points=self.num_points,
                voxel_size=0.1
            )
            point_cloud = normalize_point_cloud_global(
                point_cloud,
                roi_length=setup.ROI_Length,
                voxel_size=0.1
            )
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

            if irs_mode == "optimized":
                phase_init = torch.zeros(setup.IRS_total, dtype=torch.float32)
            else:
                phase_init = torch.zeros(setup.IRS_total, dtype=torch.float32)

            self.samples.append((
                point_cloud,
                roi_raw_t,
                roi_occ_t,
                phase_init,
                X_fixed_cpu.clone()
            ))

        print(f"[ROILDMDataset-{irs_mode}] finish building all samples")

        # 预计算优化相位
        self.precomputed_phases = None
        if irs_mode == "optimized" and phase_optimizer is not None:
            self._precompute_phases(save_dir)

    def _precompute_phases(self, save_dir):
        import os
        cache_path = None
        if save_dir is not None:
            cache_path = os.path.join(save_dir, "precomputed_phases.pth")
            if os.path.exists(cache_path):
                print(f"[Precompute] loading cached phases from {cache_path}")
                self.precomputed_phases = torch.load(cache_path, map_location="cpu")
                print(f"[Precompute] loaded {self.precomputed_phases.shape[0]} samples")
                return

        print(f"[Precompute] optimizing phases for {self.n} samples x {setup.Tau} frames...")
        phases = []
        for i in range(self.n):
            if i % 100 == 0:
                print(f"[Precompute] sample {i}/{self.n}")
            _, _, roi_occ_t, _, X_fixed = self.samples[i]
            # 将输入移到 phase_optimizer 所在的设备
            roi_occ_t_dev = roi_occ_t.to(self.phase_optimizer.device)
            X_fixed_dev = X_fixed.to(self.phase_optimizer.device)
            sample_phases = []
            for t in range(setup.Tau):
                phase_t = self.phase_optimizer.optimize(roi_occ_t_dev, X_fixed_dev)
                sample_phases.append(phase_t.cpu())
            phases.append(torch.stack(sample_phases, dim=0))
        self.precomputed_phases = torch.stack(phases, dim=0)
        print(f"[Precompute] done. shape: {self.precomputed_phases.shape}")

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.precomputed_phases, cache_path)
            print(f"[Precompute] saved to {cache_path}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        point_cloud, roi_raw_t, roi_occ_t, phase_init, X_fixed = self.samples[idx]
        if self.irs_mode == "optimized" and self.precomputed_phases is not None:
            phase_seq = self.precomputed_phases[idx]
            return point_cloud, roi_raw_t, roi_occ_t, phase_seq, X_fixed
        return point_cloud, roi_raw_t, roi_occ_t, phase_init, X_fixed