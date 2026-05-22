# data.py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import math
import setup

def generate_ROI():
    def initialize_space(): return np.zeros((setup.ROI_Length, setup.ROI_Length, setup.ROI_Length))
    
    def object_11(space):
        obj = np.zeros((8, 8, 8))
        obj[[0, 0, 1, 1], [0, 1, 0, 1], 0:3] = 1
        obj[[0, 0, 1, 1], [6, 7, 6, 7], 0:3] = 1
        obj[[6, 6, 7, 7], [0, 1, 0, 1], 0:3] = 1
        obj[[6, 6, 7, 7], [6, 7, 6, 7], 0:3] = 1
        obj[:, :, 3:5] = 1; obj[7, :, 7] = 1; obj[7,:,5:7]=1
        return obj
        
    def object_22(space):
        obj = np.zeros((8, 8, 8), dtype=np.uint8)
        top_z_start, top_z_end = 5, 8
        for z in range(top_z_start, top_z_end):
            for x in range(8):
                for y in range(8):
                    if x in [0, 7] or y in [0, 7] or z in [top_z_start, top_z_end - 1]: obj[x, y, z] = 1
        leg_start, leg_end = 2, 6
        for z in range(0, 5):
            for x in range(leg_start, leg_end):
                for y in range(leg_start, leg_end):
                    if x in [leg_start, leg_end - 1] or y in [leg_start, leg_end - 1] or z in [0, 3]: obj[x, y, z] = 1
        return obj
        
    def object_33(space):
        obj = np.zeros((8, 8, 8), dtype=np.uint8)
        obj[:, :, 5:8] = 1; obj[1:7, 1:7, 6:7] = 0
        leg_coords = [(0, 0), (0, 6), (6, 0), (6, 6)]
        for lx, ly in leg_coords:
            obj[lx:lx + 2, ly:ly + 2, 0:5] = 1; obj[lx + 1:lx + 1, ly + 1:ly + 1, 1:3] = 0
        return obj
        
    def object_44(space):
        obj = np.ones((8, 8, 8))
        obj[2:6, 2:6, :] = 0; obj[:, :, 6:] = 1
        return obj

    def check_space(space, x, y, z): return np.all(space[x:x + 8, y:y + 8, z:z + 8] == 0)
    
    def randomly_place_objects(space):
        object_placers = [object_11, object_22, object_33, object_44]
        random.shuffle(object_placers)
        
        # 允许随机放置 1 到 2 个物体，恢复你想要的复杂场景
        num_objects = random.randint(1, 1)
        
        for i in range(num_objects):
            obj = object_placers[i](space)
            placed = False
            attempts = 0
            
            # 加入最大尝试次数防死循环保护机制
            while not placed and attempts < 100:
                x = random.randint(0, setup.ROI_Length - 8)
                y = random.randint(0, setup.ROI_Length - 8)
                z = random.randint(0, setup.ROI_Length - 8)
                if check_space(space, x, y, z):
                    space[x:x + 8, y:y + 8, z:z + 8] = obj
                    placed = True
                attempts += 1
                
        return space

    space = initialize_space()
    space = randomly_place_objects(space)
    return space

def extract_point_cloud_from_voxel(ROI_np, num_points=2048, voxel_size=0.1):
    occ_indices = np.argwhere(ROI_np > 0.5)
    if len(occ_indices) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    
    centers = occ_indices.astype(np.float32) * voxel_size + (voxel_size / 2.0)
    idx = np.random.choice(len(centers), num_points, replace=True)
    sampled_centers = centers[idx]
    
    jitter = np.random.uniform(low=-voxel_size/2.0, high=voxel_size/2.0, size=(num_points, 3))
    point_cloud = sampled_centers + jitter
    return point_cloud.astype(np.float32)

def data_progress_amp_phase(data_complex: torch.Tensor) -> torch.Tensor:
    amp = torch.abs(data_complex).reshape(-1) # 不能减去 mean 和除以 std！直接使用真实物理幅值
    phase = torch.angle(data_complex).reshape(-1)
    return torch.cat([amp, torch.sin(phase), torch.cos(phase)], dim=0).float()



def calculate_value_ROI_simple(ROI_voxel, phase, X, H_dict, Power_sigma, device="cuda"):
    H_ROI_UE = H_dict["H_ROI_UE"]
    H_IRS1_UE = H_dict["H_IRS1_UE"]
    H_IRS2_UE = H_dict["H_IRS2_UE"]
    H_ROI_IRS1 = H_dict["H_ROI_IRS1"]
    H_ROI_IRS2 = H_dict["H_ROI_IRS2"]
    H_IRS1_ROI = H_dict["H_IRS1_ROI"]
    H_IRS2_ROI = H_dict["H_IRS2_ROI"]
    H_BS_ROI = H_dict["H_BS_ROI"]
    H_BS_IRS1 = H_dict["H_BS_IRS1"]
    H_BS_IRS2 = H_dict["H_BS_IRS2"]

    IRS_Number = setup.IRS_Number
    S = ROI_voxel.reshape(-1).to(device)
    S_c = torch.complex(S.float(), torch.zeros_like(S.float()))

    v1 = torch.exp(1j * phase[:IRS_Number]).to(torch.complex64)
    v2 = torch.exp(1j * phase[IRS_Number:]).to(torch.complex64)

    Bmat = S_c[:, None] * H_ROI_UE
    H_BS_ROI_UE = H_BS_ROI.matmul(Bmat)
    C = H_BS_ROI * S_c[None, :]

    C1 = C.matmul(H_ROI_IRS1)
    C2 = C.matmul(H_ROI_IRS2)
    H_BS_ROI_IRS1_UE = (C1 * v1[None, :]).matmul(H_IRS1_UE)
    H_BS_ROI_IRS2_UE = (C2 * v2[None, :]).matmul(H_IRS2_UE)

    IR1 = H_IRS1_ROI * S_c[None, :]
    IR2 = H_IRS2_ROI * S_c[None, :]
    B1 = IR1.matmul(H_ROI_UE)
    B2 = IR2.matmul(H_ROI_UE)
    H_BS_IRS1_ROI_UE = (H_BS_IRS1 * v1[None, :]).matmul(B1)
    H_BS_IRS2_ROI_UE = (H_BS_IRS2 * v2[None, :]).matmul(B2)

    H_total = (H_BS_ROI_UE + H_BS_ROI_IRS1_UE + H_BS_ROI_IRS2_UE + H_BS_IRS1_ROI_UE + H_BS_IRS2_ROI_UE)
    receive_signals = H_total.matmul(X)

    std = math.sqrt(Power_sigma)
    noise = torch.complex(torch.randn_like(receive_signals.real) * std, torch.randn_like(receive_signals.imag) * std)
    return receive_signals + noise

class ROIPairedDataset(Dataset):
    def __init__(self, n_samples, H_dict, num_points=2048, device="cuda"):
        self.n = n_samples
        self.H_dict = H_dict
        self.device = device
        self.num_points = num_points
        self.Signal1 = torch.tensor([-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j, -1+1j, 3-3j, 3-1j, 3+3j, 3+1j, 1-3j, 1-1j, 1+3j, 1+1j], dtype=torch.complex64, device=device)
        pilot = torch.stack([self.Signal1[0], self.Signal1[1], self.Signal1[2], self.Signal1[3]]).view(4, 1)
        self.tensor_a = self.H_dict["tensor_a"]
        self.X_fixed = self.tensor_a * pilot

    def __len__(self): return self.n

    def __getitem__(self, idx):
        ROI_np = generate_ROI().astype("float32")
        ROI_voxel = torch.tensor(ROI_np)[None, :, :, :]
        
        # 提取点云，目前坐标位于 [0, 1.6] 物理区间内
        point_cloud = extract_point_cloud_from_voxel(ROI_np, num_points=self.num_points, voxel_size=0.1)
        
        # ========================================================
        # 严谨的绝对坐标归一化 (定位与重建联合任务的核心)
        # ========================================================
        # 房间最大尺寸为 16 * 0.1 = 1.6 米。
        # 将绝对物理坐标直接映射到 [-1, 1]，完美保留物体在房间里的绝对位置。
        max_extent = setup.ROI_Length * 0.1
        point_cloud = (point_cloud / max_extent) * 2.0 - 1.0
        point_cloud = torch.tensor(point_cloud)

        # ========================================================
        # 提取基准信号 X 的物理特征 (幅值与相位) -> [12维]
        # ========================================================
        X_fixed_cpu = self.X_fixed.detach().cpu()
        X_amp = torch.abs(X_fixed_cpu).reshape(-1)
        # 同样删除这里的 X_amp_z 计算，直接用 X_amp
        X_phase = torch.angle(X_fixed_cpu).reshape(-1)
        X_feat = torch.cat([X_amp, torch.sin(X_phase), torch.cos(X_phase)], dim=0).float()
        # 产生 [Tau, 32] 维度的均匀分布动态相位，激活散射多样性
        phases = torch.rand(setup.Tau, setup.IRS_total).float() * 2 * np.pi
        
        cond_list = []
        for t in range(setup.Tau):
            # 将每一帧的动态相位送入计算物理方程 Y = XH + N
            Y_t = calculate_value_ROI_simple(ROI_voxel, phases[t].to(self.device), self.X_fixed, self.H_dict, setup.Power_sigma, self.device)
            
            # 提取接收信号 Y_t 的特征 (幅值与相位) -> [12维]
            Y_feat = data_progress_amp_phase(Y_t.detach().cpu()) 
            
            # 提取这一帧 IRS 反射板的具体相位状态 -> [64维] (32维sin + 32维cos)
            phi_t = phases[t].reshape(-1)
            IRS_feat = torch.cat([torch.sin(phi_t), torch.cos(phi_t)], dim=0).float()
            
            # 【物理特征集结】：X特征(12) + Y特征(12) + IRS特征(64) = 88维
            cond_t = torch.cat([X_feat, Y_feat, IRS_feat], dim=0)
            cond_list.append(cond_t)

        # 最终条件张量形状: [8, 88]
        cond = torch.cat(cond_list, dim=0).view(setup.Tau, -1).float()
        return point_cloud.float(), cond