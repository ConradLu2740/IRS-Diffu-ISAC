# IRS_Diffu_ISAC 项目 Code Wiki

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目架构总览](#2-项目架构总览)
3. [目录结构](#3-目录结构)
4. [模块详解](#4-模块详解)
   - 4.1 [setup.py — 仿真参数与信道预计算](#41-setuppy--仿真参数与信道预计算)
   - 4.2 [data.py — IRS 场景数据处理](#42-datapy--irs-场景数据处理)
   - 4.3 [data_no_irs.py — 无 IRS 场景数据处理](#43-data_no_irspy--无-irs-场景数据处理)
   - 4.4 [models.py — 核心模型定义](#44-modelspy--核心模型定义)
   - 4.5 [models_unet.py — U-Net 模型定义](#45-models_unetpy--u-net-模型定义)
   - 4.6 [train.py — 主训练脚本](#46-trainpy--主训练脚本)
   - 4.7 [train_no.py — 无 IRS 训练脚本](#47-train_nopy--无-irs-训练脚本)
   - 4.8 [train_r.py — 随机 IRS 训练脚本](#48-train_rpy--随机-irs-训练脚本)
5. [关键类与函数索引](#5-关键类与函数索引)
6. [数据流与训练管线](#6-数据流与训练管线)
7. [依赖关系](#7-依赖关系)
8. [项目运行方式](#8-项目运行方式)
9. [已知问题与代码不一致性](#9-已知问题与代码不一致性)
10. [公平实验对比框架（新增）](#10-公平实验对比框架新增)
   - 10.1 [学术短板分析](#101-学术短板分析)
   - 10.2 [train_unified.py — 统一训练脚本](#102-train_unifiedpy--统一训练脚本)
   - 10.3 [phase_optimizer.py — 相位优化模块](#103-phase_optimizerpy--相位优化模块)
   - 10.4 [run_experiments.py — 批量实验调度](#104-run_experimentspy--批量实验调度)
   - 10.5 [analyze_results.py — 统计分析与可视化](#105-analyze_resultspy--统计分析与可视化)

---

## 1. 项目概述

### 1.1 项目背景

IRS_Diffu_ISAC 是一个将**智能反射面（Intelligent Reflecting Surface, IRS）**技术与**扩散模型（Diffusion Model）**相结合的深度学习研究项目，面向**通感一体化（Integrated Sensing and Communication, ISAC）**场景。

### 1.2 项目目标

利用接收到的无线信号（含 IRS 反射路径），通过条件潜在扩散模型（Conditional Latent Diffusion Model）重建感兴趣区域（Region of Interest, ROI）内物体的 **3D 点云**表示。

### 1.3 技术路线

项目采用**两阶段训练管线**：

1. **Stage 1 — PointVAE 训练**：训练一个点云变分自编码器（Point Cloud VAE），将 3D 点云编码到低维潜在空间，并能够从潜在表示解码重建点云。
2. **Stage 2 — Latent Diffusion Model 训练**：在 PointVAE 的潜在空间中训练条件扩散模型，以物理信号特征（发射信号、接收信号、IRS 相位）为条件，生成目标点云的潜在表示，再通过 VAE 解码器还原为 3D 点云。

### 1.3 三种实验场景

| 场景 | 训练脚本 | IRS 状态 | 数据模块 | 信号模型 |
|------|----------|----------|----------|----------|
| IRS + 随机相位 | `train.py` | 启用，每帧随机相位 | `data.py` | 5 条传播路径（含 IRS 反射） |
| 无 IRS | `train_no.py` | 禁用，相位为零占位 | `data_no_irs.py` | 仅 BS-ROI-BS 直射路径 |
| 随机 IRS | `train_r.py` | 启用，每帧随机相位 | `data.py` | 5 条传播路径（含 IRS 反射） |

---

## 2. 项目架构总览

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     IRS_Diffu_ISAC 系统架构                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │ setup.py │───▶│ data.py  │───▶│      train.py            │  │
│  │ 信道预计算│    │ data_no  │    │      train_no.py         │  │
│  └──────────┘    │ _irs.py  │    │      train_r.py          │  │
│                  └──────────┘    └──────────┬───────────────┘  │
│                                           │                    │
│                                           ▼                    │
│                                  ┌──────────────────┐         │
│                                  │    models.py     │         │
│                                  │  models_unet.py  │         │
│                                  └──────────────────┘         │
│                                                               │
│  训练管线:                                                     │
│  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ ROI 生成│──▶│ 信号仿真  │──▶│ VAE 编码  │──▶│ LDM 训练  │    │
│  └────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │                                            │          │
│       ▼                                            ▼          │
│  ┌────────┐                                  ┌──────────┐    │
│  │ 点云提取│                                  │ 条件采样  │    │
│  └────────┘                                  └────┬─────┘    │
│                                                   │          │
│                                                   ▼          │
│                                            ┌──────────┐     │
│                                            │ VAE 解码  │     │
│                                            └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 两阶段训练管线

```
Stage 1: PointVAE 训练
═══════════════════════
  点云 X ──▶ [Encoder] ──▶ (μ, logvar) ──▶ 重参数化 ──▶ z ──▶ [Decoder] ──▶ 重建点云 X̂
                                                              │
                                              Loss = ChamferDistance(X, X̂) + KL(μ, logvar)

Stage 2: Latent Diffusion Model 训练
═════════════════════════════════════
  物理信号条件 c ──▶ [CondEncoder] ──▶ c_seq
                                       │
  z₀ ──▶ 前向加噪 q(zₜ|z₀) ──▶ zₜ ──▶ [DiT/UNet] ──▶ ε̂(zₜ, t, c_seq)
                                       │
                              Loss = MSE(ε̂, ε)

推理阶段:
═══════
  随机噪声 zₜ ──▶ [DDPM 反向去噪 T 步] ──▶ z₀ ──▶ [VAE Decoder] ──▶ 点云 X̂
                      │
                 条件 c = 物理信号特征
                 使用 Classifier-Free Guidance (CFG)
```

### 2.3 模块依赖关系图

```
setup.py ◄────────────────────────────────────────────┐
   │                                                   │
   ├──▶ data.py                                        │
   │      │                                            │
   │      ├──▶ ROIPairedDataset                        │
   │      └──▶ calculate_value_ROI_simple              │
   │                                                   │
   ├──▶ data_no_irs.py                                 │
   │      │                                            │
   │      ├──▶ ROIVAEDataset                           │
   │      ├──▶ ROILDMDataset                           │
   │      └──▶ simulate_received_signal                │
   │                                                   │
   ├──▶ models.py ◄────────────────────────────────────┤
   │      │                                            │
   │      ├──▶ PointVAE                                │
   │      ├──▶ AdvancedCondEncoder                     │
   │      ├──▶ LatentDiT1D_CrossAttn                   │
   │      └──▶ LatentDiT_Token_CrossAttn (⚠️ 未定义)   │
   │                                                   │
   ├──▶ models_unet.py                                 │
   │      │                                            │
   │      └──▶ UNet1DLatent                            │
   │                                                   │
   ├──▶ train.py ◄── data.py, models.py               │
   ├──▶ train_no.py ◄── data_no_irs.py, models.py     │
   └──▶ train_r.py ◄── data.py, models.py             │
```

---

## 3. 目录结构

```
IRS_Diffu_ISAC/
└── source_code/
    ├── setup.py          # 仿真参数配置与物理信道预计算
    ├── data.py           # IRS 场景数据生成与信号计算
    ├── data_no_irs.py    # 无 IRS 场景数据生成、遮挡模拟与信号仿真
    ├── models.py         # 神经网络模型定义（PointVAE、条件编码器、DiT）
    ├── models_unet.py    # U-Net 架构模型定义（带 FiLM 调制的 1D U-Net）
    ├── train.py          # 主训练脚本（IRS + 随机相位场景）
    ├── train_no.py       # 无 IRS 场景训练脚本
    └── train_r.py        # 随机 IRS 相位场景训练脚本
```

---

## 4. 模块详解

### 4.1 setup.py — 仿真参数与信道预计算

**文件路径**: `source_code/setup.py`（124 行）

**职责**: 定义无线通信仿真所需的全部物理参数、坐标配置，并提供信道矩阵的预计算功能。

#### 4.1.1 全局仿真参数

| 参数 | 值 | 说明 |
|------|----|------|
| `Tau` | 8 | 时间步数（信号帧数） |
| `Time_slot` | 1 | 时隙长度 |
| `P_SNR` | 20 | 信噪比（dB） |
| `Power_sigma` | 0.01 | 噪声功率 |
| `ROI_Length` | 16 | ROI 网格边长（体素数） |
| `ROI_Number` | 4096 | ROI 体素总数（16³） |
| `N_IRS` | 4 | IRS 面板每行/列元素数 |
| `IRS_Number` | 16 | 单个 IRS 面板元素总数（4×4） |
| `IRS_total` | 32 | 两个 IRS 面板元素总数（2×16） |

#### 4.1.2 坐标配置

| 变量 | 形状 | 说明 |
|------|------|------|
| `start` | (3,) | ROI 起始坐标 [0.1, 0.1, 0.1] |
| `end` | (3,) | ROI 结束坐标 [1.6, 1.6, 1.6] |
| `pBS_0` | (3,) | 参考基站位置 [-4, 0.5, 2] |
| `pROI_0` | (3,) | 参考 ROI 中心 [0.5, 0.5, 0.5] |
| `pUE_0` | (3,) | 参考用户位置 [2, 0.5, 0] |
| `pUE` | (4, 3) | 4 个用户位置 |
| `pBS` | (4, 3) | 4 个基站位置 |
| `pIRS_P1` | (3,) | IRS 面板 1 中心 [1.0, 2.0, 1.0] |
| `pIRS_P2` | (3,) | IRS 面板 2 中心 [-1.0, -1.0, 1.0] |

#### 4.1.3 关键函数

##### `Power_SNR(sigma, snr) -> float`
根据噪声功率和 SNR 计算发射功率。
- **参数**: `sigma` — 噪声功率; `snr` — 信噪比（dB）
- **返回**: 发射功率值
- **公式**: `Power = 10^(snr/10) * sigma * 64`

##### `get_Channel(a, b) -> np.ndarray`
计算两点/点集之间的信道系数（复数）。
- **参数**: `a`, `b` — 坐标数组，支持广播
- **返回**: 复数信道系数 `H = √0.1 * exp(j*2π*d/0.01) / d`
- **说明**: 基于自由空间传播模型，载波波长 0.01m

##### `make_irs_rotated() -> (np.ndarray, np.ndarray)`
生成两个 IRS 面板的旋转后坐标。
- **返回**: `(pIRS_1_rotated, pIRS_2_rotated)` — 各形状为 (16, 3)
- **说明**: IRS 面板 1 旋转使法线对准 (1,1,1) 方向；面板 2 对准 (1,1,-1) 方向

##### `make_roi_grid() -> np.ndarray`
生成 ROI 区域的均匀网格点坐标。
- **返回**: 形状 (4096, 3) 的坐标数组

##### `precompute_channels(device) -> dict`
预计算所有信道矩阵并转为 GPU 张量。
- **参数**: `device` — 计算设备（"cuda" 或 "cpu"）
- **返回**: `H_dict` 字典，包含以下键：

| 键名 | 形状 | 说明 |
|------|------|------|
| `tensor_a` | 标量 | 发射功率缩放因子 |
| `H_ROI_UE` | (4096, 4) | ROI→UE 信道 |
| `H_IRS1_UE` | (16, 4) | IRS1→UE 信道 |
| `H_IRS2_UE` | (16, 4) | IRS2→UE 信道 |
| `H_ROI_IRS1` | (4096, 16) | ROI→IRS1 信道 |
| `H_ROI_IRS2` | (4096, 16) | ROI→IRS2 信道 |
| `H_IRS1_ROI` | (4096, 16) | IRS1→ROI 信道 |
| `H_IRS2_ROI` | (4096, 16) | IRS2→ROI 信道 |
| `H_BS_ROI` | (4096, 4) | BS→ROI 信道 |
| `H_BS_IRS1` | (16, 4) | BS→IRS1 信道 |
| `H_BS_IRS2` | (16, 4) | BS→IRS2 信道 |

---

### 4.2 data.py — IRS 场景数据处理

**文件路径**: `source_code/data.py`（198 行）

**职责**: 生成 IRS 场景下的训练数据，包括 ROI 体素生成、点云提取、物理信号计算与特征构造。

#### 4.2.1 关键函数

##### `generate_ROI() -> np.ndarray`
生成包含随机放置物体的 ROI 体素空间。
- **返回**: 形状 (16, 16, 16) 的 float32 数组，1 表示占据，0 表示空闲
- **说明**: 定义了 4 种物体模板：
  - `object_11` — 桌形结构（4 腿 + 平台 + 背板）
  - `object_22` — 空心箱体（外壳 + 内部中空 + 底座）
  - `object_33` — 桌子变体（顶面 + 4 腿）
  - `object_44` — 带通道的实体块
- 每次随机放置 1 个物体，位置随机，最大尝试 100 次

##### `extract_point_cloud_from_voxel(ROI_np, num_points=2048, voxel_size=0.1) -> np.ndarray`
从体素空间提取点云。
- **参数**: `ROI_np` — 体素数组; `num_points` — 采样点数; `voxel_size` — 体素尺寸（米）
- **返回**: 形状 (num_points, 3) 的 float32 点云数组
- **说明**: 在占据体素中心均匀采样，添加 [-voxel_size/2, voxel_size/2] 范围的随机抖动

##### `data_progress_amp_phase(data_complex: torch.Tensor) -> torch.Tensor`
从复数信号提取幅值与相位特征。
- **参数**: `data_complex` — 复数张量
- **返回**: 拼接张量 `[|Y|, sin(∠Y), cos(∠Y)]`

##### `calculate_value_ROI_simple(ROI_voxel, phase, X, H_dict, Power_sigma, device) -> torch.Tensor`
计算 IRS 场景下的接收信号，考虑 5 条传播路径。
- **参数**:
  - `ROI_voxel` — 形状 (1, L, L, L) 的 ROI 体素
  - `phase` — 形状 (IRS_total,) 的 IRS 相位向量
  - `X` — 形状 (4, 1) 的发射信号
  - `H_dict` — 预计算信道字典
  - `Power_sigma` — 噪声功率
- **返回**: 形状 (4, 1) 的复数接收信号
- **5 条传播路径**:
  1. **BS → ROI → UE**（直射散射路径）
  2. **BS → ROI → IRS1 → UE**（经 IRS1 反射）
  3. **BS → ROI → IRS2 → UE**（经 IRS2 反射）
  4. **BS → IRS1 → ROI → UE**（IRS1 前向散射）
  5. **BS → IRS2 → ROI → UE**（IRS2 前向散射）

#### 4.2.2 数据集类

##### `ROIPairedDataset(Dataset)`
生成点云与条件信号配对数据集。

| 属性 | 说明 |
|------|------|
| `n_samples` | 样本数量 |
| `H_dict` | 预计算信道字典 |
| `num_points` | 点云采样点数 |
| `Signal1` | 16QAM 信号星座点 |
| `X_fixed` | 固定导频信号，形状 (4, 1) |

**`__getitem__(idx)` 返回**:
- `point_cloud` — 形状 (num_points, 3)，归一化到 [-1, 1]
- `cond` — 形状 (Tau, 88)，条件特征序列
  - 每帧 88 维 = X 特征(12) + Y 特征(12) + IRS 相位特征(64)

---

### 4.3 data_no_irs.py — 无 IRS 场景数据处理

**文件路径**: `source_code/data_no_irs.py`（326 行）

**职责**: 生成无 IRS 场景下的训练数据，增加了遮挡模拟和更丰富的信号处理功能。

#### 4.3.1 全局参数

| 参数 | 值 | 说明 |
|------|----|------|
| `Scatter_x_1` | 1 | 物体 1 散射系数 |
| `Scatter_x_2` | 1 | 物体 2 散射系数 |
| `Scatter_x_3` | 1 | 物体 3 散射系数 |
| `Scatter_x_4` | 1 | 物体 4 散射系数 |

#### 4.3.2 关键函数

##### `generate_ROI() -> np.ndarray`
生成 ROI 体素空间（与 data.py 类似但物体定义不同）。
- 4 种物体模板：`place_object_1` 到 `place_object_4`
- 每个物体乘以对应散射系数
- 固定放置 1 个物体

##### `extract_point_cloud_from_voxel(ROI_np, num_points=2048, voxel_size=0.1) -> np.ndarray`
从体素提取点云（均匀分配 + 随机补充策略）。
- **说明**: 先均匀分配基础采样数，剩余点随机补充，抖动范围为 voxel_size × 0.25

##### `normalize_point_cloud_global(point_cloud, roi_length=16, voxel_size=0.1) -> np.ndarray`
全局归一化点云坐标到 [-1, 1]。
- **公式**: `pc_norm = (pc - center) / center`，其中 `center = roi_length * voxel_size / 2`

##### `denormalize_point_cloud_global(point_cloud_norm, roi_length=16, voxel_size=0.1) -> np.ndarray`
反归一化，从 [-1, 1] 恢复到物理坐标。

##### `apply_occlusion_to_roi(ROI_raw, ld=0.05) -> np.ndarray`
基于基站视角的遮挡模拟。
- **参数**: `ROI_raw` — 原始 ROI; `ld` — 遮挡判定距离阈值
- **返回**: 遮挡后的 ROI（不可见体素置零）
- **说明**: 将 4 个基站分为 4 组，每组计算可见性，任一基站可见即保留

##### `build_fixed_16qam_pilot_vector(num_tx, device="cpu", normalize_power=True) -> torch.Tensor`
构建 16QAM 固定导频向量。
- **参数**: `num_tx` — 发射天线数; `normalize_power` — 是否归一化功率
- **返回**: 形状 (num_tx, 1) 的复数导频向量
- **说明**: 使用 16QAM 星座点 {-3,-1,1,3}×{-3,-1,1,3}，功率归一化因子 √10

##### `simulate_received_signal(ROI_voxel, phase, X, H_dict, device) -> torch.Tensor`
无 IRS 场景的接收信号仿真。
- **参数**: 与 `calculate_value_ROI_simple` 类似，但 `phase` 参数保留接口不使用
- **返回**: 形状 (16, 1) 的复数接收信号
- **信号模型**: `Y = H_BS_ROI_BS * X + N`
  - 仅考虑 BS → ROI → BS 直射散射路径
  - `H_BS_ROI_BS = H_BS_ROI @ diag(S) @ H_ROI_BS`

#### 4.3.3 数据集类

##### `ROIVAEDataset(Dataset)`
仅用于 VAE 训练的简单数据集。

**`__getitem__(idx)` 返回**:
- `point_cloud` — 形状 (num_points, 3)，归一化点云
- `roi_raw_t` — 形状 (16, 16, 16)，原始 ROI 体素

##### `ROILDMDataset(Dataset)`
用于 LDM 训练的完整数据集，在初始化时预生成全部样本。

**`__getitem__(idx)` 返回**（5 元组）:
- `point_cloud` — 形状 (num_points, 3)，归一化点云
- `roi_raw_t` — 形状 (16, 16, 16)，原始 ROI 体素
- `roi_occ_t` — 形状 (16, 16, 16)，遮挡后 ROI 体素
- `phase_init` — 形状 (IRS_total,)，初始相位（全零占位）
- `X_fixed` — 形状 (BS_Number, 1)，固定导频信号

---

### 4.4 models.py — 核心模型定义

**文件路径**: `source_code/models.py`（218 行）

**职责**: 定义项目中的核心神经网络模型，包括点云 VAE、条件编码器和基于 DiT 的扩散模型。

#### 4.4.1 PointVAE

```python
class PointVAE(nn.Module):
    def __init__(self, num_points=2048, z_dim=256)
```

点云变分自编码器，将 3D 点云编码为低维潜在向量并重建。

| 组件 | 结构 | 说明 |
|------|------|------|
| **Encoder** | Conv1d(3→64→128→256→512→1024) + MaxPool | 逐点卷积 + 全局最大池化 |
| **fc_mu** | Linear(1024, z_dim) | 均值投影 |
| **fc_logvar** | Linear(1024, z_dim) | 对数方差投影（clamp 到 [-20, 20]） |
| **Decoder** | Linear(z_dim→512→1024→2048→num_points*3) | MLP 解码器 |

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `encode` | `(x: [B, N, 3]) -> (mu: [B, z_dim], logvar: [B, z_dim])` | 编码点云为均值和方差 |
| `reparam` | `(mu, logvar) -> z: [B, z_dim]` | 重参数化技巧采样 |
| `decode` | `(z: [B, z_dim]) -> [B, N, 3]` | 从潜在向量解码点云 |
| `forward` | `(x) -> (reconstructed_x, mu, logvar, z)` | 完整前向传播 |

#### 4.4.2 AdvancedCondEncoder

```python
class AdvancedCondEncoder(nn.Module):
    def __init__(self, seq_len=8, input_size=114, hidden_size=128, out_emb=256)
```

条件编码器，将物理信号序列编码为条件嵌入序列。

| 组件 | 结构 | 输入→输出形状 |
|------|------|--------------|
| **LSTM** | 2 层双向 LSTM | [B, 8, input_size] → [B, 8, hidden_size×2] |
| **feature_proj** | Linear + LayerNorm + GELU + Linear | [B, 8, hidden_size×2] → [B, 8, hidden_size] |
| **TransformerEncoder** | 2 层 TransformerEncoderLayer (4 heads) | [B, 8, hidden_size] → [B, 8, hidden_size] |
| **final_proj** | Linear + LayerNorm + SiLU + Linear | [B, 8, hidden_size] → [B, 8, out_emb] |

**前向传播**: `forward(physical_signals: [B, Tau, input_size]) -> c_seq: [B, Tau, out_emb]`

#### 4.4.3 TimestepEmbedder

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256)
```

正弦位置编码器，将扩散时间步编码为嵌入向量。
- **输入**: `t: [B]`（整数时间步）
- **输出**: `[B, hidden_size]`
- **实现**: 正弦/余弦频率编码 + 2 层 MLP（SiLU 激活）

#### 4.4.4 DiTBlock_CrossAttn

```python
class DiTBlock_CrossAttn(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0)
```

带交叉注意力的 DiT（Diffusion Transformer）块。

| 子模块 | 说明 |
|--------|------|
| **Self-Attention** | 自注意力层，关注潜在 token 内部结构 |
| **Cross-Attention** | 交叉注意力层，query=x, key/value=c_seq |
| **MLP** | 2 层 MLP（GELU 激活），扩展比 4× |
| **AdaLN** | 自适应层归一化，由时间步嵌入调制（6 个参数：shift/scale/gate × 2） |

**前向传播**: `forward(x: [B, T, D], t_emb: [B, D], c_seq: [B, S, D]) -> [B, T, D]`

#### 4.4.5 FinalLayer

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels)
```

最终输出层，带 AdaLN 调制。
- **AdaLN**: 由时间步嵌入生成 shift 和 scale（2 个参数）
- **输出**: Linear(hidden_size, out_channels)

#### 4.4.6 LatentDiT1D_CrossAttn

```python
class LatentDiT1D_CrossAttn(nn.Module):
    def __init__(self, z_dim=256, cond_emb=256, hidden_size=256, depth=4, num_heads=8)
```

1D 潜在空间 DiT 扩散模型，使用交叉注意力融合条件信息。

| 组件 | 说明 |
|------|------|
| **x_embed** | Linear(token_dim, hidden_size)，token 嵌入 |
| **pos_embed** | 可学习位置编码，形状 (1, 16, hidden_size) |
| **t_embedder** | TimestepEmbedder，时间步编码 |
| **cond_proj** | Linear(cond_emb, hidden_size)，条件投影 |
| **blocks** | 4 个 DiTBlock_CrossAttn 块 |
| **final_layer** | FinalLayer，输出预测噪声 |

**关键设计**:
- 将 256 维潜在向量切分为 16 个 token（每个 16 维）
- 条件序列通过交叉注意力注入每个 DiT 块
- 初始化时 AdaLN 和 FinalLayer 的最后一层权重/偏置置零

**前向传播**: `forward(zt: [B, z_dim], t: [B], cond_seq: [B, Tau, cond_emb]) -> [B, z_dim]`

---

### 4.5 models_unet.py — U-Net 模型定义

**文件路径**: `source_code/models_unet.py`（359 行）

**职责**: 定义基于 U-Net 架构的 1D 潜在扩散模型，使用 FiLM 调制和 Classifier-Free Guidance。

#### 4.5.1 辅助模块

##### `TimestepEmbedder`
与 models.py 中同名类功能相同，正弦位置编码 + MLP。

##### `ConditionPool(cond_emb=256, out_dim=256)`
条件序列池化模块。
- **输入**: `cond_seq: [B, Tau, cond_emb]`
- **操作**: 对时间维度取均值 → 2 层 MLP（SiLU 激活）
- **输出**: `[B, out_dim]`

##### `FiLM(cond_dim, num_channels)`
Feature-wise Linear Modulation 模块。
- **输入**: `x: [B, C, L]`, `cond_vec: [B, cond_dim]`
- **操作**: 从条件向量生成 scale 和 shift，调制特征图
- **公式**: `x * (1 + scale) + shift`

##### `ResBlock1D(in_ch, out_ch, cond_dim, dropout=0.0)`
带 FiLM 调制的 1D 残差块。
- **结构**: GroupNorm → FiLM → SiLU → Conv1d → GroupNorm → FiLM → SiLU → Dropout → Conv1d + Skip Connection
- **条件注入**: 两次 FiLM 调制（归一化后、卷积前）

##### `SelfAttention1D(channels, num_heads=4)`
1D 自注意力模块。
- **结构**: GroupNorm → Conv1d(QKV) → Multi-Head Attention → Conv1d(Proj) + Residual

##### `Downsample1D(channels)` / `Upsample1D(channels)`
1D 下采样/上采样模块。
- **下采样**: Conv1d(kernel=4, stride=2, padding=1)
- **上采样**: Nearest 插值(×2) + Conv1d(kernel=3, padding=1)

#### 4.5.2 UNet1DLatent

```python
class UNet1DLatent(nn.Module):
    def __init__(
        self,
        token_dim=32,
        num_latent_tokens=32,
        cond_emb=256,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_heads=4,
        dropout=0.0,
        max_cond_len=8,
    )
```

完整的 1D U-Net 潜在扩散模型。

**架构**:

```
输入: [B, 32, 32] (token 形式)
  │
  ├── in_proj: Conv1d(32, 128)
  │
  ├── Encoder:
  │   ├── enc1: ResBlock(128→128) + SelfAttn + Downsample
  │   ├── enc2: ResBlock(128→256) + SelfAttn + Downsample
  │   └── enc3: ResBlock(256→512) + SelfAttn
  │
  ├── Middle:
  │   ├── mid1: ResBlock(512→512)
  │   ├── mid_attn: SelfAttn
  │   └── mid2: ResBlock(512→512)
  │
  ├── Decoder:
  │   ├── up2 + cat(e2) → dec2: ResBlock(768→256) + SelfAttn
  │   └── up1 + cat(e1) → dec1: ResBlock(384→128) + SelfAttn
  │
  ├── out_norm + out_act + out_proj: Conv1d(128, 32)
  │
  输出: [B, 32, 32] (token 形式)
```

**条件注入方式**:
1. 时间步 `t` → `TimestepEmbedder` → 时间嵌入向量
2. 条件序列 `cond_seq` → `ConditionPool`（均值池化 + MLP）→ 条件向量
3. 两者相加 → `cond_vec`
4. `cond_vec` 通过 FiLM 调制注入每个 ResBlock

**CFG 支持**:
- `null_cond_token` — 可学习的空条件参数，形状 (1, max_cond_len, cond_emb)
- `forward_with_cfg()` — 同时计算有条件和无条件预测，按 CFG 公式合并

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `forward` | `(zt_tokens, t, cond_seq, return_flat, force_uncond) -> tokens or flat` | 标准前向传播 |
| `forward_with_cfg` | `(zt_tokens, t, cond_seq, cfg_scale, return_flat) -> tokens or flat` | 带 CFG 的前向传播 |
| `build_cond_vector` | `(t, cond_seq, force_uncond) -> cond_vec` | 构建条件向量 |
| `get_null_cond` | `(batch_size, cond_len, device) -> null_cond` | 获取空条件 token |

---

### 4.6 train.py — 主训练脚本

**文件路径**: `source_code/train.py`（548 行）

**职责**: 完整的两阶段训练管线（IRS + 随机相位场景），包括 VAE 训练、潜在空间统计、LDM 训练和条件采样推理。

#### 4.6.1 DDPMScheduler

```python
class DDPMScheduler:
    def __init__(self, T=400, beta_start=1e-4, beta_end=2e-2, device="cuda")
```

DDPM 扩散调度器。

| 属性 | 说明 |
|------|------|
| `betas` | 线性噪声调度，形状 (T,) |
| `alphas` | `1 - betas` |
| `abar` | 累积乘积 `∏alphas` |
| `sqrt_abar` | `√abar`，用于前向加噪 |
| `sqrt_1m_abar` | `√(1-abar)`，用于前向加噪 |

**`q_sample(z0, t, noise)`**: 前向加噪 `zₜ = √ᾱₜ·z₀ + √(1-ᾱₜ)·ε`

#### 4.6.2 损失函数

##### `chamfer_distance_loss(p1, p2) -> torch.Tensor`
Chamfer 距离损失，衡量两个点云集之间的双向最近点距离。
- **输入**: `p1`, `p2` — 形状 [B, N, 3]
- **返回**: 标量损失值

##### `point_vae_loss(x, reconstructed_x, mu, logvar, kl_weight=1e-6) -> (total, cd, kl)`
PointVAE 总损失。
- **组成**: `total = CD_loss + kl_weight × KL_loss`
- **CD_loss**: Chamfer 距离
- **KL_loss**: `-0.5 × mean(1 + logvar - μ² - exp(logvar))`

#### 4.6.3 训练函数

##### `train_PointVAE(vae, train_loader, test_loader, device, epochs, lr, kl_weight, save_dir)`
VAE 训练循环。
- **优化器**: Adam
- **保存策略**: 最佳模型 + 每 50 epoch 检查点 + 最终模型
- **输出**: `vae_best.pth`, `vae_epoch_N.pth`, `vae_latest.pth`, `vae_history.npy`, `vae_loss_curve.png`

##### `estimate_latent_stats(vae, loader, device, max_batches=200) -> (z_mean, z_std)`
估计潜在空间的全局均值和标准差。
- **返回**: 标量 `z_mean` 和 `z_std`（全局统计）

##### `train_1D_DDPM(vae, condenc, epsnet, sched, train_loader, test_loader, z_mean, z_std, device, epochs, lr_cond, lr_eps, cond_drop_prob, save_dir)`
1D 潜在扩散模型训练循环。
- **VAE 冻结**: 训练时 VAE 参数不更新
- **Classifier-Free Guidance**: 以 `cond_drop_prob` 概率随机置零条件
- **优化器**: Adam，条件编码器和噪声网络分别设置学习率
- **保存策略**: 最佳模型 + 每 50 epoch 检查点 + 最终模型

##### `sample_conditional_1D(vae, condenc, epsnet, sched, cond, z_mean, z_std, device, cfg_scale=2.0) -> torch.Tensor`
条件采样，从噪声逐步去噪生成点云。
- **CFG 公式**: `ε̂ = ε̂_uncond + cfg_scale × (ε̂_cond - ε̂_uncond)`
- **去噪**: 从 T 到 0 逐步去噪，每步计算均值并添加噪声（t>0 时）
- **返回**: 重建点云，形状 (B, num_points, 3)

#### 4.6.4 main() 主流程

```
1. 预计算物理信道 (precompute_channels)
2. 构造训练/测试数据集 (ROIPairedDataset)
3. 实例化模型 (PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn, DDPMScheduler)
4. 训练 PointVAE (500 epochs)
5. VAE 重建核对（保存 GT 与重建点云）
6. 统计潜在空间分布 (estimate_latent_stats)
7. 训练 LDM/DDPM (500 epochs)
8. 条件推理测试（保存 GT 与生成点云）
```

**默认参数**:
- `train_data=6`, `test_data=1600`, `batch_size=32`, `num_points=2048`
- VAE: `z_dim=256`, `lr=1e-3`, `kl_weight=1e-6`
- LDM: `cond_emb=256`, `hidden_size=256`, `depth=4`, `num_heads=8`, `lr_cond=1e-3`, `lr_eps=1e-4`
- DDPM: `T=1000`, `cfg_scale=2.0`

---

### 4.7 train_no.py — 无 IRS 训练脚本

**文件路径**: `source_code/train_no.py`（745 行）

**职责**: 无 IRS 场景下的 LDM 训练，使用 `data_no_irs.py` 的数据集和信号模型。

#### 4.7.1 与 train.py 的主要区别

| 特征 | train.py | train_no.py |
|------|----------|-------------|
| 数据模块 | data.py | data_no_irs.py |
| 数据集类 | ROIPairedDataset | ROILDMDataset |
| 信号模型 | 5 条路径（含 IRS） | 仅 BS-ROI-BS |
| VAE 训练 | 包含 | 不包含（加载预训练） |
| 条件维度 | 88/帧 | 128/帧 |
| 噪声网络 | LatentDiT1D_CrossAttn | LatentDiT_Token_CrossAttn (⚠️) |
| 潜在空间 | 1D 向量 (256) | Token 形式 (32×32) |
| q_sample 形状 | view(-1, 1) | view(-1, 1, 1) |

#### 4.7.2 独有组件

##### `CachedLatentDataset(Dataset)`
缓存 VAE 编码后的潜在表示，避免重复编码。
- **初始化时**: 遍历基础数据集，用 VAE 编码器将点云编码为 μ
- **返回**: `(mu, roi_raw_t, roi_occ_t, phase_init, X_fixed)`

##### `build_x_feature(X) -> torch.Tensor`
构造发射信号特征（32 维）。
- **输入**: 复数信号 X
- **输出**: `[normalize(real(X)), normalize(imag(X))]`

##### `build_y_phase_feature(Y, phase) -> torch.Tensor`
构造接收信号 + 相位特征（96 维）。
- **输出**: `[normalize(real(Y)), normalize(imag(Y)), sin(phase), cos(phase)]`
- **每帧总特征**: x_feat(32) + y_phase_feat(96) = 128 维

##### `rollout_physical_sequence(roi_occ, phase_init, pilot_matrix, H_dict, device) -> torch.Tensor`
展开物理信号序列（无 IRS 版本）。
- **输出**: 形状 [B, Tau, 128]
- **说明**: 每帧使用固定导频信号，IRS 相位设为全零（不参与信号生成）

##### `estimate_latent_stats_from_cached(loader, device, max_batches) -> (z_mean, z_std)`
从缓存的潜在表示中统计均值和标准差。
- **返回**: 逐元素统计，形状 (1, num_latent_tokens, token_dim)

##### `train_one_epoch_ldm(...)` / `eval_one_epoch_ldm(...)`
单轮训练/验证函数，每轮动态计算物理信号特征。

##### `sample_one_ldm(...) -> z0`
单样本采样，返回潜在表示 z₀。

##### `save_ldm_result(...)`
保存 LDM 推理结果，包括 GT/预测点云的 .npy 文件和 3D 对比图。

#### 4.7.3 main() 主流程

```
1. 预计算物理信道
2. 构建 ROILDMDataset（含遮挡模拟）
3. 加载预训练 VAE
4. 缓存训练/验证集的潜在表示 (CachedLatentDataset)
5. 实例化条件编码器和噪声网络
6. 训练 LDM (train_ldm)
7. 保存推理结果 (save_ldm_result)
```

**默认参数**:
- `total_samples=7000`, `train_ratio=0.9`, `batch_size=8`, `num_points=2048`, `epochs=500`
- VAE: `hidden_dim=512`, `token_dim=32`, `num_latent_tokens=32` (⚠️ 与 models.py 不一致)
- LDM: `cond_emb=256`, `hidden_size=256`, `depth=6`, `num_heads=8`

---

### 4.8 train_r.py — 随机 IRS 训练脚本

**文件路径**: `source_code/train_r.py`（753 行）

**职责**: 随机 IRS 相位场景下的 LDM 训练，使用 `data.py` 的数据集和信号模型。

#### 4.8.1 与 train_no.py 的区别

| 特征 | train_no.py | train_r.py |
|------|-------------|------------|
| 数据模块 | data_no_irs.py | data.py |
| 信号仿真 | `simulate_received_signal`（无 IRS） | `simulate_received_signal`（data.py 版本，含 IRS） |
| 相位策略 | 全零占位 | 每帧随机生成 `2π·rand()` |
| 条件特征 | x(32) + y_phase(96) = 128/帧 | x(32) + y_phase(96) = 128/帧 |
| 日志前缀 | `[LDM-NoIRS]` | `[LDM-RandomIRS]` |

#### 4.8.2 rollout_physical_sequence 的差异

**train_no.py 版本**:
```python
phase_t = torch.zeros(B, setup.IRS_total, device=device)  # 全零占位
```

**train_r.py 版本**:
```python
phase_t = 2.0 * torch.pi * torch.rand(B, setup.IRS_total, device=device)  # 每帧随机相位
```

#### 4.8.3 main() 配置差异

```python
# train_r.py 的配置
cfg = {
    "scheme": "random_irs",
    "irs_enabled": True,
    "phase_update": False,
    "phase_mode": "random_each_step",
    ...
}
```

---

## 5. 关键类与函数索引

### 5.1 setup.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `Power_SNR` | 函数 | `(sigma, snr) -> float` |
| `get_Channel` | 函数 | `(a, b) -> np.ndarray` |
| `make_irs_rotated` | 函数 | `() -> (np.ndarray, np.ndarray)` |
| `make_roi_grid` | 函数 | `() -> np.ndarray` |
| `precompute_channels` | 函数 | `(device) -> dict` |

### 5.2 data.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `generate_ROI` | 函数 | `() -> np.ndarray` |
| `extract_point_cloud_from_voxel` | 函数 | `(ROI_np, num_points, voxel_size) -> np.ndarray` |
| `data_progress_amp_phase` | 函数 | `(data_complex) -> torch.Tensor` |
| `calculate_value_ROI_simple` | 函数 | `(ROI_voxel, phase, X, H_dict, Power_sigma, device) -> torch.Tensor` |
| `ROIPairedDataset` | 类 | `(n_samples, H_dict, num_points, device)` |

### 5.3 data_no_irs.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `generate_ROI` | 函数 | `() -> np.ndarray` |
| `extract_point_cloud_from_voxel` | 函数 | `(ROI_np, num_points, voxel_size) -> np.ndarray` |
| `normalize_point_cloud_global` | 函数 | `(point_cloud, roi_length, voxel_size) -> np.ndarray` |
| `denormalize_point_cloud_global` | 函数 | `(point_cloud_norm, roi_length, voxel_size) -> np.ndarray` |
| `apply_occlusion_to_roi` | 函数 | `(ROI_raw, ld) -> np.ndarray` |
| `build_fixed_16qam_pilot_vector` | 函数 | `(num_tx, device, normalize_power) -> torch.Tensor` |
| `simulate_received_signal` | 函数 | `(ROI_voxel, phase, X, H_dict, device) -> torch.Tensor` |
| `ROIVAEDataset` | 类 | `(n_samples, num_points)` |
| `ROILDMDataset` | 类 | `(n_samples, H_dict, num_points, device)` |

### 5.4 models.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `PointVAE` | 类 | `(num_points=2048, z_dim=256)` |
| `AdvancedCondEncoder` | 类 | `(seq_len=8, input_size=114, hidden_size=128, out_emb=256)` |
| `TimestepEmbedder` | 类 | `(hidden_size, frequency_embedding_size=256)` |
| `DiTBlock_CrossAttn` | 类 | `(hidden_size, num_heads, mlp_ratio=4.0)` |
| `FinalLayer` | 类 | `(hidden_size, out_channels)` |
| `LatentDiT1D_CrossAttn` | 类 | `(z_dim=256, cond_emb=256, hidden_size=256, depth=4, num_heads=8)` |
| `modulate` | 函数 | `(x, shift, scale) -> Tensor` |

### 5.5 models_unet.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `TimestepEmbedder` | 类 | `(hidden_size, frequency_embedding_size=256)` |
| `ConditionPool` | 类 | `(cond_emb=256, out_dim=256)` |
| `FiLM` | 类 | `(cond_dim, num_channels)` |
| `ResBlock1D` | 类 | `(in_ch, out_ch, cond_dim, dropout=0.0)` |
| `SelfAttention1D` | 类 | `(channels, num_heads=4)` |
| `Downsample1D` | 类 | `(channels)` |
| `Upsample1D` | 类 | `(channels)` |
| `UNet1DLatent` | 类 | `(token_dim=32, num_latent_tokens=32, cond_emb=256, base_channels=128, channel_mults=(1,2,4), num_heads=4, dropout=0.0, max_cond_len=8)` |

### 5.6 train.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `DDPMScheduler` | 类 | `(T=400, beta_start=1e-4, beta_end=2e-2, device="cuda")` |
| `chamfer_distance_loss` | 函数 | `(p1, p2) -> Tensor` |
| `point_vae_loss` | 函数 | `(x, reconstructed_x, mu, logvar, kl_weight) -> (total, cd, kl)` |
| `save_loss_curve` | 函数 | `(train_losses, test_losses, title, save_path)` |
| `evaluate_vae` | 函数 | `(vae, loader, device, kl_weight) -> (loss, cd, kl)` |
| `train_PointVAE` | 函数 | `(vae, train_loader, test_loader, device, epochs, lr, kl_weight, save_dir)` |
| `estimate_latent_stats` | 函数 | `(vae, loader, device, max_batches) -> (z_mean, z_std)` |
| `evaluate_1D_DDPM` | 函数 | `(vae, condenc, epsnet, sched, loader, z_mean, z_std, device)` |
| `train_1D_DDPM` | 函数 | `(vae, condenc, epsnet, sched, train_loader, test_loader, z_mean, z_std, device, epochs, lr_cond, lr_eps, cond_drop_prob, save_dir)` |
| `sample_conditional_1D` | 函数 | `(vae, condenc, epsnet, sched, cond, z_mean, z_std, device, cfg_scale) -> Tensor` |
| `main` | 函数 | `(device, train_data, test_data, batch_size, num_points)` |

### 5.7 train_no.py / train_r.py

| 名称 | 类型 | 签名 |
|------|------|------|
| `DDPMScheduler` | 类 | `(T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda")` |
| `CachedLatentDataset` | 类 | `(base_dataset, vae, device, print_prefix)` |
| `build_x_feature` | 函数 | `(X) -> Tensor` |
| `build_y_phase_feature` | 函数 | `(Y, phase) -> Tensor` |
| `rollout_physical_sequence` | 函数 | `(roi_occ, phase_init, pilot_matrix, H_dict, device) -> Tensor` |
| `estimate_latent_stats_from_cached` | 函数 | `(loader, device, max_batches) -> (z_mean, z_std)` |
| `train_one_epoch_ldm` | 函数 | `(cond_encoder, eps_model, scheduler, loader, optimizer, z_mean, z_std, H_dict, device, cond_drop_prob)` |
| `eval_one_epoch_ldm` | 函数 | `(cond_encoder, eps_model, scheduler, loader, z_mean, z_std, H_dict, device)` |
| `sample_one_ldm` | 函数 | `(cond_encoder, eps_model, scheduler, roi_occ_t, phase_init, X_fixed, H_dict, z_mean, z_std, device, cfg_scale) -> Tensor` |
| `save_ldm_result` | 函数 | `(save_dir, vae, cond_encoder, eps_model, scheduler, dataset_raw, H_dict, z_mean, z_std, device, sample_idx, cfg_scale)` |
| `train_ldm` | 函数 | `(cond_encoder, eps_model, train_loader, val_loader, H_dict, device, epochs, lr_cond, lr_eps, cond_drop_prob, diffusion_steps, beta_start, beta_end, save_dir)` |
| `main` | 函数 | `(device, vae_ckpt_path, total_samples, train_ratio, batch_size, num_points, epochs, save_dir)` |

---

## 6. 数据流与训练管线

### 6.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据生成阶段                                 │
│                                                                     │
│  generate_ROI()          extract_point_cloud()                      │
│  [16×16×16] 体素   ────▶  [2048×3] 点云                            │
│       │                          │                                  │
│       │                     归一化到 [-1,1]                          │
│       │                          │                                  │
│       ▼                          ▼                                  │
│  ROI_voxel ──────▶ calculate_value_ROI_simple() ──▶ Y_t (复数信号) │
│       │                    或                               │      │
│       │           simulate_received_signal()                   │      │
│       │                    │                                    │      │
│       │                    ▼                                    │      │
│       │           data_progress_amp_phase()                     │      │
│       │           build_x_feature() / build_y_phase_feature()   │      │
│       │                    │                                    │      │
│       │                    ▼                                    │      │
│       │           条件特征 cond [Tau, 88/128]                    │      │
│       │                                                         │      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 1: VAE 训练                                 │
│                                                                     │
│  点云 X ──▶ [Encoder] ──▶ (μ, logvar) ──▶ 重参数化 ──▶ z          │
│                                                               │     │
│  z ──▶ [Decoder] ──▶ X̂                                           │
│                                                               │     │
│  Loss = ChamferDistance(X, X̂) + kl_weight × KL(μ, logvar)       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 2: LDM 训练                                 │
│                                                                     │
│  点云 X ──▶ [冻结 VAE Encoder] ──▶ μ ──▶ 标准化 ──▶ z₀           │
│                                                     │               │
│  条件 cond ──▶ [CondEncoder] ──▶ c_seq              │               │
│                                    │                │               │
│  z₀ + 噪声 ε ──▶ q_sample ──▶ zₜ  │                │               │
│                         │          │                │               │
│                         ▼          ▼                │               │
│                    [DiT/UNet](zₜ, t, c_seq) ──▶ ε̂  │               │
│                                    │                │               │
│                    Loss = MSE(ε̂, ε)                 │               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    推理阶段                                          │
│                                                                     │
│  随机噪声 zₜ ──▶ [DDPM 反向去噪 T 步] ──▶ z₀                     │
│                      │                                              │
│                 每步:                                                │
│                 ε̂_cond = Model(zₜ, t, c_seq)                       │
│                 ε̂_uncond = Model(zₜ, t, c_null)                    │
│                 ε̂ = ε̂_uncond + cfg_scale × (ε̂_cond - ε̂_uncond)    │
│                 μ = (zₜ - β/√(1-ᾱ) × ε̂) / √α                     │
│                 zₜ₋₁ = μ + √β × ε  (t > 0)                        │
│                                                                     │
│  z₀ ──▶ 标准化逆变换 ──▶ [VAE Decoder] ──▶ 点云 X̂                │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 三种场景对比

| 维度 | train.py | train_no.py | train_r.py |
|------|----------|-------------|------------|
| **IRS 状态** | 启用 | 禁用 | 启用 |
| **相位策略** | 每帧随机 | 全零占位 | 每帧随机 |
| **信号路径** | 5 条（含 IRS 反射） | 1 条（BS-ROI-BS） | 5 条（含 IRS 反射） |
| **遮挡模拟** | 无 | 有 | 无 |
| **条件维度/帧** | 88 | 128 | 128 |
| **条件特征** | X(12)+Y(12)+IRS(64) | X(32)+Y+Phase(96) | X(32)+Y+Phase(96) |
| **潜在空间** | 1D 向量 (256) | Token (32×32) | Token (32×32) |
| **噪声网络** | LatentDiT1D_CrossAttn | LatentDiT_Token_CrossAttn ⚠️ | LatentDiT_Token_CrossAttn ⚠️ |
| **VAE 训练** | 包含 | 不包含（加载预训练） | 不包含（加载预训练） |
| **潜在缓存** | 无 | CachedLatentDataset | CachedLatentDataset |

---

## 7. 依赖关系

### 7.1 外部依赖

| 库 | 用途 | 使用模块 |
|----|------|----------|
| `torch` | 深度学习框架（模型定义、训练、推理） | 全部 |
| `torch.nn` | 神经网络模块 | models.py, models_unet.py |
| `torch.nn.functional` | 函数式 API（MSE、插值等） | train*.py |
| `torch.utils.data` | 数据集与数据加载器 | data*.py, train*.py |
| `numpy` | 数值计算（ROI 生成、坐标处理） | setup.py, data*.py |
| `scipy.spatial.transform` | 旋转矩阵计算（IRS 面板旋转） | setup.py |
| `matplotlib` | 可视化（损失曲线、3D 点云对比图） | train*.py |
| `math` | 数学函数 | data*.py, models.py |
| `random` | 随机数生成（物体放置） | data*.py |
| `os` | 文件系统操作 | train*.py |

### 7.2 内部模块依赖矩阵

```
                setup  data  data_no_irs  models  models_unet  train  train_no  train_r
setup             -     ×        ×          ×         ×         ×       ×        ×
data              ×     -        -          -         -         ×       -        ×
data_no_irs       ×     -        -          -         -         -       ×        -
models            -     -        -          -         -         ×       ×        ×
models_unet       -     -        -          -         -         -       -        -
train             ×     ×        -          ×         -         -       -        -
train_no          ×     -        ×          ×         -         -       -        -
train_r           ×     ×        -          ×         -         -       -        -
```

（× 表示依赖，- 表示不依赖）

### 7.3 安装依赖

项目未提供 `requirements.txt`，根据代码推断所需安装：

```bash
pip install torch torchvision  # PyTorch（需 CUDA 支持）
pip install numpy
pip install scipy
pip install matplotlib
```

---

## 8. 项目运行方式

### 8.1 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.12+（需 CUDA 支持）
- **GPU**: 推荐 NVIDIA GPU（CUDA），无 GPU 时自动降级到 CPU
- **内存**: 建议 16GB+（数据集预生成需要较大内存）

### 8.2 运行命令

#### 场景 1: IRS + 随机相位（完整两阶段训练）

```bash
cd source_code
python train.py
```

此脚本会依次执行 VAE 训练和 LDM 训练，输出到 `./model/` 和 `./outputs_npy/`。

#### 场景 2: 无 IRS（仅 LDM 训练，需预训练 VAE）

```bash
cd source_code
python train_no.py
```

需确保 `./outputs_vae/best.pth` 存在（预训练 VAE 权重），输出到 `./outputs_ldm_no_irs/`。

#### 场景 3: 随机 IRS（仅 LDM 训练，需预训练 VAE）

```bash
cd source_code
python train_r.py
```

需确保 `./outputs_vae/best.pth` 存在，输出到 `./outputs_ldm_random_irs/`。

### 8.3 可配置参数

#### train.py main() 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | "cuda" | 计算设备 |
| `train_data` | 6 | 训练样本数 |
| `test_data` | 1600 | 测试样本数 |
| `batch_size` | 32 | 批大小 |
| `num_points` | 2048 | 点云采样点数 |

#### train_no.py / train_r.py main() 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | "cuda" | 计算设备 |
| `vae_ckpt_path` | "./outputs_vae/best.pth" | VAE 权重路径 |
| `total_samples` | 7000 | 总样本数 |
| `train_ratio` | 0.9 | 训练集比例 |
| `batch_size` | 8 | 批大小 |
| `num_points` | 2048 | 点云采样点数 |
| `epochs` | 500 | 训练轮数 |
| `save_dir` | "./outputs_ldm_*" | 输出目录 |

### 8.4 输出文件说明

| 文件 | 说明 |
|------|------|
| `vae_best.pth` | VAE 最佳模型权重 |
| `vae_latest.pth` | VAE 最终模型权重 |
| `vae_history.npy` | VAE 训练历史 |
| `vae_loss_curve.png` | VAE 损失曲线图 |
| `condenc_best.pth` | 条件编码器最佳权重 |
| `epsnet_best.pth` | 噪声网络最佳权重 |
| `latent_stats.pth` | 潜在空间统计量（z_mean, z_std） |
| `ldm_history.npy` | LDM 训练历史 |
| `ldm_loss_curve.png` | LDM 损失曲线图 |
| `PC_gt_*.npy` | GT 点云（numpy 格式） |
| `PC_hat_*.npy` | 生成点云（numpy 格式） |
| `sample_*_compare.png` | GT vs 预测 3D 对比图 |

---

## 9. 已知问题与代码不一致性

### 9.1 缺失类定义

**问题**: `LatentDiT_Token_CrossAttn` 类在 `train_no.py`（第 16 行）和 `train_r.py`（第 16 行）中从 `models` 模块导入，但 `models.py` 中**未定义该类**。

```python
# train_no.py / train_r.py 中的导入
from models import (
    PointVAE,
    AdvancedCondEncoder,
    LatentDiT_Token_CrossAttn,  # ⚠️ models.py 中不存在
)
```

**影响**: `train_no.py` 和 `train_r.py` 无法直接运行，会抛出 `ImportError`。

**推测**: 该类可能是 `UNet1DLatent`（`models_unet.py`）的别名或替代实现，或者 `models.py` 中缺少该类的定义。

### 9.2 PointVAE 接口不一致

**问题**: `train_no.py` 和 `train_r.py` 中实例化 `PointVAE` 时使用了 `models.py` 中不存在的参数。

```python
# train_no.py / train_r.py 中的实例化
vae = PointVAE(
    num_points=num_points,
    hidden_dim=512,        # ⚠️ models.py 中无此参数
    token_dim=32,          # ⚠️ models.py 中无此参数
    num_latent_tokens=32,  # ⚠️ models.py 中无此参数
)

# models.py 中的定义
class PointVAE(nn.Module):
    def __init__(self, num_points=2048, z_dim=256):  # 仅 2 个参数
```

**影响**: 使用 `models.py` 中的 `PointVAE` 运行 `train_no.py` / `train_r.py` 会抛出 `TypeError`。

**推测**: `train_no.py` / `train_r.py` 使用了更新版本的 `PointVAE`（基于 token 的 VAE），该版本尚未同步到 `models.py`。

### 9.3 VAE decode 返回值不一致

**问题**: `train_no.py` 和 `train_r.py` 中 `vae.decode()` 的返回值被解包为元组。

```python
# train_no.py / train_r.py 中
recon_pred, _ = vae.decode(z_pred.unsqueeze(0))  # ⚠️ 期望返回元组

# models.py 中的定义
def decode(self, z):
    out = self.decoder(z)
    return out.view(-1, self.num_points, 3)  # 仅返回单个张量
```

**影响**: 使用 `models.py` 中的 `PointVAE` 会抛出 `ValueError`（无法解包非元组对象）。

### 9.4 DDPMScheduler 重复定义

**问题**: `DDPMScheduler` 类在 `train.py`、`train_no.py` 和 `train_r.py` 中各自独立定义，且 `q_sample` 方法的 reshape 方式不同。

```python
# train.py
a = self.sqrt_abar[t].view(-1, 1)    # 2D reshape

# train_no.py / train_r.py
a = self.sqrt_abar[t].view(-1, 1, 1) # 3D reshape
```

**影响**: 功能正确但代码冗余，建议提取为公共模块。

### 9.5 save_loss_curve 重复定义

**问题**: `save_loss_curve` 函数在 `train.py`、`train_no.py` 和 `train_r.py` 中各自独立定义，参数和实现略有差异。

### 9.6 H_dict 缺失键

**问题**: `data_no_irs.py` 中的 `simulate_received_signal` 函数使用了 `H_dict["H_ROI_BS"]`，但 `setup.py` 的 `precompute_channels` 函数未返回该键。

```python
# data_no_irs.py 第 220-221 行
H_ROI_BS = H_dict["H_ROI_BS"]   # ⚠️ precompute_channels 未返回此键
H_BS_ROI = H_dict["H_BS_ROI"]
```

`precompute_channels` 返回的 `H_dict` 中仅有 `H_BS_ROI`（形状 (4096, 4)），没有 `H_ROI_BS`。

**影响**: 运行 `train_no.py` 或 `train_r.py` 调用 `simulate_received_signal` 时会抛出 `KeyError`。

**推测**: `H_ROI_BS` 可能应为 `H_BS_ROI` 的共轭转置，或 `precompute_channels` 中遗漏了该信道的计算。

### 9.7 data.py 中 generate_ROI 与 data_no_irs.py 的差异

**问题**: 两个文件中的 `generate_ROI()` 函数定义了不同的物体模板，但函数名相同，可能导致混淆。

---

## 10. 公平实验对比框架（新增）

### 10.1 学术短板分析

原始项目的三组实验（train.py / train_no.py / train_r.py）存在两个 P0 级学术问题：

**P0-1：实验对比不公平** — train.py 使用完全不同的模型架构（LatentDiT1D + z_dim VAE），与 train_no.py/train_r.py 的 Token 模式架构不一致，导致无法判断 IRS 的效果差异来自 IRS 本身还是架构差异。

**P0-2：IRS 有效性未显著证明** — 仅 1 次训练的 CD=0.509 vs 0.526，效应量 Cohen's d ≈ 0.34（小效应），无统计显著性检验。

解决方案：统一架构 + 多轮独立训练 + 多重统计检验。

### 10.2 train_unified.py — 统一训练脚本

**路径**: `source_code/train_unified.py`

**功能**: 参数化统一训练脚本，支持 4 种 IRS 模式。

**命令行参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `--irs_mode` | none/zero/random/optimized | 必选，IRS 模式 |
| `--seed` | int (default=42) | 随机种子 |
| `--total_samples` | int (default=7000) | 总样本数 |
| `--epochs` | int (default=2) | 训练轮数 |
| `--batch_size` | int (default=16) | 批量大小 |
| `--vae_ckpt` | str | 预训练 VAE 权重路径 |
| `--occlusion` | flag | 启用 BS 遮挡模拟 |

**四种 IRS 模式**:
| 模式 | 信号路径 | 相位策略 |
|------|----------|----------|
| none | BS-ROI-BS (1条) | 全零占位 |
| zero | 全路径 (5条) | 全零 |
| random | 全路径 (5条) | 每帧随机 2π·U(0,1) |
| optimized | 全路径 (5条) | 坐标下降优化 |

**统一架构**: PointVAE(Token) + AdvancedCondEncoder(80dim) + LatentDiT_Token_CrossAttn

**使用示例**:
```bash
python train_unified.py --irs_mode none --seed 42 --epochs 500 --save_dir ./outputs_none
python train_unified.py --irs_mode random --seed 123 --epochs 500 --save_dir ./outputs_random
```

### 10.3 phase_optimizer.py — 相位优化模块

**路径**: `source_code/phase_optimizer.py`

**类**: `PhaseOptimizer`
- `optimize(ROI_voxel, X, n_irs_elements=32, n_iter=10)` → 返回最优相位向量
- 算法：坐标下降，逐元素扰动比较接收信号功率

**函数**: `compute_received_signal_irs(ROI_voxel, phase, X, H_dict, power_sigma, device)`
- 计算 5 条信号路径（含双 IRS 反射）的接收信号 Y

### 10.4 run_experiments.py — 批量实验调度

**路径**: `source_code/run_experiments.py`

**功能**: 按实验矩阵批量运行训练并收集 CD 指标。

**实验矩阵**: 4 种 irs_mode × 5 seeds = 最多 20 个实验

**特性**:
- 断点续跑：已完成的 seed+mode 组合自动跳过
- 自动 CD 评估：每个实验结束后在 10 个样本上计算 Chamfer Distance
- 输出 `experiment_results.json` 汇总所有实验数据

**使用示例**:
```bash
python run_experiments.py --total_samples 7000 --epochs 500 --vae_ckpt ./model/vae_best.pth --output_dir ./experiments
```

### 10.5 analyze_results.py — 统计分析与可视化

**路径**: `source_code/analyze_results.py`

**功能**: 对 experiment_results.json 进行统计检验和可视化。

**统计检验**:
- Bootstrap 95% 置信区间
- Independent t-test（vs NoIRS baseline）
- One-way ANOVA（四组对比）
- Cohen's d 效应量

**输出**:
- `report.md`: Markdown 格式统计报告
- `statistical_analysis.png`: 三面板图表（箱线图 + 置信区间图 + 效应量图）

**使用示例**:
```bash
python analyze_results.py --input ./experiments/experiment_results.json --output_dir ./analysis
```

---

*文档生成时间: 2025年*
*基于源码版本: source_code/ 目录下的 11 个 Python 文件*
