# 太空 ISAC 扩展（Phase 1）：星-地 ISAC 动态仿真

> 在原有 IRS-Diffu-ISAC（RIS 辅助 ISAC + 扩散模型 3D 点云重建）基础上的太空场景扩展。
> 目标：从室内近程理论仿真升级为面向太空通感一体化（ISAC-NTN）的星-地 ISAC 动态仿真。
> 更新日期：2026-08-05

---

## 1. 场景定义

**星-地 ISAC**：LEO 卫星作为基站（BS），对地面/近空目标区域（ROI）进行感知，同时服务地面站（UE）通信。

```
LEO卫星(BS, ~420km, 7.66km/s) ──→ 地面目标(ROI) ──→ 地面站(UE)
          │                              │
          └── RIS：星载(10m面板) 或 地面(1m面板) ─┘
```

三种 RIS 部署模式（对比实验）：
- `none`：无 RIS（基线）
- `sat`：星载 RIS（挂在卫星上，10m 量级面板）
- `ground`：地面 RIS（地面站旁，1m 量级面板）

## 2. 核心物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 载频 | 30 GHz（λ=1cm） | 毫米波 |
| 卫星 | ISS (TLE 25544) | 高度 ~420km，速度 7.66 km/s，周期 92.9 min |
| 目标区域 | 30°N, 120°E | 可配置 |
| 地面站 | 30°N, 119.5°E | 可配置 |
| 帧 | 8 帧 × 1s | 过境窗口中心附近 |
| ROI | 16³ 体素 × 5m | 地面目标区 80m |
| 多普勒 | -610 ~ +610 kHz | LEO @30GHz 真实量级 |
| 时延 | ~2.5 ms | 星地往返 |
| SNR | 20 dB | 功率标定 |

## 3. 模块架构

```
setup_sat.py   物理层：SGP4 轨道传播（ECI→ECEF）、动态几何、远场信道、多普勒/时延
data_sat.py    数据层：每帧 5 路径信道预计算、目标点云生成、条件特征扩展
train_sat.py   训练层：三模式对比训练（复用 models.py + train.py）
eval_sat.py    评估层：CD / F-Score / Voxel IoU + 可视化
verify_sat.py  物理验证：轨道参数、过境动力学、信道公式（可重复运行）
```

### 3.1 setup_sat.py — 物理层
- `Satrec.twoline2rv(TLE)` 加载真实卫星轨道，SGP4 传播
- ECI→ECEF 转换（含 GMST 恒星时、地球自转速度修正）
- `find_overpass_windows()`：搜索仰角 > 阈值的过境窗口
- `SatISACScenario.build_frames()`：生成 8 帧动态几何（卫星/星载RIS/地面RIS/目标/地面站）
- `get_channel_far()`：远场信道 `√0.1·exp(j2πd/λ)/d` + 多普勒相位 `e^{j2πf_d·t}`
- **多普勒相位用相对帧 0 时间累积**（绝对时间会导致数值失效——已修复的坑）

### 3.2 data_sat.py — 数据层
- `SatScenarioChannels`：每帧预计算 5 路径信道矩阵（BS→ROI→UE 结构与原 setup.py 对齐）
- `calculate_value_sat()`：接收信号 = 直达 + IRS 反射 + IRS 前向（多普勒注入）
- 条件特征每帧 `[12(X幅值/相位) + 12(Y幅值/相位) + 32(IRS相位) + 4(多普勒/时延/距离/仰角)]`
  - `none`: 28 维，`sat`/`ground`: 60 维
- 功率标定：以中心帧参考链路 SNR=20dB

### 3.3 train_sat.py — 训练
- 复用 PointVAE + AdvancedCondEncoder + LatentDiT1D_CrossAttn（零架构改动）
- 两阶段：VAE → LDM（与 train.py 相同管线）
- **VAE KL 修复**：默认 kl_weight 1e-6 → 1e-4，支持 kl_warmup_epochs 退火（防后验坍缩）

### 3.4 eval_sat.py — 评估
- CD（Chamfer Distance）：基础重建误差
- F-Score（τ=0.1/0.2）：点云结构正确性
- Voxel IoU（16³）：体素占用一致率
- 输出：GT vs 预测 3D 对比图、CD 箱线图、eval_metrics.json

## 4. 用法

```bash
# 物理验证（轨道/多普勒/信道，~1 min）
python verify_sat.py

# 数据 smoke
python data_sat.py

# 训练（三模式对比）
python train_sat.py --modes none sat ground --train_data 32 --test_data 8 \
    --vae_epochs 12 --ldm_epochs 10 --T 100 --kl_weight 1e-4 --kl_warmup 6

# 评估
python eval_sat.py --modes none sat ground --save_dir ./sat_model
```

## 5. Phase 2：动态 RIS 相位跟踪（已完成）

`phase_optimizer_sat.py` — 动态几何下的 RIS 相位优化（解析相位对齐，闭式解），支持：
- `random`：随机相位（基线）
- `track_K=1`：逐帧理想跟踪（每帧重新优化相位）
- `track_K=2/4/8`：分段跟踪（每 K 帧更新一次，模拟 RIS 重构速率限制）

### 5.1 关键结果（verify_tracking.py）

| 策略 | 平均接收功率相对随机提升 |
|------|------------------------|
| 理想跟踪（逐帧） | **+283.4%** |
| 分段 K=2 | +227.9% |
| 分段 K=4 | +132.9% |
| 分段 K=8 | +4.4% |

**核心洞察**：RIS 重构速率受限时相位会"过期"——K=8（8 秒更新一次）时增益几乎消失。
这是"RIS 重构速率 vs 动态信道相干时间"权衡的论文级结果。

### 5.2 训练集成（data_sat.py phase_mode）
- `phase_mode='random'`：每帧随机相位（基线）
- `phase_mode='tracked'`：每帧解析优化相位（动态跟踪）
- 对比（sat 模式，12 VAE + 10 LDM epochs）：random CD=0.151 vs **tracked CD=0.139**

### 5.3 物理模型改进
- 新增 `IRS_GAIN`（每元素增益，默认 100≈20dB）：大孔径 RIS 的面板孔径增益补偿路径损耗
- 修复前：IRS 路径功率仅为直达的 0.04%（相位优化无意义）
- 修复后：IRS 路径主导（~876%），相位优化效果显著

## 6. 多轨道 / Ka 频段鲁棒性（已完成）

`verify_robustness.py` — 轨道（ISS / Starlink）× 频段（30GHz / 28GHz）4 组合验证。

| 轨道 | 频段 | 高度 km | 速度 km/s | 多普勒 kHz | 时延 ms |
|------|------|--------|----------|-----------|--------|
| ISS | 30 GHz | 418.4 | 7.659 | [-506.6, 517.4] | 2.48 |
| ISS | 28 GHz | 418.4 | 7.659 | [-472.8, 482.9] | 2.48 |
| Starlink | 30 GHz | 389.1 | 7.675 | [-556.6, 576.3] | 2.08 |
| Starlink | 28 GHz | 389.1 | 7.675 | [-519.5, 537.9] | 2.08 |

- 物理一致性验证：多普勒随频率线性缩放（30→28GHz 下降 28/30 倍）；低轨 Starlink 时延更小
- 训练级鲁棒性：Starlink + 28GHz + tracked + 地面目标 → CD=0.183（管线稳定）
- 地面目标模板：car / uav / building / tank / tower / cubesat 6 类（`generate_ground_roi`）

## 7. 已验证结果（小规模 smoke）

- 轨道：高度 418km / 速度 7.66 km/s / 周期 92.88 min（与 ISS 真实值吻合）
- 过境动力学：仰角单峰（12°→33°→12°）、距离 U 型、多普勒 S 型曲线（-610→+610 kHz）
- 信道公式三项验证 PASS：幅度 √0.1/d、相位 2πd/λ、多普勒相位注入
- 三模式训练管线全通；KL 修复后 VAE CD 稳定下降、潜在空间利用充分
- 12 VAE + 10 LDM epochs（32 训练样本）：训练 CD none 0.233 / **sat 0.137** / ground 0.169
  （注：小样本评估有波动，正式对比需更大训练集与更多评估样本）

## 8. 已知限制与下一步

- [ ] 评估样本太少（8 个），指标波动大 → 需统一评估协议（50-100 评估样本）
- [ ] 小规模训练预测只恢复"位置和大致形状"，细节缺失 → 正式训练规模
- [ ] RIS 相位目前为随机相位 → **Phase 2：动态 RIS 相位跟踪**（分段优化，考虑 RIS 重构速率 vs 信道相干时间）
- [ ] 地面目标模板沿用室内 4 模板 → 需补充车辆/无人机等地面目标专用模板
- [ ] 轨道用单一 TLE（ISS）→ 可扩展 Starlink 等其他 LEO 轨道做鲁棒性实验
- [ ] 载频 30GHz → 可对比 Ka 28GHz / 星间链路频段

## 9. 感知任务升级：目标分类 + 姿态估计（已完成）

`task_sat.py` — 从点云重建升级到感知任务（通感一体：只用通信接收信号特征做感知）。

### 9.1 物理基础
- 类别 → RCS/形状差异 → 回波功率与多径模式不同
- 姿态 → 散射体分布变化 → 接收信号模式随姿态变化
- 微多普勒 → 运动部件调制回波（RATR 方法）
- 特征用 dB 尺度（通信标准），含 RCS 功率特征

### 9.2 结果（MLP 骨干，固定测试集 150 样本）
| 配置 | 分类 acc | 姿态 MAE |
|------|---------|---------|
| none + random | 0.620 | ~95° |
| sat + random | 0.573 | ~98° |
| sat + tracked | **0.620** | ~87° |

- 分类 acc 远超随机（0.167），证明从通信信号可做目标分类
- 混淆矩阵与 RCS 相关：building 88% / tank 76% / cubesat 69% 好；uav 35% / car 46% 差
- RIS 感知赋形：sat+tracked 略优于 sat+random（方向与功率增益一致，幅度有限）

### 9.3 物理结论（已验证）
1. **窄带观测下连续姿态估计不可行**（MAE≈90°=随机）——需宽带距离像/ISAR
2. **SNR 非分类瓶颈**：0dB→60dB 仅 0.35→0.40；信息瓶颈在观测结构（窄带+8帧+4天线）
3. 分类性能与目标 RCS 直接相关（物理规律）

### 9.4 踩坑记录
- 星-地场景功率标定系数 ~1e11 导致特征尺度爆炸（1e12 vs 1）→ dB 尺度修复
- 固定样本训练严重过拟合（400 样本 vs 200k 参数）→ 在线生成（数据增强）+ weight decay
- AdvancedCondEncoder（LSTM+Transformer）在小分类任务训练困难 → MLP 骨干更稳

### 9.5 感知增强：观测结构（天线/帧）
| 配置（sat+tracked） | 分类 acc |
|------|---------|
| 4 天线 + 8 帧（基线） | 0.620 |
| 16 天线 + 8 帧 | 0.747 |
| **16 天线 + 16 帧** | **0.847** |

物理结论：空间自由度（天线）与时间多样性（帧）都是感知信息源，
增加观测结构显著提升分类（0.62→0.85）。姿态仍不可行（需距离分辨率的宽带波形）。

### 9.6 宽带化：距离像（HRRP）突破姿态瓶颈（已完成）

`compute_range_profile`：宽带信号（1GHz/512 子载波）距离像，三维质心对齐到视线投影。

| 任务 | 窄带（4天线+8帧） | 宽带距离像 |
|------|-----------------|-----------|
| 分类 acc（多任务） | 0.383 | **0.867**（+128%） |
| 姿态 MAE（纯任务） | ~90°（不可行） | **29.1°**（可估！） |
| 姿态 MAE（多任务） | 91° | 89° |

物理结论：
1. **宽带距离像包含姿态信息**（HRRP 物理）：非对称目标（car）姿态敏感；
   对称目标（uav/cubesat）有姿态模糊（绕 z 旋转自相似）——物理规律
2. 距离像类别信息极强：固定位置无噪声可达 0.958；位置随机+噪声后 0.617-0.867
3. **多任务 trade-off**：分类与姿态竞争，纯任务均可达各自最优
4. 探针实验陷阱：近场几何（UE 2m）会产生虚假的可估性（0.88）——
   真实星-地是远场（UE 50km），必须用真实几何验证
5. 距离模糊：128 子载波@1GHz 距离窗 38m < ROI 80m → aliasing；需 ≥512 子载波

## 10. 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `setup_sat.py` | 轨道+动态几何+信道+多普勒/时延 | ✅ 物理验证 PASS |
| `data_sat.py` | 动态数据生成（none/sat/ground） | ✅ smoke 通过 |
| `train_sat.py` | 三模式对比训练 | ✅ smoke 通过 |
| `eval_sat.py` | CD/F-Score/IoU 评估+可视化 | ✅ 跑通 |
| `verify_sat.py` | 物理验证脚本 | ✅ ALL PASS |
| `smoke_test.py` | 原项目全链路最小复现 | ✅ 保留 |
| `sat_verify/` | 验证图（orbit_3d / overpass_dynamics） | ✅ |
| `sat_model/` | 训练产物（*.pth 被 gitignore） | 临时 |
