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

### 9.7 ISAR：多帧距离像（目标转动）

`compute_isar_sequence`：目标自转 ω，M 帧宽带距离像 → 距离-时间序列。

**分类（真实结果）**：ISAR 序列分类 acc **0.933**（最高；1D 距离像 0.867、窄带 0.383）

**姿态（重要修正——物理上界）**：
- 严格对照后发现：早期"姿态突破"（21-34°）均来自**探针实验的双倍旋转伪影**
  （探针把姿态角用了两次：roi 已含 ang 又加 theta0，制造虚假的姿态可估性）
- task_sat 的 ISAR 实现物理正确（与单倍旋转逐帧相关 1.000），真实姿态 MAE ≈ 90°
- **物理结论**：星-地远场 + 简单旋转对称模板 + 单站视线，绝对姿态基本不可估
  （需要非对称目标/ISAR 2D 成像/多站；8³ 模板对称性太强）
- 分类不受影响：姿态不可估不阻碍类别识别（类别由 RCS/形状总体决定）

**多任务优化**：独立编码器（分类/姿态分离）+ DualMLP 均保持分类 0.9+，
但姿态仍 ~90°——确认是物理上界而非架构问题。

### 9.8 探针实验教训（重要）
- 探针/验证脚本的几何必须与正式实现完全一致（近场/双倍旋转会产生虚假乐观）
- 任何"突破性"结果都要用独立实现交叉验证

## 10. 工程化 Demo（感知-通信闭环）

工程导向（非论文）交付：`source_code/` 下的可运行闭环系统。

| 文件 | 作用 |
|------|------|
| `train_sensing.py` | 单目标感知（分类+定位），宽带距离像，CPU 实时 |
| `train_sensing_multi.py` | 多目标感知（K=2 组分类+定位） |
| `demo.py` | 闭环 demo：感知→IRS 配置→通信增益（静态 4 图） |
| `demo_live.py` | 生成单文件 HTML 播放器（多场景切换+动画+UTC 时间轴） |
| `make_animation.py` | 生成 GIF 动画（matplotlib，无 ffmpeg 依赖） |
| `demo_multi.py` | 多目标闭环：检测多目标 + IRS 指向 |
| `run_demo.sh` | 一键启动 |

### 已验证结果
- 单目标闭环：感知分类 83%（30 次试验），IRS 感知辅助增益 +233%（oracle 达成率 98%）
- 多目标闭环：检测 2/2，IRS 指向增益 +289%（oracle 达成率 94%）
- 感知即使分类错，粗定位仍带来大部分通信增益（工程价值）
- 多目标分类难（0.24）：单站观测信号混合，物理限制（检测/定位可用）

### 演示形态
- `isac_demo/demo_live.html`：交互播放器（场景下拉/播放/速度/时间轴/UTC）
- `isac_demo/demo_animation.gif`：直接可看的动画
- 运行：`bash run_demo.sh` / `python demo_live.py` / `python make_animation.py`

## 11. 文件清单

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
| `mot_data.py` | 多目标场景数据生成（5 类 10 目标） | ✅ |
| `mot_tracker.py` | 帧间匹配 + 轨迹聚合（z 约束） | ✅ |
| `train_detect.py` | 检测器训练（距离像分类） | ✅ |
| `demo_mot.py` | MOT 2D 动画（PNG 帧） | ✅ |
| `demo_mot_html.py` | MOT 3D 交互 HTML（Plotly） | ✅ |
| `sdr_io.py` | SDR IQ 格式定义 + 读写 | ✅ |
| `sdr_ingest.py` | 导入管线（IQ→FFT→距离像） | ✅ |
| `demo_sdr.py` | SDR 无硬件回放 demo | ✅ |
| `baseline_classic.py` | 2D-CFAR + MUSIC 经典基线 vs ML 对比 | ✅ |

---

## 16. 经典感知基线：2D-CFAR + MUSIC vs ML（已完成）

**目的**：客观回答经典雷达信号处理在星-地 ISAC 场景的水平，与 ML 感知公平对比。

### 方法
- 2D-CA-CFAR（P_fa=1e-4，卷积向量化）在距离-多普勒图上检测（ISAR 序列慢时间 FFT）
- 单帧绝对距离像 1D-CFAR 检测区域能量质心 → 沿视线定位（bin→米回归标定，R²≈0.83）
- MUSIC（ULA 8 元 λ/2，64 快拍）目标方向测向
- ML：SensingMLP 输入**绝对距离像**（修正版特征，K=1024）vs 旧距离像（相对质心）
- 同测试集 60 样本，SNR 20dB，seed 固定

### 关键结果（2026-08-08，n_test=60）

| 方法 | 检测率 | 沿视线 RMSE | 横向 RMSE | 分类准确率 |
|------|--------|------------|-----------|-----------|
| **2D-CFAR**（经典，检测） | **1.000** | — | — | — |
| **1D-CFAR**（经典，定位） | — | **8.14 m** | —（无角度分辨力） | — |
| **ML（绝对距离特征）** | — | **2.27 m** | 11.84 m | 0.733 |
| **ML（相对质心=类别先验）** | — | — | 2D **22.63 m** | 0.817 |

MUSIC 目标方向测向 MAE **0.017°**（合成点源快照验证算法自洽；与 CFAR/ML 信号不同源）。
    ML（质心对齐形状特征）参照：2D RMSE 21.86 m / 分类 0.700。

### 重要发现（物理 + 工程）

1. **特征构造缺陷（工程 bug）**：`data_sat.compute_range_profile` 的 d_proj 相对**体素质心**
   （`rel = p - p_center`），目标在 ROI 中的**绝对位置信息在特征层被丢弃**——
   实测不同位置的单体素 RP 质心 bin 恒定。旧/形状特征 ML 定位≈类别先验（2D RMSE 22.6 m / 21.9 m）；
   改用绝对距离特征后 2D RMSE 降至 12.1 m（沿视线 2.3 m）。
2. **远场角度分辨物理墙**：斜距 695 km → 1 m 横向偏移仅 0.00008°；
   80 m ROI 全宽 0.0066° << ULA 8 元分辨力 22.5° → **角度对 ROI 内目标定位无信息**。
   横向定位只能靠类别先验/多普勒（ML 横向 RMSE 15.7 m 即此墙的量化）。
3. **经典方法完全可用**：2D-CFAR 检测率 100%，沿视线定位 RMSE 7.0 m（与 ML 同量级），
   无需训练即可部署——ML 优势在于沿视线精度略优（3.1 m vs 7.0 m）与分类能力。

### 修复建议（✅ 已完成 2026-08-08）
- `compute_range_profile` 新增 `center="roi"` 模式：d_proj 相对 ROI 中心（保留绝对位置，
  差分时延避免 K=512 回卷）；`SatROIDataset` 在 `rp_align=False` 时自动使用。
  修复后 ML 定位：2D RMSE 12.1 m（相对质心 22.6 m），沿视线 2.3 m，分类 0.80；
  闭环 demo 不受影响（感知增益 +309.4%，oracle 达成 97.6%）。
  **可复现性**：2026-08-08 起全部实验入口固定 seed=42（torch/np/python random），
  两次连续运行输出一致；旧数字（+283%/+233%/0.812 等）为修复前历史值，仅早期 commit 可复现。
- 横向定位如需突破：需多基地/阵列（超分辨受远场几何限制，收益有限）或时序先验（MOT）。

---

## 12. 多目标追踪 MOT（已完成）

**目标**：同时追踪 10 个移动目标（家庭轿车 / 无人机 / 自行车 / 行人 / 火车 5 类），跨帧保持 ID，输出轨迹。

### 架构

```
mot_data.py      场景：5 类目标 × 随机轨迹（直线/加速/转弯），多帧采样
                 每帧 5 路径信道 → 距离像（含多普勒），生成标注
mot_tracker.py   逐帧检测（滑动窗口峰值）→ 帧间匹配（距离/速度门限）
                 → 轨迹聚合（多数投票提升类别准确率）
train_detect.py  检测器：距离像 → 目标类别 + 位置（CNN 骨干，CPU 可训）
demo_mot.py      2D 动画（matplotlib 帧序列）
demo_mot_html.py 3D 交互 HTML（Plotly，旋转/缩放/悬停）
```

### 关键结果
- 10 目标 5 类：检测召回 **0.812**、类别准确率（原始）0.518 → 轨迹聚合后显著提升
- 训练：`train_detect.py --n_scenes 25 --epochs 50`（CPU 分钟级）
- 演示：`mot_animation.gif`（2D）+ `mot_3d.html`（交互 3D）

## 13. SDR 真实数据接口（已完成）

**目标**：为真实 SDR 硬件（RTL-SDR / USRP）预留数据通道，当前无硬件时用模拟 IQ 验证全链路。

### IQ 格式定义（sdr_io.py）
- `sample_isac.iq.npz`：`{iq: complex64[N,]，fs: float, fc: float, t0: float, meta: dict}`
- 兼容性：直接对接 RTL-SDR `rtlsdr` / USRP `uhd` 的 IQ 流

### 导入管线（sdr_ingest.py）
```
IQ 采样 → 下变频/滤波 → FFT（加窗）→ 距离像 → 感知（分类+定位）
```
- 保真度验证：模拟发射 → 导入重建距离像与理论相关 **0.998**
- 演示：`demo_sdr.py`（无硬件回放，输出距离像重建对比图）

## 14. 3D 多目标追踪（已完成）

**问题**：无人机在天上飞，2D 追踪不够——需要 3D 轨迹。

### 实现要点
- 场景数据升级：目标带 3D 坐标（无人机 z ∈ [80, 300]m，地面目标 z = 0）
- 检测器输出 (class, x, y, z)；z 由距离像（含俯仰信息近似）估计
- **物理约束修复**（关键）：地面目标（行人/轿车/自行车/火车）z 锁定贴地——
  单站距离像对高度信息弱，检测器 z 预测有噪声，追踪器对已知地面目标类别强制 z=0，
  消除虚假的 z 抖动（对应 commit `fix: ground-target z constraint in MOT tracking`）
- 交互演示：`mot_3d.html`（Plotly 3D 场景，按类别着色，轨迹连线）

## 15. 工程展示与 GitHub 推广（已完成）

工程导向的对外门面：

| 项 | 内容 |
|----|------|
| README | 英文主版（国际）+ 中文版（国内科研），含 Mermaid 架构图、双 Demo GIF、Roadmap |
| Colab | 60 秒一键体验（克隆→依赖→物理验证→闭环 demo→GIF） |
| CI | GitHub Actions：模块导入 + 物理验证 + SDR 管线 smoke |
| 模板 | Issue（bug/feature）+ PR 模板 |
| LICENSE | MIT |
| 演示产物 | demo_live.html / mot_3d.html / demo_animation.gif / mot_animation.gif |
