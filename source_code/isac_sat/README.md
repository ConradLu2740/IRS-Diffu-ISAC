# source_code/isac_sat/ — 星-地 ISAC 核心代码（active）

> **当前推荐入口**：本项目所有功能从这里运行。`../legacy/` 与 `../archive/` 为历史归档。

- 基于真实 TLE（SGP4）的 LEO 卫星轨道传播
- 星载 / 地面 RIS 动态相位优化与跟踪
- 感知-通信闭环（单目标 / 多目标 / 10 目标 MOT）
- 宽带 HRRP / ISAR 序列感知、SDR IQ 数据管线

## 快速开始

```bash
# 从仓库根目录执行（无需 cd 进本目录）
make setup    # 首次：装依赖（约 2-3 分钟）
make verify   # 1 分钟物理验证
make demo     # 感知-通信闭环 demo（自动训练 + 闭环）
```

> 数据与模型权重全部**程序内合成生成**，无需下载任何外部数据集。

## 脚本总览

### 物理层与数据生成（核心依赖）

| 脚本 | 用途 | 典型命令 | 产物 |
|------|------|----------|------|
| `setup_sat.py` | 星-地动态场景物理层：TLE/SGP4 轨道、远场信道、多普勒、时延 | （被其他脚本引用） | — |
| `data_sat.py` | 动态数据生成：5 路径信道、点云样本对、`SatROIDataset` | （被训练脚本引用） | 合成数据集 |
| `phase_optimizer_sat.py` | 动态 RIS 相位优化（random / tracked 策略对比） | （被引用） | 优化相位 |

### 训练

| 脚本 | 用途 | 典型命令 | 产物 |
|------|------|----------|------|
| `train_sensing.py` | 感知模型（6 类分类 + 定位），闭环 demo 用 | `python train_sensing.py --wideband --epochs 40` | `isac_demo/sensing_best.pth` |
| `train_sensing_multi.py` | 多目标感知（最多 K=2 目标分类 + 定位） | `python train_sensing_multi.py --wideband` | `isac_demo/sensing_multi_best.pth` |
| `train_detect.py` | 10 目标检测器（5 类分类 + 定位，宽带距离像输入） | `python train_detect.py --n_scenes 25 --epochs 50` | `isac_demo/detect_best.pth` |
| `train_sat.py` | 星-地 ISAC 扩散 3D 重建对比训练（none/sat/ground） | `python train_sat.py --irs_mode sat` | `sat_model/{mode}/` |
| `task_sat.py` | 感知任务升级：目标分类 + 姿态估计 | `python task_sat.py` | 评估输出 |
| `baseline_classic.py` | 经典基线对比：2D-CFAR + MUSIC vs 学习式感知 | `python baseline_classic.py` | 对比图 |

### 验证与评估

| 脚本 | 用途 | 典型命令 | 耗时 |
|------|------|----------|------|
| `verify_sat.py` | 物理正确性：轨道参数 / 多普勒 / 信道（ALL PASS 即正确） | `python verify_sat.py` | ~1 分钟 |
| `verify_tracking.py` | 动态 RIS 相位跟踪 vs random 基线 | `python verify_tracking.py` | ~1 分钟 |
| `verify_robustness.py` | 多轨道（ISS/Starlink）× Ka 频段鲁棒性 | `python verify_robustness.py` | ~1 分钟 |
| `eval_sat.py` | 扩散重建评估：CD / F-Score / Voxel IoU + 可视化 | `python eval_sat.py` | 分钟级 |

### 演示（面向读者 / 评审）

| 脚本 | 用途 | 典型命令 | 产物 |
|------|------|----------|------|
| `demo.py` | 感知-通信闭环 demo（单目标） | `python demo.py` | `isac_demo/demo_result.png` |
| `demo_live.py` | 多场景实时演示 HTML 播放器 | `python demo_live.py --n_scenes 3` | `isac_demo/demo_live.html` |
| `demo_multi.py` | 多目标感知-通信闭环 | `python demo_multi.py` | 结果图 |
| `demo_mot.py` | 10 目标 MOT 演示（检测+跟踪） | `python demo_mot.py` | 轨迹图 |
| `demo_mot_html.py` | 3D 交互式 MOT 演示（Plotly.js 单文件） | `python demo_mot_html.py` | `isac_demo/mot_3d.html` |
| `demo_sdr.py` | SDR 数据管线演示（无硬件，仿真 IQ 回放） | `python demo_sdr.py` | 回放感知结果 |
| `make_animation.py` | 生成演示 GIF（无需 ffmpeg） | `python make_animation.py` | `isac_demo/demo_animation.gif` |
| `run_demo.sh` | 一键闭环：自动训练 + 演示 | `bash run_demo.sh` | `isac_demo/demo_result.png` |

### SDR 接口（硬件预留）

| 脚本 | 用途 |
|------|------|
| `sdr_io.py` | IQ 数据格式与 IO：保存/加载/仿真 ISAC IQ |
| `sdr_ingest.py` | 时域 IQ → FFT → 宽带距离像（HRRP）→ 感知输入 |

### 多目标追踪（MOT）

| 脚本 | 用途 |
|------|------|
| `mot_data.py` | 移动目标场景生成（5 类目标、每帧更新、边界反弹） |
| `mot_tracker.py` | 检测-关联-追踪（匈牙利匹配 + 卡尔曼滤波） |

## 通用参数（跨脚本一致）

| 参数 | 含义 | 默认 |
|------|------|------|
| `--irs_mode` | RIS 模式：`none`（无 RIS）/ `sat`（星载）/ `ground`（地面） | `sat` |
| `--phase_mode` | RIS 相位：`random`（随机基线）/ `tracked`（动态跟踪） | `tracked` |
| `--tau` | 时间帧数（8 帧覆盖 ~60km 弧长） | `8` |
| `--snr_db` | 参考 SNR（dB） | `20.0` |
| `--bs_ant` / `--ue_ant` | 卫星 / 地面站天线数 | `4` / `4` |
| `--seed` | 随机种子（固定可复现） | 各脚本默认值 |

> 所有脚本均支持 `--help` 查看完整参数。

## 产物目录

`isac_demo/` — 训练 checkpoint（`*_best.pth`）、演示 HTML、GIF、PNG 结果图。

## 二次开发入口

- 改轨道 / 频段 / 目标模板 → 见顶层 `README.md` 的「如何改造成自己的场景」
- 参数速查与示例组合 → 见顶层 `configs/README.md`
- 经典基线对比 → `python baseline_classic.py`
