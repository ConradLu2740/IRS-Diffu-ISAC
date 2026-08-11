# configs/ — 参数速查与实验配方

> 本项目所有脚本均通过命令行参数控制（`python xxx.py --help` 查看完整列表）。
> 本目录不提供"配置文件加载"（保持脚本即参数、结果可复现），而是给出**常用参数组合**，
> 方便直接复制使用，也方便改造成自己的实验。

## 参数速查（跨脚本一致）

| 参数 | 含义 | 可选值 | 默认 | 影响 |
|------|------|--------|------|------|
| `--irs_mode` | RIS 模式 | `none` / `sat` / `ground` | `sat` | 是否用 RIS、RIS 部署位置 |
| `--phase_mode` | RIS 相位策略 | `random` / `tracked` | `tracked` | 动态跟踪 vs 随机基线 |
| `--tau` | 时间帧数 | 整数 | `8` | 观测时长（帧间隔 1s） |
| `--snr_db` | 参考 SNR | 浮点 | `20.0` | 噪声功率水平 |
| `--bs_ant` / `--ue_ant` | 卫星 / 地面站天线 | 整数 | `4` / `4` | 阵列规模 |
| `--wideband` | 是否用宽带距离像特征 | 开关 | 关 | 窄带 cond vs 宽带 HRRP |
| `--rp_align` | 距离像质心对齐 | 开关 | 关（保留位置） | 定位 vs 对齐权衡 |
| `--epochs` / `--lr` / `--batch_size` | 训练超参 | 数值 | 见各脚本 | 训练 |
| `--seed` | 随机种子 | 整数 | 各脚本默认 | 可复现性 |

## 实验配方（复制即用）

以下命令均在仓库根目录执行（假设已 `make setup`）。产物默认写入
`source_code/isac_sat/isac_demo/`。

### 配方 1：RIS 价值对比（核心结论复现）

无 RIS vs 星载 RIS vs 地面 RIS 的感知-通信闭环对比：

```bash
cd source_code/isac_sat
# 三种模式各训练一次（每个约 2-5 分钟 CPU）
for mode in none sat ground; do
  python train_sensing.py --irs_mode $mode --phase_mode tracked --wideband --epochs 40
  # 训练产物会覆盖 sensing_best.pth，跑完 demo 后记录结果再切换
  python demo.py --irs_mode $mode
done
```

> 注意：checkpoint 文件名固定为 `sensing_best.pth`，多模式对比时请先备份或重命名，
> 例如 `mv isac_demo/sensing_best.pth isac_demo/sensing_$mode.pth`。

### 配方 2：动态 RIS 相位跟踪价值

```bash
make track   # verify_tracking.py：random vs 分段跟踪的接收功率对比
```

### 配方 3：宽带 vs 窄带特征

```bash
cd source_code/isac_sat
python train_sensing.py --epochs 30                    # 窄带（快）
python train_sensing.py --wideband --epochs 30         # 宽带距离像（更精确）
```

### 配方 4：经典基线对比（2D-CFAR + MUSIC）

```bash
make baseline   # baseline_classic.py：经典信号处理 vs 学习式感知
```

### 配方 5：低 SNR 鲁棒性

```bash
cd source_code/isac_sat
python train_sensing.py --wideband --snr_db 10 --epochs 40
python demo.py --snr_db 10
```

### 配方 6：多目标场景

```bash
make demo-multi   # 多目标感知闭环（train_sensing_multi + demo_multi）
make mot          # 10 目标检测 + 跟踪 + 3D HTML
```

## 如何改造成自己的实验

| 想改什么 | 改哪里 | 说明 |
|----------|--------|------|
| 换卫星 / 轨道 | `setup_sat.py` 顶部 TLE 常量 | 当前内置 ISS (25544) 与 Starlink (44714)，可用任意 NORAD ID 替换 |
| 换频段 | `setup_sat.py` 的 `FC_HZ` | 默认 30 GHz 毫米波，Ka 频段验证见 `verify_robustness.py` |
| 换目标模板 | `data_sat.py` 的 `_template_*()` | 内置 car/uav/building/tank/tower/cubesat/bicycle/pedestrian/train |
| 换自己的模型 | 训练脚本中 `nn.Module` 定义 | 输入 cond 维度见 `channels.frame_cond_dim()` |
| 加干扰/杂波 | `data_sat.py` 信道生成 | 5 路径动态信道基础上扩展 |
| 改天线阵列 | `--bs_ant` / `--ue_ant` | 命令行即可 |

## 移植到新场景的检查清单

1. 跑通 `make verify`（物理层自检 ALL PASS）
2. 用 `make demo` 建立基线结果
3. 改参数后对比 `demo_result.png` / 训练日志中的 `cls acc` / `pos err`
4. 用 `--seed` 固定随机性，保证对比公平
