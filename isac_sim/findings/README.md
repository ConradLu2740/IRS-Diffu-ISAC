# findings — 带解析界的结论模块

开源 ISAC 项目里少见的"带解析界的负结果"系列。每个 finding 包含：
解析推导 + CPU 可跑的配置扫描 + 一条 make 命令复现。

| Finding | 状态 | 入口 |
|---|---|---|
| 远场角度墙（单站交叉距离定位的物理上界） | ✅ 解析核心已入库，扫描/热力图 P2 | `far_field_angle_wall.py` |
| 重构速率 vs 相干时间（K-sweep 的信道保真度敏感性） | 计划中（依赖 channels/L1） | — |

## 远场角度墙（Finding 2）

单站（mono-static）星-地几何下，ROI 内 1 m 交叉距离偏移对应的角度张角
远小于实际阵列的 Rayleigh 分辨率，因此**角度信息物理上不足以支撑 ROI 内
交叉距离定位** —— 这是几何上界，不是实现缺陷。

解析核心（`far_field_angle_wall.py`）：
- `angular_extent(width_m, range_m)`：ROI 宽度在某斜距下张角（rad）
- `rayleigh_limit_rad(n_elements, wavelength_m, spacing_m)`：ULA Rayleigh 分辨率
- `required_array_aperture(...)`：反解"要分辨 ROI 需要多大孔径"

复现：`make finding-angle-wall`（热力图 + 双站反例出图，CPU 秒级）。

### P2 扫描结果（2026-09-02）

- **shortfall 热力图**（N × 斜距，80 m ROI @ 30 GHz）：默认场景（N=8, R=695 km）
  shortfall ≈ **1889×**；热力图显示 N 需达 ~15,000（孔径 77 m）才能分辨 ROI——
  星载平台物理上不现实，墙在全部实用配置下生效（`angle_wall_scan.png` 左图）。
- **双站反例**：利用场景中已有的地面 UE 作为第二测距源（基线 47.7 km），
  三边定位交叉距离误差 ≈ **2.19 m**（ρ=0.15 m, R=695 km），比单站角度墙
  （ML 实测 11.8 m）改善 **5.4×**——墙本身指出了逃生路线：现有 ISAC 场景
  天然是双站的，通信本端就是一个测距源。
- 含义："单站角度定位不可行"与"感知辅助通信闭环可行"并不矛盾——闭环
  依赖的是粗定位/分类，而高精度交叉距离定位应改用双站三边定位或 ISAR。
  这为后绕的（多站）感知算法实验定义了正确的目标问题。
