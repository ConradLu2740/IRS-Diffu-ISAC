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

### P2 扫描结果（2026-09-02，v2 修正几何）

- **shortfall 热力图**（N × 斜距，80 m ROI @ 30 GHz）：默认场景（N=8, R=695 km）
  shortfall ≈ **1889×**；分辨 ROI 需孔径 77 m（≈ N=15,394）——星载平台物理上
  不现实，墙在全部实用配置下生效（`angle_wall_scan.png` 左图）。
- **双站反例（实测，`verify_twostation_localization.py` / `make twostation`）：**
  用场景中已有的地面 UE 作第二测距源，真实过境几何下两站视线夹角
  γ ≈ 131°（BS 仰角 33.7°，UE 方位差 142.5°）。σ_ρ=0.15 m 时交叉距离
  RMSE **0.31 m**，比单站墙（ML 11.8 m）改善 **~38×**；
  破墙精度预算 σ_ρ < 11.84·sin γ ≈ 6.6~8.9 m（最差/默认几何）——
  CFAR 级测距精度即可破墙。
- **几何条件（重要）**：UE 必须在 BS-目标垂直面之外。Δaz→0 时两视线
  地面投影平行，2D 定位秩亏，误差与噪声同向爆炸（实测 1817 m）——
  与 γ 大小是两个独立的几何条件。
- 含义：单站 ML 的 11.8 m 不是"场景信息不足"，而是感知层只用了 BS 侧
  HRRP——破墙所需信息在双站 ISAC 场景中本来就存在，只是未被利用。
  后续感知算法的正确目标问题：**双站测距融合定位**（而非单站角度）。
