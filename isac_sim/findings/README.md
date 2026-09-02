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

复现：`make smoke-sim`（含 sanity 断言）；完整扫描热力图见路线图 P2。
