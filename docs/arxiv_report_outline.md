# arXiv Technical Report Outline（占位稿大纲）

> 目的：把 TECH_REPORT + space_isac_design 整理为可引用的 arXiv 预印本，
> 确立"扩散×ISAC-NTN 空白"的时间优先权占位，同时作为仓库的可引用锚点。
> 决策依据：2026-08-11 用户确认走"参考实现 + 技术报告占位"路线，不投期刊。
> 状态：大纲 v1（2026-09-03，随 TECH_REPORT v1.6 / Sionna 对照完成）。

---

## 0. 定位与元信息

- **arXiv 分类（主/副）**：`eess.SY`（系统工程，工程系统叙事主类）
  / `cs.IT`（信息论-通信）或 `cs.ET`；备选 `eess.SP`（信号处理，若强调感知算法）。
- **版本**：单版技术报告（technical report），后续按里程碑出 v2/v3
  （diffusion 重建集成后升级为大改版）。
- **许可**：arXiv 默认 + 仓库 MIT。
- **摘要页指向**：GitHub 仓库为 primary artifact；报告声明所有数字可
  一键复现（fixed seeds + CI）。

## 1. 标题候选（三选一，定稿前再挑）

1. *IRS-Diffu-ISAC: An Open-Source Physics-Grounded Engineering System for RIS-Aided Space ISAC (ISAC-NTN)* —— 主推：系统名前置，"open-source physics-grounded"是差异化。
2. *Toward Space ISAC with Reconfigurable Intelligent Surfaces: Real-Orbit Physics, Dynamic Phase Tracking, and a Reproducible Engineering Testbed* —— 强调物理与可复现。
3. *A Reproducible Reference Implementation for RIS-Aided LEO ISAC: Orbit Dynamics, Channel Validation, and Sensing–Communication Closed Loop* —— 强调"reference implementation"定位（与用户路线一致）。

## 2. 摘要骨架（~200 词）

1. 背景一句：ISAC 进入 3GPP Rel-20 规范轨道，但 ISAC 与 NTN 仍是两条独立轨道；扩散模型做 ISAC 环境重建已在地面/低空出现，太空场景空白（截至 2026-08 arXiv 检索无直接工作）。
2. 系统一句：开源工程系统 = SGP4 真实轨道 + 动态 RIS 相位跟踪 + 学习式感知 + 3D MOT + 感知-通信闭环 + SDR 接口，全部一键复现。
3. 数字三组：跟踪 +89%（K=1）/ K=8 退化；闭环 +309%（97.6% oracle）；双站破墙 0.31 m（38×）。
4. 方法学贡献两点：angle wall（物理上界 + 逃逸路线）；标准信道对照（CDL-D 验证两个近似的适用边界）。
5. 定位一句：作为 RIS×太空 ISAC 的社区参考实现，honest limitation 报告拉高可复现性门槛。

## 3. 章节结构（从 TECH_REPORT v1.6 映射，~14-16 页）

| arXiv 章节 | 来源 | 改写要点 |
|---|---|---|
| 1 Introduction（含 1.1 背景标准化时间线、1.2 文献定位） | TECH_REPORT §1 + .context 调研报告 | 1.2 压缩为 related-work 定位表：NSADM / Dai+AUGUST / RaLD / 4D-RaDiff 四簇 vs 本文（RIS + LEO + 真实轨道 + 开源）；引用锚点：扩散×未来网络综述 (2508.01586)、NTN ISAC 设计原则 (2604.11593)、3GPP Rel-19/20 时间线 |
| 2 System and Signal Model | TECH_REPORT §2 | 补 isac_sim 分层图（L0/L1/L2.1-Sionna CDL 对照层） |
| 3 Dynamic RIS Tracking & Closed Loop | TECH_REPORT §3 | 保留 K-sweep 主线（重配置率 vs 相干时间），对接 2026 文献（波束训练开销 2607.24003、1-bit 相位 2608.04133） |
| 4 Wideband Sensing & 3D MOT | TECH_REPORT §4 | 压缩，MOT 细节可移附录 |
| 5 Physical Findings | TECH_REPORT §5 + §6.4 | **angle wall 单独成章**：解析上界 + 双站三边定位逃逸（0.31 m / 38× / 秩亏警告）——这是报告最独特的可迁移发现，放正文不缩水 |
| 6 Standard-Channel Cross-Validation | TECH_REPORT §6.5（v1.6 新增） | Sionna CDL-D 对照三结论（|ρ(1GHz)|=0.888 LOS 地板；NLOS 去相关 ~10¹⁻⁵ s ≪ 1 s 帧；标准 K=8.98 dB 确认 K-sweep）——回应"无标准信道模型"审稿风险 |
| 7 Reproducibility & Open-Source Artifacts | TECH_REPORT §6.1-6.2 | 一键命令表 + CI + Colab；强调 seeded 复现 |
| 8 Limitations and Honest Discussion | TECH_REPORT §7 | 全部保留（这是卖点不是弱点） |
| 9 Conclusion + Roadmap | TECH_REPORT §8 + README roadmap | Roadmap：OTFS/AFDM、flow matching 基线、多星协同、3.811 NTN 剖面、OTA 采集 |
| 附录 A 复现环境 / B MOT 细节 / C 参数表 | TECH_REPORT | — |

## 4. 图表计划（arXiv 版 ~8 图 4 表）

- 图 1 系统架构（README Mermaid 转 PDF）
- 图 2 SGP4 过境几何 + 多普勒 S 曲线（复用 verify_sat 出图）
- 图 3 K-sweep 主结果（自由空间 + Rician 档位 + **CDL-D 标准 K 档**三合一）
- 图 4 angle-wall heatmap + 双站反例（复用 plot_angle_wall_scan）
- 图 5 双站三边定位 RMSE 曲面/表格图
- 图 6 Sionna 对照双面板（频率相关 + 时间自相关，复用 sionna_channel_comparison.png）
- 图 7 闭环 demo 帧 / MOT 3D 渲染
- 图 8（可选）SDR 管线保真度
- 表 1 related work 对比（四簇竞品 × RIS/LEO/真实轨道/开源四维度）
- 表 2-4 主结果汇总、双站扫描、复现命令

## 5. 差异化与占位叙事（写作红线）

1. **空白陈述用检索证据措辞**："no direct work on diffusion-based reconstruction for ISAC-NTN/LEO found on arXiv as of 2026-08"（附检索关键词与日期），不断言绝对不存在。
2. **三个单作者预印本主动区分**（NSADM 2511.19044；Dai 2603.29822 + AUGUST 2607.14778）：他们 = UAV/地面电磁点云 + 位置/SNR 引导、无 RIS、无轨道动力学；本文 = RIS 辅助 + LEO 真实轨道 + 动态相位跟踪 + 闭环 + 开源。related work 表格里显式分行。
3. **本报告 v1 不含 diffusion 实验正文**——诚实处理：标题与摘要只承诺"engineering system + physics + reference implementation"，diffusion 放 Roadmap（避免 overclaim）；仓库 legacy smoke test 可作为附录脚注。若审稿环境追问，v2 集成扩散重建后升级。
4. **flow matching 一句带过**为 future baseline，不展开。
5. 引用规范：3GPP 时间线全部引学术综述转述（Liu 2025 / Yang 2026），不直接引内部文稿。

## 6. 投稿前 checklist

- [ ] 精读 NSADM / AUGUST / RaLD 方法细节，完成表 1 对比行（防撞车，调研报告 §5 待办第 1 项）
- [ ] 图 1-8 从现有产物导出 PDF（matplotlib 统一字体/尺寸）
- [ ] LaTeX 化：TECH_REPORT.tex 已存在，以它为底改造（arXiv 单栏 + natbib）
- [ ] 作者/单位确认；orcid；funding 声明（无则写 none）
- [ ] 复现抽查：在一台干净机器上按附录 A 从零跑通全部一键命令
- [ ] arXiv 提交元信息：license 选 CC BY 4.0（便于社区引用复用）
- [ ] 提交后 48h 内在 README badge 加 arXiv 链接 + Releases 打 tag（v1.3.0 ↔ report v1.6）

## 7. 时间线建议

| 步骤 | 产出 | 预估 |
|---|---|---|
| 竞品精读 + 表 1 | 对比表 + 差异化段定稿 | 1 天 |
| LaTeX 化 + 图表导出 | arXiv 源码包 | 1-2 天 |
| 内部全文审查（对照 honesty notes） | 终稿 | 0.5 天 |
| 提交 arXiv | 预印本编号 | 当日 |
