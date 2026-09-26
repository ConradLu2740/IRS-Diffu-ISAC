# FMShape 论文内部评审封面说明

**致**：张昱老师、卢为党老师（浙江工业大学信息工程学院）
**事由**：FMShape 投稿前内部评审（v0.2 英文润色稿，2026-09-26）
**依据**：`docs/paper/FMShape_draft_v0.1.md`（草稿 v0.2）、`TECH_REPORT.md` v1.50；细节以原文为准，此处仅作导航。

## 1. 论文定位（一句话）

FMShape 是空-天 ISAC 中由实测 HRRP 条件生成目标区域三维点云形状的条件流匹配框架：VAE 潜空间 + OT-CFM，单步采样。主投 IEEE TAES（EDICS: Sensing; Radar signal processing; Machine learning），备选 IEEE IoT-J / TSP。现状：七节正文 + Fig. 1–3 + Table I–IV + 19 条全部真实文献（含 2 篇 2025–2026 邻近工作，文献核查后补引）。

## 2. 外部评审 W1–W6 整改状态

| # | 问题 | 状态 | 落点 |
|---|---|---|---|
| W1 | 统计收口 | 部分完成 | 论文已统一为「采样效率对等」叙事，保留 3 种子配对结果（+4.5% / −20.5% / −4.4%，CI 跨零）；headline 多种子 CI 投稿前需补（短评阶段不阻塞） |
| W2 | 物理真实性边界 | 已补 | 电离层敏感性段（VI-H）：Ka 波段 2–20 m vs 0.15 m 距离分辨，延迟标定前提 |
| W3 | 算法贡献升维 | 窄切片形态规避 | 无需拆篇 |
| W4 | 文档形态 | 已统一 | 论文与 TECH_REPORT 活文档分离；TECH_REPORT v1.50：CA 可达值 = 下界、ELBO 正名、±611 kHz 闭式、FM/DDPM parity |
| W5 | 生成指标偏窄 | 已补 | coverage / 1-NNA / MMD-CD（VI-G Table IV，n=32）：一步图坍缩签名（1-NNA=1.000、内部间距 0.028 vs 到参考 0.291）与十步采样分散性（coverage 0.812/0.906、1-NNA 0.906）均已量化，n=8 试点保留在证据中 |
| W6 | 用语不严谨 | 随 W4 修正 | 同批术语与 headline 口径修订 |

整改依据：`bd4f22f`（W1/W4/W6）、`1cf63d1`（W5）、`a37d436`（venue 定稿）。

## 3. 三条诚实性红线（老师可能会问，先给答案）

1. 质量主张仅为「采样效率对等」，非优越；
2. +6.9% ISAR 增益为单种子结果，已带脚注；
3. CFG 下一个 ODE step = 2 次前向，已注明。

## 4. 请两位老师重点把关

- 第三节系统模型的表述习惯：是否与团队既有论文一致；
- 第六节实验的组织方式：叙事顺序与表格呈现是否清晰。

## 5. 待老师提供

作者姓名顺序、单位与通讯作者、基金项目号与致谢措辞——提供后即填入论文首页占位符。

## 6. 一句话总评

方法论严谨、负结果与证书齐备；主要待补是多种子统计收口与实测数据验证（后者为 TAES 常见拒稿点，cover letter 已备缓解说辞）。

## 7. Cover letter 草稿（AI disclosure + 仿真定位，2026-09-26）

> In preparing the manuscript and the accompanying open-source platform, generative AI tools were used for code, experiment scaffolding, drafting, and language editing; all such content has been reviewed and verified by the authors, who take full responsibility for the accuracy of the work.
>
> All reported results are simulation-based on a physics-grounded, fully reproducible open-source platform (real SGP4 ephemerides; channel-fidelity cross-checks against a standard 3GPP CDL profile are carried in the released code rather than in the manuscript). The manuscript claims no measured data, and over-the-air validation is identified as follow-up work.

注：(i) AI 措辞按 IEEE 现行口径写（披露 + 作者对准确性负全责），不写"协议均由作者设计执行"——仓库 commit 历史显示脚手架由 AI 深度参与，披露必须与事实一致；投稿前按 TAES 当期政策核对披露位置（cover letter vs 文末致谢）。(ii) Sionna/CDL 对照只指到 released code（论文正文无此内容）。(iii) 2048/200 sweep 整合后如归因措辞有变，第二段无需改动。
