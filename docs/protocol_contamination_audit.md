# 协议污染审计备忘录 — FMShape 论文逐表"公共测试批"溯源

- **日期**: 2026-09-26
- **范围**: `docs/paper/FMShape_draft_v0.1.md` 中每一个承载数值的表格与头条论断的**批次构造方式（batch provenance）**
- **方法**: 纯只读代码审计（读脚本 + 读 JSON 证据），未运行任何训练 / 评估
- **背景 commit**: `665c595`（P_scale 配对重评）、`66998fc`（M3 配对重评）；本文档为审稿人要求的跟进而形成
- **不改动**: 论文草稿、任何源码脚本；本备忘录 + `TECH_REPORT.md` 局限性第 14 条为本次仅有的落盘

---

## A. 失败模式陈述（可直接抄入 limitations 的一段话）

本仓库的评估数据集 `SatROIDataset` 在**每次访问时现采样**：`generate_ground_target_sample` 消耗全局 `random`（`choice`/`uniform`/`randint`），`_frame_phases` 消耗全局 `torch.rand`，`calculate_value_sat` 加入 `torch.randn_like` 噪声（`source_code/isac_sat/data_sat.py:441-452, 639, 558`）。因此，任何"公共测试批（common test batch）"主张成立的前提是：**评估批必须在一次显式重播种之后、且在任何训练 RNG 消耗之前构造**。若脚本先训练（fresh-data 协议下每个 epoch 都从全局 RNG 抽新样本）再取测试批，则该批是训练后 RNG 状态的函数——不同单元格 / 不同运行 silently 得到不同批次，跨单元格或跨运行的数值差于是混入批噪声（batch noise），"同批比较"的表述 silently 失效。历史上三个跨运行比较（M1、M3、P_scale 逐单元格）即受此污染；修复标准是**配对重评（paired re-evaluation）**：每个种子一个固定构造、指纹记录的公共批，跨比较对象共享 `x0`。

---

## B. 建批方式分类口径

| 类 | 名称 | 定义 | 判定 |
|---|---|---|---|
| (a) | in-run shared batch | 脚本内所有被比较模型在同一进程、同一批次上评估（训练后取批亦可，只要被比双方共用该批张量） | SAFE |
| (b) | cross-run reseeded protocol | 不同运行各自在**建批之前**显式重播种（如 `eval_seed=999`、`seed=42`），使各运行得到同一测试批 | SAFE **仅当**重播种可证实先于建批（代码行号佐证；eval-only 脚本无训练消耗） |
| (c) | cross-run aggregation | 对两份（或更多）**独立产生**的 JSON 做差 / 相减，而各 JSON 的批次是各自独立抽取的 | CONTAMINATED |
| (d) | paired re-evaluation | 每个种子一个固定构造的公共批（指纹记录），被比单元格共享 `x0`（`X0_SEED_OFFSET=9000+seed`），逐云配对差 | SAFE（新标准） |

判定佐证机制（代码事实）：
- ROI 序列只消耗 **Python `random` 流**（`data_sat.py:441-452`），宽带 HRRP/ISAR 特征噪声按 `seed=idx` 独立取（`data_sat.py:703-714`，`RandomState(seed)`），`cond_feat` / `spread_equalize` 分支发生在**所有 RNG 消耗之后**（`data_sat.py:695-724`）——因此同参数重播种的各运行，其 GT 云序列一致；(b) 类"跨 run 同批"在代码上可达。
- 反之，`compare_gen.py` 在进程头重播种（`compare_gen.py:213-215`）但测试批在训练**之后**取（`compare_gen.py:156`）：同一次运行内 DDPM/FM/VAE 共用该批（a），但**跨运行 / 跨 mode 的批次不同**。

---

## C. 逐行审计表

### C.0 脚本级分类底表

| 脚本 | 建批相关代码 | 分类 |
|---|---|---|
| `compare_gen.py` | 重播种 `:213-215`（进程头）；训练 `run_mode :130-153`；取批 `:156`（训练后） | 批随训练后 RNG 状态：run 内 (a)；跨 run/mode 批次不同 |
| `verify_fm_train_scale.py` `run_cell` | 重播种 `:113-115`（单元格训练前）；训练 `:154-157`；取批 `:160`（训练后） | 每单元格各自批次 → 跨单元格 (c) 型污染 |
| `verify_fm_train_scale_paired.py` `build_common_batch` | 重播种 `:103-105` → 建数据集 → 取批 `:115`；指纹 sha1 `:118-121`；共享 x0 `:147-149`（`X0_SEED_OFFSET=9000`，`:64`） | (d) |
| `verify_m3_paired.py` | 复用 `build_common_batch`（导入 `:29-31`）；共享 x0 `:50-53` | (d) |
| `verify_gen_scale.py` / `verify_vae_scale.py` | 纯聚合器：读两份独立 `compare_gen.json`（`:24-32`）后做差（`:78-83`） | (c) |
| `verify_headline_multiseed.py` | 聚合 `sat_model_m1_{42,43,44}/compare_gen.json`（`:20-29`），逐种子差 `:60-70`，bootstrap CI `:32-36, :72-73` | 每种子行内 (a)；跨种子 CI/CV 跨三个不同批次（混合） |
| `verify_baselines_strong.py` | `build_eval_batch`（`verify_fm_bounds.py:51-53` 重播种先于 `:61` 建批）`:40, :156`；蒸馏 x0 固定 `:194-195` | (b)+(a) |
| `verify_fm_bounds.py` | 重播种 `:51-53` → 建批 `:61`；x0 固定 `:155-156` | (b) |
| `verify_gen_hardening.py` | 同用 `build_eval_batch` `:106`；Lipschitz 幂迭代自带 generator `:55` | (b) |
| `eval_gen_metrics.py` | `build_eval_batch` 重播种 `:60-62` → 建数据集 `:66` → 取批 `:71`（`eval_seed=999`）；采样 x0 `:209, :218, :230` | (b)+(a) |
| `train_fm_distill.py` | 评估前重播种 `eval_seed` `:133` → 建测试集 `:134` → 取批 `:139`；x0 `:145-146` | (b)（跨训练预算配对的关键设计，见 `:198` 参数注释） |
| `verify_fm_distill_diversity.py` | 重播种 `:87` → 建测试集 → 取批 `:121`；x0 `:133-134`；多样性噪声 `:67` | (b)+(a) |
| `verify_cond_shape_diversity.py` | 重播种 `:62-63` → 建数据集 `:68` → 逐样本 `:73-80` | (b) |
| `verify_info_audit.py` `block3_cfm` | 重播种 `:599` → 建数据集 `:603` → 取批 `:606`；配对抽样 generator `:656`；null A/B 的 `seed+999` 重播种 `:702-704` 仅用于**微调数据**，A/B 复测用同批同 generator `:730-731` | (b)+(a) |
| `verify_fm_shape_loop.py` `run_seed` | 重播种 `:131` → 建场景/数据集 → 取唯一 ROI `:142-144`；三策略（true/box/fm）同场景配对；8 seeds `:257` | (a) |
| `verify_cond_probe.py` `collect` | 重播种 `:56` → 建数据集 `:60` → 逐样本 `:65-66`；训练/留出为同一收集集的顺序切分 | (b)+(a) |
| `eval_sat.py` | 重播种 `:176-178`（进程头）→ 每 mode 建测试集 `:130-132` → 全 loader 评估 `:152` | run 内 (a)；多 mode 时后续 mode 批不同（该脚本不承载任何论文数值） |

### C.1 Table I — 等算力对比（4 行）

| 行 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| Spaceborne | 0.4055 / 0.3166 / −21.9% | `compare_gen.py` → `sat_model_cmp/compare_gen.json["sat"]`（三 mode 同进程一次运行，config 块逐 mode 一致可证） | (a)（`compare_gen.py:156` 行内 DDPM/FM/VAE 同批） | 否（行内 Δ 配对） | 安全（单种子观测；v1.50 scope note 已注明不得读作一般性 FM 优势） |
| Ground-RIS | 0.3069 / 0.2054 / −33.1% | 同上 `["ground"]` | (a) | 否 | 安全（同上） |
| No-RIS | 0.6037 / 0.4223 / −30.0% | 同上 `["none"]` | (a) | 否 | 安全（同上） |
| HRRP-conditioned | 0.2637 / 0.2269 / −14.0% | `compare_gen.py` → `sat_model_c1/compare_gen.json["sat"]`（另一次运行，config: 1024/200/100，与上三行 256/50/100 不同） | (a) | 否 | 安全；注意与上三行**不同 run、不同批**，论文未做跨行数值主张（仅逐行报告 Δ） |

> 跨 mode / 跨 run 备注：`compare_gen.py` 的批在训练后取（`:156`），故 sat/ground/none 三个 mode 的批次互不相同；c1 行又是另一次运行。凡跨行差值为 (c) 型——论文文本只做逐行 Δ，故不受影响。

### C.2 Table II — 强少步基线（common batch）

| 行 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| NFE=1 | DDIM 0.2583 / FM-Euler 0.2109 / student 0.1835 | `verify_baselines_strong.py` → `sat_model_cmp/strong_baselines.json` | (b)（`verify_fm_bounds.py:51-53` 重播种先于 `:61` 建批）+(a)（同进程 DDPM/DDIM/FM/student 同批；蒸馏训练发生在建批之后 `:156→:191`，不可能污染评估批） | 否 | 安全 |
| NFE=10 | DDIM 0.2256 / FM 0.3053 | 同上 | 同上 | 否 | 安全 |

### C.3 Table III — 蒸馏前沿（4 行）

| 行 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| Budget 512/60, K=2 | 0.0488 / 0.3839 / 0.19 | `train_fm_distill.py`（eval 前 `eval_seed` 重播种 `:133` 先于建批 `:139`）+ `verify_fm_distill_diversity.py`（`:87` 先于 `:121`）→ `sat_model_distill/{distill_result,distill_diversity_result}.json` | (b) | 否 | 安全 |
| Budget 1024/100, K=2 | 0.1127 / 0.5056 / 0.48 | 同上 → `sat_model_distill_full/` | (b) | 否 | 安全 |
| Teacher K=4 | 0.0640 / 0.3956 / 0.27 | 同上 → `sat_model_distill_k4/` | (b) | 否 | 安全 |
| Teacher K=10 | 0.0687 / 0.4368 / 0.29 | 同上 → `sat_model_distill_k10/` | (b) | 否 | 安全 |

> 实证佐证：四份 `distill_diversity_result.json` 的 teacher 侧数值**逐位一致**（`cd_t1=0.2318`、pairwise `0.0053`、GT-vs-postmean `0.2267`），即四次独立运行的评估批确实同一；(b) 类"重播种先于建批"在此获得数据级确认。loss 列（0.2535→0.0958）为训练侧标量，与批次无关。

### C.4 Table IV — 分布保真度（5 行）

| 行 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| FMShape NFE=1 | 0.625 / 0.875 / 1.000 / 0.308 | `eval_gen_metrics.py`（`eval_seed=999`，重播种 `:60-62` 先于建批 `:66→:71`；同进程评估全部五行）→ `sat_model_cmp/gen_metrics.json` | (b)+(a) | 否 | 安全（checkpoint 逐行记录于 JSON） |
| FMShape NFE=10 | 0.500 / 0.750 / 0.875 / 0.480 | 同上 | (b)+(a) | 否 | 安全 |
| DDPM NFE=100 | 0.625 / 0.750 / 1.000 / 0.379 | 同上 | (b)+(a) | 否 | 安全 |
| Distilled student | 0.375 / 0.875 / 1.000 / 0.391 | 同上 | (b)+(a) | 否 | 安全 |
| VAE oracle | 1.000 / 1.000 / 0.000 / 0.584 | 同上 | (b)+(a) | 否 | 安全 |

> 正文引用的 0.031（一步图内部 spread）、0.314、0.076、0.666、参考集自距 0.827 均可在同一 JSON 的 `pairwise_cd_gen_internal` / `mmd_cd_offdiag` / `reference.mmd_cd_ref_self` 字段对上，同批产物。

### C.5 Fig. 3(a)/(b)

| 位置 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| Fig. 3(a) | c1 的 FM NFE 曲线 + DDPM 100 点 + VAE 上界；NFE*≤2 交叉带 | `paper_figs/fig3_nfe_curve.py:18` 读 `sat_model_c1/compare_gen.json`（单次运行内共享批次） | (a) | 否 | 安全（该批上 FM1=0.2269 < DDPM 0.2637，NFE*=1，与"≤2"表述相容） |
| Fig. 3(b) | DDIM / FM-Euler / 1 步学生曲线 | 同脚本 `:19` 读 `sat_model_cmp/strong_baselines.json` | (b) | 否 | 安全 |

### C.6 Section VI-B 头条数字

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| 3-seed paired test | 逐种子差 +4.5% / −20.5% / −4.4%；bootstrap 95% CI 跨零；DDPM100 跨种子 CV 18% | `verify_headline_multiseed.py` → `isac_demo/headline_multiseed.json`（聚合 `sat_model_m1_{42,43,44}/compare_gen.json`，三次独立运行） | **混合**：每个种子的 FM1−DDPM100 在该 run 内同批配对（a）；跨种子的 CI/CV 把三个**不同批次**上的差放在一起统计（c 型聚合） | 部分：CI 方差被批噪声放大（保守方向） | 可辩护：论文据此只主张 parity 且 M1a/M1b=false 已如实记录（JSON `verdicts`）；若未来需要更强主张，需按 (d) 做逐种子公共批配对重评——**待办（非缺陷）** |
| crossover NFE* | sat/none ≤ 2；ground = 1 | `verify_fm_bounds.py` claim3（sat 批上 crossover=1，`sat_model_cmp/verify_fm_bounds.json`）+ Table I 各行行内比较 | (b)+(a) | 否 | 安全（论文表述为宽松上界） |
| 收敛阶 −0.87 | slope_euler | `verify_fm_bounds.py` → `sat_model_cmp/verify_fm_bounds.json` | (b) | 否 | 方向安全；**待办**：当前 JSON 记录 `slope_euler=−0.852`（n_eval=32），论文的 −0.87 源自早期运行（记录于 `docs/optimization_roadmap.md` §2.1），JSON 已被覆盖、无法从代码确定产生 −0.87 的那次运行的参数——需论文侧核对措辞（本次不改论文） |
| 直线性 0.006 | FM/DDPM 曲率比 0.00614 | 同 JSON claim2 | (b) | 否 | 安全 |
| Lipschitz 3.54 | L̂=3.5381 | `verify_gen_hardening.py` → `sat_model_cmp/verify_gen_hardening.json`（评估批同 `build_eval_batch`） | (b) | 否 | 安全 |
| student 0.1835 vs 0.2109 / 0.3839 vs 0.2664 | 跨批诚实记账 | Table II 批 / Table III 批 | 两个 (b) 批次 | 否 | 安全（论文已明示两个批次不同） |

### C.7 Section VI-C 头条数字

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| Δ(0)=0.302；21.8% 方差解释；恒等式残差 1.6e-7；Δ(t) 非递增 | 0.301944；0.302/1.3869=21.8%；1.64e-07 | `verify_info_audit.py` `block3_cfm`（重播种 `:599` 先于建批 `:606`）→ `isac_demo/info_audit_c1.json`（loads `sat_model_c1/sat`） | (b) | 否 | 安全 |
| condenc 未坍塌：输出 std 6.2×；shuffle 0.29 | 6.21；0.290 | 同 JSON `condition_sensitivity` | (b) | 否 | 安全 |
| 多样性比 0.02；pairwise 0.0053（16 噪声） | 0.024；0.0053 | `verify_fm_distill_diversity.py` → `sat_model_distill/distill_diversity_result.json` | (b)（eval_seed 999） | 否 | 安全 |
| guidance sweep 0.3060→0.2725→0.2664 | 同 JSON `cfg_sweep`（teacher cd_gt, w=0/1/2，同 x0=999 配对） | (b)+(a) | 否 | 安全 |
| 最近条件对 0.663 / 0.0265；ρ=−0.015 | 0.6634；0.0265；−0.0145（96 scenes / 4560 pairs） | `verify_cond_shape_diversity.py`（重播种 `:62-63` 先于建批 `:68`）→ `sat_model_c1/cond_shape_diversity.json` | (b) | 否 | 安全 |

### C.8 Section VI-D 头条数字（ISAR 增广）

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| Δ(0) raw / eq / matched-budget control | 0.131 / 0.154 / 0.144（推出 −9.0% / +17.6% / +6.9%） | `verify_info_audit.py` 三次独立运行 → `isac_demo/info_audit_c3_raw.json`（loads `sat_model_c3_raw/sat`）、`info_audit_c3_eq.json`（loads `sat_model_c3/sat`）、`info_audit.json`（C5 对照，loads `sat_model_hrrp512/sat`，512/60） | (b)：三次均 `seed=42` 重播种先于建批（`:599→:606`），协议字段逐位一致（`fm_seed=7, n_eval=64, bins=10`）；GT 云序列跨 run 一致（ROI 只消耗 Python random 流 `data_sat.py:441-452`；宽带噪声按 `seed=idx` `:703-714`；`cond_feat`/`spread_equalize` 在所有 RNG 消耗之后 `:695-724`）——即三次 Δ(0) 是在**同一 GT 批**上对条件处理的对比 | 否 | 安全（single-seed caveat 论文已载） |
| CD 0.4011 raw / 0.4065 eq（+1.3%） | `sat_model_c3_raw/compare_gen.json["sat"]` vs `sat_model_c3/compare_gen.json["sat"]`（两次独立 `compare_gen.py` 运行） | (c) 名义，但**批次一致性经验证**：两份 JSON 的 VAE oracle CD 逐位相同（`0.008261645213`）——同一 VAE + 同数据流 + 同 RNG 消耗顺序（`spread_equalize` 在所有 RNG 消耗之后生效，cond_dim 均为 544、架构相同）⇒ 两运行测试批同一 | 否（已验证） | 安全（附指纹证据）；记账：此类"跨 run 同批"今后必须附此类指纹佐证 |

### C.9 Section VI-E / V-D（蒸馏前沿正文）

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| V-D/VI-E 前沿段（loss 0.2535→0.0958；K=4→10 "7% diversity / 10% CD"） | 各 `distill_result.json` `loss_last` 与 Table III 行派生 | 同 C.3 | (b)（数值）+ 派生算术 | 否 | 安全 |

### C.10 Section VI-F（闭环）

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| 0.840→0.932（+10.9%）；ℓ1 0.58×；η_total ≈0.736 | eta_box 0.8404 / eta_fm 0.9321；l1 158.75→91.5 | `verify_fm_shape_loop.py`（`run_seed` 重播种 `:131` 先于建批 `:144`；true/box/fm 三策略每种子同场景配对；8 seeds）→ `isac_demo/fm_shape_loop.json` | (a) | 否 | 安全（S1 判定 false 已如实记录：盒子 ℓ1=158.75 略超注册区间 [70,150]） |

### C.11 Section VI-H（消融 / 泛化 / 可复现）

| 论断 | 数值 | 来源 | 分类 | 受影响? | 现状 |
|---|---|---|---|---|---|
| VAE oracle 0.0093 / 0.891 vs FM 0.2269 / 0.174 | `sat_model_c1/compare_gen.json`（`vae_oracle` 与 `fm["1"]` 同 run 同批） | (a) | 否 | 安全 |
| **容量归因（承重）**：残差 gap ≈ 30× 与潜容量/训练规模一致，而非生成目标 | 同 run 内 `vae_oracle` vs `fm["1"]` 比值（上一行） | **(a)** | 否 | 安全——同 run 同批比值，是"容量主因"叙事的**主承重证据** |
| **容量归因（承重）**：扩大 VAE 把 oracle floor 压到 0.0086/0.0079（低于主配置 0.0093） | `sat_model_scale_{42,43}/compare_gen.json` 的 `vae_oracle` vs `sat_model_c1` 的 0.0093 | **(c)** | **是（批噪声）** | 跨 run 比 oracle：案例 2 已证同权重不同批 oracle 可差 13%（0.0076 vs 0.0086）。主张方向（更大 VAE → 更低 ceiling）由 M1 类证据支持，但**点估计跨 run 不可直接比**。引用须附批噪声保留意见；若需铁证，按 (d) 在同一公共批上重评三个 VAE 的 oracle |
| M3 sweep 段：G-a −44.3%/−4.8%；G-b 35.0×→44.7×、30.7×→32.8× | `verify_gen_scale.py` → `isac_demo/gen_scale.json`（聚合 `sat_model_scale_<seed>` 与 `sat_model_m3_<seed>` 两份**独立产生**的 `compare_gen.json` 后做差） | **(c)** | **是** | **已被配对重评取代**：`verify_m3_paired.py` → `sat_model_cmp/m3_paired.json`（commit `66998fc`）给出 NFE=1 两种子一致改善（+11.8%/+18.8%，6/8 与 7/8 云，gap 7.8×→6.8× / 7.1×→5.8×）、NFE=10 种子不一致（−30.4% / +39.1%，3/8 与 7/8）——原"两种子均恶化"的证伪是软的。论文该段数字尚未按配对口径更新 → **待办** |
| 泛化 13–35% | TECH_REPORT 泛化表五行（cmp sat/none/ground + Starlink + N=64），逐行为各 run 行内配对 Δ | (a) 逐行；跨 run 范围为定性陈述 | 否（定性） | 安全（方向性主张；每行 Δ 行内配对） |
| ionosphere 段 | 无数值（物理陈述） | — | n/a | 否 | 安全 |

### C.12 摘要 / 引言头条句

| 论断 | 继承 |
|---|---|
| "accuracy comparable to the 100-step diffusion benchmark" / "parity" | C.6 3-seed test（混合类，parity 为保守主张） |
| "21.8% condition-explained variance" | C.7（info_audit_c1，(b)） |
| "convergence order −0.87 / straightness 0.006" | C.6（(b)；−0.87 措辞待核对） |
| "0.840 → 0.932" | C.10（(a)） |

---

## D. 三个跨运行案例的机制详述与补救状态

**案例 1 — P_scale-original（训练预算扫描，已修）**。`verify_fm_train_scale.py` `run_cell` 在单元格开头重播种（`:113-115`），但 fresh-data 训练（`:154-157`，每 epoch 经 `SatROIDataset.__getitem__` 消耗全局 `random`/`torch` 流）之后才取测试批（`:160`）——每个 (预算, 种子) 单元格记录在自己独有的 8 云批次上，跨预算比较混入批噪声（种子间符号翻转部分是 n=8 功效问题）。**补救**：`verify_fm_train_scale_paired.py`（commit `665c595`）用已存 `*_latest.pth` 在每种子一个固定公共批上重评全部单元格：`build_common_batch` 重播种先于建批（`:103-105` → `:115`）、sha1 批次指纹存档（`:118-121`；seed 42 → `a3b0522e80d2`，seed 43 → `d64e22f630aa`）、跨单元格共享 x0（`:147-149`）。结论翻转为"无一致 scale 效应"（median +0.0048/+0.0074，3/8 云改善，聚合符号由 ≤2 云驱动）。

**案例 2 — M3（生成训练预算，已修）**。`verify_gen_scale.py` 是纯聚合器：读 `sat_model_scale_<seed>` 与 `sat_model_m3_<seed>` 两份独立 `compare_gen.json`（`:24-32`）后做差（`:78-83`）；两份 JSON 的测试批各自在训练后现采样，故 Ga/Gb 证伪混有批噪声。直接佐证：同一缩放版 VAE（M3 协议复用）在两运行上的 oracle CD 为 0.0076（scale run）vs 0.0086（m3 run），相差 13%——同权重不同批即此差异。**补救**：`verify_m3_paired.py`（commit `66998fc`）复用同一 `build_common_batch`/x0 协议（导入 `:29-31`；x0 `:50-53`；指纹 seed 42 → `20e3cd6f82a4`、seed 43 → `ff9fa19c2465`）。配对结论：NFE=1 一致改善、NFE=10 种子不一致，oracle gap 仍 ~6×；原 Ga/Gb 证伪按配对口径作废（软证伪）。

**案例 3 — M1（VAE 训练规模，未修）**。`verify_vae_scale.py` 同为聚合器：`sat_model_m1_<seed>`（50ep/256）vs `sat_model_scale_<seed>`（200ep/1024）两份独立 `compare_gen.json` 做差（`:24-32, :78-83`），批次各自独立抽取。该实验仅被 `TECH_REPORT.md` §6.12 引用，**未被论文引用**；仓库中不存在 `verify_vae_scale_paired.py`——**尚未配对，待办**。其 V1/V4（VAE 天花板类结论，同 run 内比值）受影响较小，V2/V3（跨 run 的生成 CD 比较）按 (c) 处理，引用时须附带批噪声保留意见。

---

## E. 复发防护（未来实验的建批纪律）

1. **重播种先于任何 RNG 消耗**：评估批必须在显式 `manual_seed/random.seed/np.random.seed` 之后构造，且该重播种必须早于训练（或任何其他消耗全局 RNG 的步骤）；eval-only 脚本在 `main()` 开头重播种（现有正面样板：`verify_fm_bounds.py:51-53`、`eval_gen_metrics.py:60-62`、`verify_fm_distill_diversity.py:87`、`train_fm_distill.py:133`、`verify_info_audit.py:599`、`verify_fm_shape_loop.py:131`）。
2. **严禁无配对重评的跨 JSON 做差**：两份独立产生的 JSON 之间的差值主张（(c) 类）一律视为受污染，除非先做 (d) 配对重评，或能附批次一致性指纹（如 C.8 中 VAE oracle 逐位相同那一类证据）。
3. **记录批次指纹**：每个评估 JSON 存档 `sha1(pc_gt)`/`sha1(cond)`（现成实现 `verify_fm_train_scale_paired.py:118-121`）；跨 run "同批"主张必须附指纹。
4. **新标准 = 配对重评 (d)**：每种子一个固定公共批 + 跨比较对象共享 x0（`9000+seed`）+ 逐云配对差 + 预注册判读口径；论文/技术报告引用以配对 JSON 为准，原聚合值降级为"各自批次上的单次观测"。
5. **`eval_seed` 协议是底线而非上限**：`--eval_seed` 重播种（=999）使跨训练预算可比（C.3 已实证），但它不替代跨模型来源的配对重评。

---

## 附：无法从代码验证的事项（明确记账）

1. **论文 −0.87 vs 当前 JSON −0.852**：`sat_model_cmp/verify_fm_bounds.json` 现记录 `slope_euler=−0.8524`（n_eval=32）；−0.87 记录于 `docs/optimization_roadmap.md` §2.1，产生它的那次运行的参数无法从现存代码/JSON 确定（已被覆盖）。方向性结论（≈−1）不受影响，论文措辞需作者核对。
2. **C.8 中 c3_raw vs c3 的"同批"**：依据是两份 JSON 的 VAE oracle CD 逐位相同（0.008261645213）+ RNG 流演化一致性论证；未重跑 `compare_gen.py` 验证（本审计禁止训练）。如需铁证可重跑该脚本比对指纹。
3. **M1 的最终状态**：已确认无 `verify_vae_scale_paired.py`（不存在即未配对）；是否值得补做配对重评由作者决定（M1 不进论文）。
4. **`verify_headline_multiseed.py` 的 CI**：3 个种子的 bootstrap 实质为 min/max 包络（`boot_ci` 对 3 点重采样）；"95% CI"的统计强度有限，论文已按"跨零"作保守解读。

---

## 附：待办核销记录（2026-09-26）

**「无法验证 1」（收敛阶 −0.87）已核销。** 论文中的 −0.87 来自 `verify_fm_bounds.py` 在 `n_eval=32` 但 `batch_size=16`（有效 16 云）协议下的运行。规范协议（`--seed 42 --n_eval 32 --batch_size 32 --x0_seed 7`，有效 32 云）复跑两次逐位一致（确定性确认），斜率为 **−0.889**，crossover NFE=1（DDPM NFE=100 CD 0.4362 vs FM Euler NFE=1 CD 0.4234），直线性比 0.0057。收敛阶估计随评估批大小变动：n=8 → −0.94，n=16 → −0.87，n=32 → −0.89——量级结论（≈ Euler 的 −1）稳健，点估计不稳健。处置：论文改用规范协议的 −0.89 并在 VI-B 标注批大小范围；`verify_fm_bounds.py` 的 protocol 块已补记 `batch_size / nfe_list / n_effective / test_data / mode`（此前缺失导致旧 JSON 无法归因）；新 JSON 已提交。

**「无法验证 4」（headline CI 口径）已核销。** `headline_multiseed.json` 的 `ci95_*` 为 3 种子差值的 min/max 包络（bootstrap 重采样 3 点等价于包络），统计强度有限但方向保守。处置：论文 VI-B 不再使用 "bootstrap 95% CI" 措辞（3 点重采样 ≡ 包络，借统计程序之名不妥），改为 "the three per-seed differences (+4.5%, −20.5%, and −4.4%) straddle zero"——明示 n=3、直接给范围，比 CI 措辞更短更强。JSON 字段名 `ci95_*` 保留（历史产物），解读口径以本条为准。

**Table IV n=32：被加强，不是被证伪。** n=8 试测时十步采样器的离散度不可见（coverage 0.500/0.750、1-NNA 0.875），曾可能被读成"多步无优势"；评估规模升到 n=32 后十步采样器 coverage 0.812/0.906、1-NNA 0.906，一步图 1-NNA=1.000 / 内距 0.028——Remark 1 的分布学验证**加强**。口径登记：此条属于「先修协议再解读数字」的正例，falsification 登记记 **strengthened**（评估粒度不足曾掩盖效应），不记 retracted/falsified。

**复发防护新增一条**：凡 protocol 记录块必须包含全部影响批构造与拟合的参数（batch_size、nfe_list、有效样本数），否则证据 JSON 不可归因——`verify_fm_bounds.py` 的旧 protocol 块即因此无法追溯 −0.852 的来路。

**复发防护再添一条（2026-09-27 事件）**：多 seed 复用 verify 脚本时必须显式指定输出文件名——`verify_info_audit.py` 默认写 `isac_demo/info_audit.json`，W1 代理跑 seed 44 审计时未指定输出，覆盖了论文引用的 seed-42 C5 对照审计（Δ(0)=0.1436），已从 git 恢复。规则：凡输出路径带默认值的脚本，批量运行一律显式传 `--out`/输出名，收尾时用 `git status` 确认被引用证据文件零改动。
