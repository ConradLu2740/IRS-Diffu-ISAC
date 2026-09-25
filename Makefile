# =============================================================================
# IRS-Diffu-ISAC — 一键化入口
#
# 用法：
#   make setup    # 首次：创建 .venv 并安装依赖（约 2-3 分钟）
#   make verify   # 1 分钟物理验证（轨道 / 多普勒 / 信道）
#   make demo     # 感知-通信闭环 demo（自动训练感知模型 + 闭环）
#   make demo-live / demo-anim / demo-multi / demo-sdr / demo-mot
#   make track    # RIS 动态相位跟踪权衡
#   make help     # 查看全部目标
#
# 所有命令均基于仓库根目录定位，任意目录下执行 make 均可。
# 数据与模型权重均为程序内合成生成，无需任何外部下载。
# =============================================================================

VENV    := .venv
PY      := $(VENV)/bin/python
ISAC    := source_code/isac_sat
LEGACY  := source_code/legacy

.DEFAULT_GOAL := help

help: ## 显示全部可用命令
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## 首次安装：创建虚拟环境并安装依赖
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ 环境就绪。下一步：make verify"

verify: ## 物理验证：轨道 / 多普勒 / 信道（约 1 分钟）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_sat.py

demo: ## 感知-通信闭环 demo（自动训练感知模型 + 闭环，无需手动准备）
	cd $(ISAC) && bash run_demo.sh

demo-live: ## 生成多场景实时演示 HTML（交互式播放器）
	cd $(ISAC) && ../../$(VENV)/bin/python demo_live.py --n_scenes 3

demo-anim: ## 生成演示 GIF 动画
	cd $(ISAC) && ../../$(VENV)/bin/python make_animation.py

demo-multi: ## 多目标感知-通信闭环（先训练 multi 模型再演示）
	cd $(ISAC) && ../../$(VENV)/bin/python train_sensing_multi.py --wideband
	cd $(ISAC) && ../../$(VENV)/bin/python demo_multi.py

track: ## RIS 动态相位跟踪与重配置速率权衡
	cd $(ISAC) && ../../$(VENV)/bin/python verify_tracking.py

track-rician: ## P1：莱斯衰落（K=10/5/0 dB）下的 K-sweep 稳健性（多种子）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_tracking_rician.py

sdr: ## SDR 数据管线演示（无需硬件：仿真 IQ → 回放感知）
	cd $(ISAC) && ../../$(VENV)/bin/python demo_sdr.py

mot: ## 3D 多目标跟踪（10 目标，训练检测器 + 跟踪 + 动画）
	cd $(ISAC) && ../../$(VENV)/bin/python train_detect.py --n_scenes 25 --epochs 50
	cd $(ISAC) && ../../$(VENV)/bin/python demo_mot.py
	cd $(ISAC) && ../../$(VENV)/bin/python demo_mot_html.py

baseline: ## 经典基线对比（2D-CFAR + MUSIC vs 学习式感知）
	cd $(ISAC) && ../../$(VENV)/bin/python baseline_classic.py

smoke: ## 全链路最小复现（legacy 扩散重建 smoke test）
	cd $(LEGACY) && ../../$(VENV)/bin/python smoke_test.py

smoke-sim: ## isac_sim 分层骨架冒烟（信道/波形/RIS/通信/感知/跟踪，秒级）
	$(PY) tests/test_smoke_isac_sim.py

smoke-fm: ## Flow Matching 冒烟（CFM 数学 / 训练收敛 / ODE 采样，秒级）
	$(PY) tests/test_smoke_fm.py

verify-ris-sdr: ## RIS 恒模 QCQP 的 SDR 最优性证书（双侧括号 + 秩-1 证书，~30s）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_ris_sdr_certificate.py --n_seeds 4

verify-gen-hardening: ## 生成侧补强：Lipschitz 实测 + NFE=1 多样性 + 全模式指标表
	cd $(ISAC) && ../../$(VENV)/bin/python verify_gen_hardening.py

verify-decomp: ## 闭环最优性分解（16 种子 + Bootstrap 95% CI）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_optimality_decomposition.py --n_seeds 16

verify-p1-gates: ## P1 前置证伪门：κ(H_V) / VAU 坍缩 / FIM 可分离性（~10 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_p1_gates.py --n_seeds 6

verify-baselines: ## 强 baseline 同口径对比（DDIM 少步 / FM / 渐进蒸馏，GPU ~2-3 分钟）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_baselines_strong.py

verify-tracking-dp: ## RIS 分段重构 DP 最优调度（精确间隙 + 穷举证书，~16 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_tracking_dp.py --n_seeds 8

verify-pareto: ## 感知-通信 Pareto 前沿 / 多帧融合 / HRRP 信息底噪（~13 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_isac_pareto.py

verify-fim: ## 相位设计 pilot FIM 与 eta_est（信道估计因子，~10 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_phase_fim.py

verify-waveforms: ## OTFS/AFDM 真实多普勒验证（ICI 恒等式 / BER / SIR / ISAR 模型证书，~2 分钟）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_waveform_doppler.py

verify-info-audit: ## 互信息审计（Fano 阶梯 / Van Trees / CFM 恒等式，~3 分钟）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_info_audit.py

verify-fm-shape: ## FM 生成形状 vs 手工盒子先验的闭环对比（G-κ 门修复，~11 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_fm_shape_loop.py --n_seeds 8

verify-elbo: ## ELBO 一致性 + 逐维白化 A/B 证书（需先训 sat_model_elbo，~5 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_elbo_consistency.py

verify-headline: ## M1 多种子头条验证（需先训 sat_model_m1_<seed>，~1 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_headline_multiseed.py --seeds 42 43 44

verify-vae-scale: ## VAE 训练规模实验（需先训 sat_model_scale_<seed>，~1 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_vae_scale.py --seeds 42 43

verify-gen-scale: ## M3 生成训练预算实验（需先训 sat_model_m3_<seed>，~1 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_gen_scale.py --seeds 42 43

verify-cond-probe: ## 条件信息充分性门禁（cond/HRRP → 潜变量探针 R²，~10 秒）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_cond_probe.py --n_total 384

train-fm: ## Flow Matching 训练（扩散同架构/同数据，3 种 IRS 模式）
	cd $(ISAC) && ../../$(VENV)/bin/python train_fm.py

compare-gen: ## 扩散 vs Flow Matching 等算力对比（共享 VAE，输出 NFE 曲线 + JSON）
	cd $(ISAC) && ../../$(VENV)/bin/python compare_gen.py

finding-angle-wall: ## 角度墙配置扫描热力图 + 双站反例出图（CPU 秒级）
	$(PY) isac_sim/findings/plot_angle_wall_scan.py

verify-sionna: ## L2：Sionna CDL 标准信道对照（3GPP TR 38.901，需 pip install sionna）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_sionna_channel.py

twostation: ## 双站三边定位实验（角度墙实证反例：噪声×方位差扫描，~1 min）
	cd $(ISAC) && ../../$(VENV)/bin/python verify_twostation_localization.py

clean: ## 清理演示产物（checkpoints / HTML / GIF / PNG）
	rm -rf $(ISAC)/isac_demo
	@echo "已清理 isac_demo/"
