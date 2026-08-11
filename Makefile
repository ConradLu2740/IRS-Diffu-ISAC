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

clean: ## 清理演示产物（checkpoints / HTML / GIF / PNG）
	rm -rf $(ISAC)/isac_demo
	@echo "已清理 isac_demo/"
