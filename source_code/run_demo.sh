#!/usr/bin/env bash
# 星-地 ISAC 感知-通信闭环 demo 快速启动
set -e
cd "$(dirname "$0")"

# 1. 训练感知模型（若不存在）
if [ ! -f ./isac_demo/sensing_best.pth ]; then
    echo "[run_demo] 训练感知模型..."
    ../.venv/bin/python train_sensing.py --irs_mode sat --phase_mode tracked --wideband --epochs 40
fi

# 2. 运行闭环 demo
echo "[run_demo] 运行感知-通信闭环 demo..."
../.venv/bin/python demo.py

echo "[run_demo] 完成，结果图: ./isac_demo/demo_result.png"
