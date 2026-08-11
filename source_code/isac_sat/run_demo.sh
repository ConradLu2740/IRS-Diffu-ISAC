#!/usr/bin/env bash
# 星-地 ISAC 感知-通信闭环 demo 快速启动
# 用法：在仓库根目录执行  bash source_code/isac_sat/run_demo.sh  （或 make demo）
set -e
cd "$(dirname "$0")"

# 从仓库根定位虚拟环境（兼容：优先 REPO_ROOT/.venv，其次 PATH 中的 python3）
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY=python3
    echo "[run_demo] 未找到 $REPO_ROOT/.venv，使用系统 python3（请先 make setup）"
fi

# 1. 训练感知模型（若不存在）
if [ ! -f ./isac_demo/sensing_best.pth ]; then
    echo "[run_demo] 训练感知模型..."
    "$PY" train_sensing.py --irs_mode sat --phase_mode tracked --wideband --epochs 40
fi

# 2. 运行闭环 demo
echo "[run_demo] 运行感知-通信闭环 demo..."
"$PY" demo.py

echo "[run_demo] 完成，结果图: ./isac_demo/demo_result.png"
