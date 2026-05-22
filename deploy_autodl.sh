#!/bin/bash

echo "========================================="
echo " IRS_Diffu_ISAC AutoDL 一键部署脚本"
echo " 适配 RTX 5090 (CUDA 12.8 + PyTorch 2.7+)"
echo "========================================="

cd /root/autodl-tmp

if [ ! -d "IRS_Diffu_ISAC" ]; then
    mkdir -p IRS_Diffu_ISAC
fi

cd IRS_Diffu_ISAC

if [ -f "source_code.zip" ]; then
    echo "[1/5] 解压代码..."
    unzip -o source_code.zip -d source_code/
    rm source_code.zip
    echo "  完成"
else
    echo "[1/5] 未找到 source_code.zip，跳过解压"
    echo "  请确保代码已在 /root/autodl-tmp/IRS_Diffu_ISAC/source_code/ 目录下"
fi

echo "[2/5] 检查 PyTorch 版本..."
PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  当前 PyTorch: $PYTORCH_VER, CUDA: $CUDA_VER"

MAJOR=$(echo $PYTORCH_VER | cut -d. -f1)
MINOR=$(echo $PYTORCH_VER | cut -d. -f2)

if [ "$MAJOR" -lt 2 ] || ([ "$MAJOR" -eq 2 ] && [ "$MINOR" -lt 6 ]); then
    echo "  ⚠️  PyTorch < 2.6，RTX 5090 需要升级！"
    echo "  正在安装 PyTorch 2.7 + CUDA 12.8..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
    echo "  完成"
else
    echo "  PyTorch 版本满足要求"
fi

echo "[3/5] 安装其他依赖..."
pip install scipy matplotlib -q
echo "  完成"

echo "[4/5] 验证环境..."
cd source_code
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn, LatentDiT_Token_CrossAttn
print('  Models: OK')

from setup import precompute_channels
H = precompute_channels('cpu')
print(f'  H_ROI_BS in H_dict: {\"H_ROI_BS\" in H}')
print('  Setup: OK')
"
echo "  完成"

echo "[5/5] 创建输出目录..."
mkdir -p model outputs_npy outputs_vae outputs_ldm_no_irs outputs_ldm_random_irs
echo "  完成"

echo ""
echo "========================================="
echo " 部署完成！"
echo "========================================="
echo ""
echo " 训练命令（推荐使用 tmux）："
echo "   tmux new -s train"
echo "   Stage 1 (VAE):    python train.py"
echo "   复制权重:          cp model/vae_best.pth outputs_vae/best.pth"
echo "   Stage 2 (No IRS): python train_no.py"
echo "   Stage 2 (Ran IRS):python train_r.py"
echo ""
echo " 监控命令："
echo "   GPU状态:   watch -n 1 nvidia-smi"
echo "   检查权重:  ls -lh model/*.pth"
echo ""
echo " RTX 5090 预估训练时间："
echo "   VAE:   约 0.5-1 小时"
echo "   LDM:   约 2-4 小时/场景"
echo "   合计:  约 5-9 小时（¥14-26）"
echo ""
