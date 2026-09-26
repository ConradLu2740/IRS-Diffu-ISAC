"""fig2_framework.py — 论文 Fig. 2：FMShape 框架图（推理主链路 + 训练支路）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

C_IN, C_ENC, C_NET, C_DEC = "#4C72B0", "#55A868", "#C44E52", "#8172B3"
C_STEP, C_TRAIN = "#B8963E", "#999999"

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 7.0); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.0, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, color="#444444", ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls))


# ================= 训练支路（上排，虚线） =================
box(0.25, 5.55, 2.35, 0.90, "Point cloud $\\mathbf{P}$\n(ground truth)", "#F4F4F4", "#888888")
box(3.35, 5.55, 2.05, 0.90, "VAE encoder\n$\\mathbf{x}_1=(\\mathbf{z}-\\mu)/\\sigma$\n(whitened latent)", "#F0ECF7", C_DEC, fs=7.4)
box(6.15, 5.55, 2.60, 0.90, "Path: $\\mathbf{x}_t=(1-t)\\,\\mathbf{x}_0+t\\,\\mathbf{x}_1$\n$t\\sim U(0,1)$, fresh data / epoch", "#FBFBFB", C_TRAIN, fs=7.4)
box(9.55, 5.55, 2.20, 0.90, "FM loss (5):\nregress $\\mathbf{x}_1-\\mathbf{x}_0$\n(simulation-free)", "#FBFBFB", C_TRAIN, fs=7.4)

arrow(2.60, 6.00, 3.35, 6.00, color=C_TRAIN, ls=(0, (4, 3)), lw=1.2)
arrow(5.40, 6.00, 6.15, 6.00, color=C_TRAIN, ls=(0, (4, 3)), lw=1.2)
arrow(8.75, 6.00, 9.55, 6.00, color=C_TRAIN, ls=(0, (4, 3)), lw=1.2)
ax.text(6.0, 6.62, "training (simulation-free)", fontsize=7.4, color=C_TRAIN, ha="center", style="italic")

# ================= 推理主链路（中排） =================
box(0.25, 3.30, 2.35, 1.05, "Measured HRRP\n$\\mathbf{h}\\in\\mathbb{R}^{512}$\n(centroid-aligned)", "#EAF0F8", C_IN)
box(0.25, 1.95, 2.35, 0.80, "Noise\n$\\mathbf{x}_0\\sim\\mathcal{N}(0,\\mathbf{I})$, $d$ = 256", "#F4F4F4", "#888888")
box(3.35, 3.30, 2.05, 1.05, "Condition\nencoder (LSTM)\n$\\to$ embedding $c$", "#E8F3EA", C_ENC)
box(6.15, 2.95, 2.60, 1.75,
    "OT-CFM velocity net $v_\\theta$\n1-D DiT + cross-attention\n(2 blocks, 256 hidden)\n\nCFG: $v_{cfg}=v_{\\emptyset}+w\\,(v_c-v_{\\emptyset})$, $w$=2",
    "#FBEAEA", C_NET, fs=7.6)
box(9.55, 3.30, 2.20, 1.05, "Single Euler step\n$\\hat{\\mathbf{x}}_1=\\mathbf{x}_0+v_{cfg}(\\mathbf{x}_0,0)$\n(1 ODE step)", "#FFF6E5", C_STEP, fs=7.6)

arrow(2.60, 3.82, 3.35, 3.82)
arrow(5.40, 3.82, 6.15, 3.82)
arrow(2.60, 2.35, 8.75, 2.35, color="#888888")
arrow(8.75, 2.35, 9.55, 3.30, color="#888888")
ax.text(5.6, 2.52, "one ODE step (2 forward passes under CFG)", fontsize=6.8, color="#888888", ha="center")

# ================= 解码列（右下） =================
box(9.55, 1.95, 2.20, 0.80, "VAE decoder\n(per-dim de-whitening)", "#F0ECF7", C_DEC, fs=7.6)
box(9.55, 0.85, 2.20, 0.80, "Reconstructed shape\n$\\hat{\\mathbf{P}}$ (512 × 3)", "#F0ECF7", C_DEC, fs=7.6)
arrow(10.65, 3.30, 10.65, 2.75, color=C_STEP)
arrow(10.65, 1.95, 10.65, 1.65, color=C_DEC)

# ================= 定理 callout =================
ax.add_patch(FancyBboxPatch((1.6, 0.18), 7.0, 0.62, boxstyle="round,pad=0.06",
                            fc="#FFF9E8", ec=C_STEP, lw=1.2))
ax.text(5.1, 0.49, "Theorem: the single-step output equals the conditional mean $\\mu(c)$ (Section V)",
        fontsize=7.8, ha="center", va="center", color="#7A5F14")

fig.tight_layout()
path = os.path.join(OUT, "fig2_framework.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"saved: {path}")
