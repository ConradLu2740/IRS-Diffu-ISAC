"""fig1_scenario.py — 论文 Fig. 1：空-地 ISAC 场景示意图"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

C_SAT, C_COMM, C_SENSE, C_RIS, C_ROI = "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#B8963E"

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.8); ax.axis("off")

theta = np.linspace(np.pi * 1.03, np.pi * 1.97, 200)
ex, ey = 5 + 7.2 * np.cos(theta), 0.55 + 2.5 * np.sin(theta)
ax.plot(ex, ey, color="#8C8C8C", lw=2.2)
ax.fill_between(ex, ey, -1.0, color="#ECECEC", zorder=0)
ax.text(1.15, 0.72, "Earth", fontsize=9, color="#777777", style="italic")

ax.add_patch(FancyBboxPatch((7.55, 1.30), 0.9, 0.52, boxstyle="round,pad=0.04",
                            fc="#DCE9DA", ec=C_COMM, lw=1.4))
ax.text(8.0, 1.56, "UE", ha="center", va="center", fontsize=8.5)

ax.add_patch(Rectangle((4.45, 1.70), 1.5, 0.34, fill=False, ec=C_RIS, lw=1.5))
for x in np.linspace(4.52, 5.88, 11):
    ax.plot([x, x + 0.05], [1.74, 2.00], color=C_RIS, lw=1.0)
ax.text(5.2, 1.40, "RIS phase $\\phi_t$", ha="center", fontsize=8.5, color=C_RIS)

ax.add_patch(Rectangle((1.05, 1.70), 1.6, 1.10, fc="#F6EFDB", ec=C_ROI, lw=1.6))
for gx in np.linspace(1.05, 2.65, 6):
    ax.plot([gx, gx], [1.70, 2.80], color=C_ROI, lw=0.55, alpha=0.75)
for gy in np.linspace(1.70, 2.80, 6):
    ax.plot([1.05, 2.65], [gy, gy], color=C_ROI, lw=0.55, alpha=0.75)
ax.text(1.85, 1.36, "ROI voxels $\\mathbf{V}$", ha="center", fontsize=8.5, color="#8A7420")
ax.text(1.85, 2.98, "target region", ha="center", fontsize=7.5, color="#8A7420", style="italic")

ax.add_patch(Arc((5, 5.0), 7.6, 3.0, theta1=18, theta2=162, color=C_SAT, lw=1.3, ls=(0, (5, 3))))
ax.add_patch(FancyBboxPatch((4.40, 4.66), 1.2, 0.58, boxstyle="round,pad=0.05",
                            fc="#D9E2F0", ec=C_SAT, lw=1.5))
ax.plot([4.02, 4.40], [4.95, 4.95], color=C_SAT, lw=1.5)
ax.plot([5.60, 5.98], [4.95, 4.95], color=C_SAT, lw=1.5)
ax.add_patch(Rectangle((3.82, 4.71), 0.2, 0.48, fc=C_SAT, ec=C_SAT))
ax.add_patch(Rectangle((5.98, 4.71), 0.2, 0.48, fc=C_SAT, ec=C_SAT))
ax.text(5.0, 5.62, "LEO satellite (SGP4 / TLE)", ha="center", fontsize=9.5, color=C_SAT)
ax.text(8.85, 4.78, "slant range $\sim$695 km\nDoppler $\pm$611 kHz", fontsize=7.5,
        color="#555555", ha="center")

ax.add_patch(FancyArrowPatch((4.55, 4.62), (1.98, 2.88), arrowstyle="-|>", mutation_scale=13,
                             color=C_SENSE, lw=1.7, ls=(0, (4, 2))))
ax.add_patch(FancyArrowPatch((1.98, 2.88), (2.75, 1.72), arrowstyle="-|>", mutation_scale=13,
                             color=C_SENSE, lw=1.7, ls=(0, (4, 2))))
ax.text(3.66, 4.02, "sensing echo\n(direct path)", fontsize=8, color=C_SENSE, ha="left")

ax.add_patch(FancyArrowPatch((5.45, 4.66), (5.05, 2.10), arrowstyle="-|>", mutation_scale=13,
                             color=C_COMM, lw=1.7))
ax.add_patch(FancyArrowPatch((5.30, 2.04), (7.60, 1.56), arrowstyle="-|>", mutation_scale=13,
                             color=C_COMM, lw=1.7))
ax.text(6.75, 2.85, "comm link\n(RIS-assisted)", fontsize=8, color=C_COMM, ha="center")

ax.add_patch(FancyBboxPatch((0.35, 3.30), 3.15, 1.28, boxstyle="round,pad=0.08",
                            fc="#FBFBFB", ec="#BBBBBB", lw=1.0))
ax.text(1.92, 4.32, "per frame $t=1,\\dots,T$ ($T$=8)", fontsize=8, ha="center", color="#444444")
ax.text(1.92, 3.88, "slant range · round-trip delay\nline-of-sight Doppler", fontsize=8,
        ha="center", color="#444444")
ax.text(1.92, 3.48, "$\\Rightarrow$ HRRP $\\mathbf{h}_t\\in\\mathbb{R}^{512}$\n(centroid-aligned)", fontsize=8,
        ha="center", color=C_SENSE)

ax.text(5.0, 0.28, "Sensing echo is dominated by the direct satellite–ROI–ground path (structurally independent of $\\phi_t$); "
        "the RIS shapes the comm link.", fontsize=7.5, ha="center", color="#666666")

fig.tight_layout()
path = os.path.join(OUT, "fig1_scenario.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"saved: {path}")
