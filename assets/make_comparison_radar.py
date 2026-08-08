"""Generate capability-coverage radar chart comparing IRS-Diffu-ISAC vs related open-source projects.
Output: docs/comparison_radar.svg (vector, GitHub-friendly)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as <text> (searchable, smaller)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---- capability scoring (0=absent, 1=partial/framework-level, 2=full) ----
dims = [
    "RIS modeling",
    "Diffusion\n3D recon",
    "Sensing-Comm\nclosed loop",
    "Real LEO\norbit (SGP4)",
    "Multi-object\n3D tracking",
    "SDR data\ninterface",
    "Physics\nverification",
    "Instant\ndemo (Colab)",
]
projects = {
    "IRS-Diffu-ISAC":        [2, 2, 2, 2, 2, 2, 2, 2],
    "5G ISAC Sys-Level":     [0, 0, 1, 0, 0, 0, 2, 0],
    "ISAC-PLM (802.11ay)":   [0, 0, 1, 0, 0, 1, 2, 1],
    "PassiveDOA-ISAC-RIS":   [1, 0, 0, 0, 0, 0, 1, 0],
    "Diffusion 3D (PVD)":    [0, 2, 0, 0, 0, 0, 2, 2],
}
colors = {
    "IRS-Diffu-ISAC":      "#d62728",
    "5G ISAC Sys-Level":   "#1f77b4",
    "ISAC-PLM (802.11ay)": "#ff7f0e",
    "PassiveDOA-ISAC-RIS": "#2ca02c",
    "Diffusion 3D (PVD)":  "#9467bd",
}

N = len(dims)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(9.5, 8.6), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

# grid
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_ylim(0, 2)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["", "partial", "full"], fontsize=9, color="#666666")
ax.grid(color="#dddddd", linewidth=0.8)

for label, vals in projects.items():
    data = vals + vals[:1]
    ax.plot(angles, data, linewidth=2.2, color=colors[label], label=label)
    ax.fill(angles, data, color=colors[label], alpha=0.06)

# dimension labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=10.5)

# highlight our project
our = projects["IRS-Diffu-ISAC"] + projects["IRS-Diffu-ISAC"][:1]
ax.plot(angles, our, linewidth=3.4, color=colors["IRS-Diffu-ISAC"], label="IRS-Diffu-ISAC", zorder=5)
ax.fill(angles, our, color=colors["IRS-Diffu-ISAC"], alpha=0.14, zorder=4)

ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=10, frameon=True)
ax.set_title("Capability Coverage vs Related Open-Source Projects",
             fontsize=15, fontweight="bold", pad=28)

plt.tight_layout()
plt.savefig("assets/comparison_radar.svg", bbox_inches="tight", facecolor="white")
print("saved assets/comparison_radar.svg")
