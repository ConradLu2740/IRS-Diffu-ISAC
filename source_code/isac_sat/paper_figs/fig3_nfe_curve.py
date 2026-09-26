"""fig3_nfe_curve.py — 论文 Fig. 3：质量-NFE 曲线（真实实验数据）
(a) HRRP 条件化 FMShape 的 NFE 曲线 + DDPM 基线 + VAE 上界
(b) 强少步基线（DDIM / FM-Euler / 1 步蒸馏学生）
数据: sat_model_c1/compare_gen.json, sat_model_cmp/strong_baselines.json
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ISAC = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "..", "..", "..", "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

c1 = json.load(open(os.path.join(ISAC, "sat_model_c1", "compare_gen.json")))["sat"]
sb = json.load(open(os.path.join(ISAC, "sat_model_cmp", "strong_baselines.json")))["results"]

fm_nfe = sorted(int(k) for k in c1["fm"])
fm_cd = [c1["fm"][str(n)]["cd"] for n in fm_nfe]
ddpm_cd = c1["ddpm"]["100"]["cd"]
oracle_cd = c1["vae_oracle"]["cd"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))

# ---------- (a) HRRP-conditioned NFE curve ----------
ax1.plot(fm_nfe, fm_cd, "o-", color="crimson", lw=1.8, ms=5, label="FMShape (Euler)")
ax1.axhline(oracle_cd, ls="--", c="gray", alpha=0.7, lw=1.2, label="VAE oracle")
ax1.scatter([100], [ddpm_cd], marker="*", s=170, color="steelblue", zorder=5,
            label="DDPM (NFE=100)")
ax1.annotate(f"NFE=1: {fm_cd[0]:.3f}", xy=(1, fm_cd[0]), xytext=(1.35, fm_cd[0] - 0.055),
             fontsize=7.5, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson", lw=0.9))
ax1.axvspan(0.9, 2.1, color="gold", alpha=0.14)
ax1.text(1.45, 0.60, "equal-quality\ncrossover\nNFE$^*$ $\\leq$ 2", fontsize=7, color="#7A5F14", ha="left")
ax1.set_xscale("log")
ax1.set_xlabel("NFE (network evaluations)")
ax1.set_title("(a) HRRP-conditioned FMShape", fontsize=8.5)
ax1.set_ylim(0, 0.68)
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(fontsize=7, loc="upper right", framealpha=0.9)

# ---------- (b) strong few-step baselines ----------
ddim = sorted((v["nfe"], v["cd"]) for k, v in sb.items() if k.startswith("DDIM"))
fm_e = sorted((v["nfe"], v["cd"]) for k, v in sb.items() if k.startswith("FM_") and "distill" not in k)
dist = [(v["nfe"], v["cd"]) for k, v in sb.items() if "distill" in k][0]
ddpm_s = sb["DDPM_T"]

ax2.plot([n for n, _ in ddim], [c for _, c in ddim], "s--", color="#55A868", lw=1.5, ms=5,
         label="DDIM (zero-retrain)")
ax2.plot([n for n, _ in fm_e], [c for _, c in fm_e], "o-", color="crimson", lw=1.5, ms=5,
         label="FMShape (Euler)")
ax2.scatter([dist[0]], [dist[1]], marker="D", s=60, color="#8172B3", zorder=5,
            label="FM-distill (1-step)")
ax2.scatter([ddpm_s["nfe"]], [ddpm_s["cd"]], marker="*", s=170, color="steelblue", zorder=5,
            label="DDPM (NFE=100)")
ax2.annotate(f"{dist[1]:.3f}", xy=dist, xytext=(1.25, dist[1] - 0.045), fontsize=7.5,
             color="#8172B3", arrowprops=dict(arrowstyle="->", color="#8172B3", lw=0.9))
ax2.set_xscale("log")
ax2.set_xlabel("NFE (network evaluations)")
ax2.set_title("(b) Few-step baselines (common batch)", fontsize=8.5)
ax2.set_ylim(0.15, 0.36)
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(fontsize=6.6, loc="upper right", framealpha=0.9)

fig.supylabel("Chamfer Distance $\\downarrow$", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
path = os.path.join(OUT, "fig3_nfe_curve.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"saved: {path}")
