"""
make_animation.py — 生成演示动画 GIF（基于 demo 数据，Pillow writer）

复用 demo_live 的仿真数据生成逻辑，用 matplotlib 渲染动画帧，
保存为 GIF（无需 ffmpeg）。

用法：
  python make_animation.py
  输出：./isac_demo/demo_animation.gif
"""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 中文字体（macOS）
for _f in ["PingFang SC", "Arial Unicode MS", "Heiti TC", "Songti SC", "STHeiti"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

import setup_sat as ss
from data_sat import (SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      compute_range_profile, _SIGNAL1, generate_ground_target_sample)
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing import SensingMLP

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)

    roi, cid_true, ang_true = generate_ground_target_sample()
    pos_true = np.argwhere(roi > 0.5).astype(np.float32).mean(axis=0)
    pos_true = (pos_true / 16.0 * 2.0 - 1.0)[:2]

    rp = compute_range_profile(roi, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0, align=False)
    with torch.no_grad():
        logits, pred_pos = model(torch.from_numpy(rp).float().unsqueeze(0))
    cid_pred = logits.argmax(1).item()
    pos_pred = pred_pos[0].numpy()[:2]

    res = 16
    roi_est = np.zeros((res, res, res), dtype=np.float32)
    cx = int((pos_pred[0] + 1) / 2 * res); cy = int((pos_pred[1] + 1) / 2 * res)
    roi_est[max(cx - 2, 0):min(cx + 3, res), max(cy - 2, 0):min(cy + 3, res), 7:9] = 1.0
    roi_true_t = torch.tensor(roi.astype(np.float32)).reshape(-1)
    roi_est_t = torch.tensor(roi_est.astype(np.float32)).reshape(-1)

    n = len(frames)
    t0 = datetime(*scenario.start_utc)
    elevs = [f["elevation_deg"] for f in frames]
    utcs = [(t0 + timedelta(seconds=f["t_abs_sec"])).strftime("%H:%M:%S") for f in frames]
    powers = {"random": [], "sensed": [], "oracle": []}
    for Ht in channels.channels_per_frame:
        n_irs = Ht["H_ROI_IRS"].shape[1]
        powers["random"].append(opt._power(Ht, roi_true_t, X, torch.rand(n_irs) * 2 * np.pi))
        powers["sensed"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_est_t, X)))
        powers["oracle"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_true_t, X)))
    all_p = [p for v in powers.values() for p in v]
    p_min, p_max = min(all_p), max(all_p)
    p_norm = {k: [(p - p_min) / (p_max - p_min + 1e-9) for p in v] for k, v in powers.items()}

    gain_s = 100 * (np.mean(powers["sensed"]) / np.mean(powers["random"]) - 1)
    cls_ok = cid_pred == cid_true

    # ---- 动画 ----
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 0.8])
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_power = fig.add_subplot(gs[1, :])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    sat_line, = ax_scene.plot([], [], "o-", color="#4fc3f7", lw=2, ms=6, label="Satellite")
    link_line, = ax_scene.plot([], [], "--", color="#ff5252", lw=1, alpha=0.6)
    beam_poly = None

    def init():
        ax_scene.set_xlim(0, 10); ax_scene.set_ylim(0, 8)
        ax_scene.set_aspect("equal"); ax_scene.set_xticks([]); ax_scene.set_yticks([])
        ax_scene.set_title("Satellite Overpass (side view)")
        # 地面
        ax_scene.axhline(0.5, color="#3a5a3a", lw=3)
        ax_scene.fill_between([0, 10], 0, 0.5, color="#1d2c1d")
        # 目标 / 地面站
        ax_scene.plot(4.4, 0.6, "o", color="#7fd151", ms=12)
        ax_scene.text(4.4, 0.9, CLASS_NAMES[cid_true], color="#7fd151", fontsize=10)
        ax_scene.plot(6.2, 0.6, "o", color="#ffb74d", ms=10)
        ax_scene.text(6.2, 0.9, "UE", color="#ffb74d", fontsize=10)
        # 感知位置
        ax_scene.plot(4.4 + pos_pred[0] * 0.8, 0.7, "x", color="#ff5252", ms=10, mew=2)
        ax_power.set_xlim(0, n - 1); ax_power.set_ylim(0, 1.05)
        ax_power.set_xlabel("Frame"); ax_power.set_ylabel("Norm. power")
        ax_power.grid(True, ls="--", alpha=0.4)
        ax_power.set_title("Comm. power by IRS strategy")
        for k, c, lbl in [("random", "#9aa7bd", "random"), ("sensed", "#ff5252", "sensed"), ("oracle", "#4fc3f7", "oracle")]:
            ax_power.plot([], [], color=c, label=lbl)
        ax_power.legend(loc="upper left", fontsize=9)
        return []

    def update(i):
        prog = i / max(n - 1, 1)
        elev = elevs[i]
        # 卫星位置（天空弧线）
        sx = 0.5 + 9.0 * prog
        sy = 3.5 + (elev / 60.0) * 3.5
        sat_line.set_data([sx], [sy])
        link_line.set_data([sx, 4.4], [sy, 0.6])
        # 功率曲线
        ax_power.lines[-3].set_data(range(i + 1), p_norm["random"][:i + 1])
        ax_power.lines[-2].set_data(range(i + 1), p_norm["sensed"][:i + 1])
        ax_power.lines[-1].set_data(range(i + 1), p_norm["oracle"][:i + 1])
        # 信息面板
        ax_info.clear(); ax_info.axis("off")
        info_lines = [
            f"Frame {i + 1}/{n} | UTC {utcs[i]} | Elev {elevs[i]:.1f} deg",
            f"Class: {CLASS_NAMES[cid_pred]} {'OK' if cls_ok else 'MISS'} (true {CLASS_NAMES[cid_true]})",
            f"Pos: true({pos_true[0]:+.2f},{pos_true[1]:+.2f}) sensed({pos_pred[0]:+.2f},{pos_pred[1]:+.2f})",
            f"Closed-loop gain: +{gain_s:.0f}% vs random",
        ]
        for j, s in enumerate(info_lines):
            ax_info.text(0.05, 0.85 - j * 0.18, s, fontsize=11, transform=ax_info.transAxes)
        ax_info.set_title("Sensing result", fontsize=11)
        return []

    anim = FuncAnimation(fig, update, frames=n, init_func=init, blit=False, repeat=True)
    gif_path = os.path.join(OUT_DIR, "demo_animation.gif")
    anim.save(gif_path, writer=PillowWriter(fps=2.0))
    print(f"[make_animation] GIF: {gif_path}")
    print(f"[make_animation] 分类 {'正确' if cls_ok else '错误'}, 增益 +{gain_s:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成演示动画 GIF")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=16)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
