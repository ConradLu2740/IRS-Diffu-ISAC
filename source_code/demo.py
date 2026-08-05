"""
demo.py — 星-地 ISAC 端到端感知-通信闭环演示

流程（卫星过境）：
  1. 感知：宽带距离像 → 目标分类（6 类）+ 位置估计
  2. 通信：感知结果 → IRS 相位配置（指向目标）→ 接收功率
  3. 闭环对比：感知辅助（sensed） vs 理想（oracle） vs 随机（random）

工程说明：
  - 感知模型由 train_sensing.py 训练（checkpoint: isac_demo/sensing_best.pth）
  - IRS 相位用解析对齐（phase_optimizer_sat.py）
  - 感知闭环 = 用估计位置构建 ROI → 优化相位（体现"感知驱动配置"）

运行：
  python demo.py [--checkpoint ./isac_demo/sensing_best.pth]
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import (SatROIDataset, SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      compute_range_profile, _SIGNAL1, IRS_GAIN)
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing import SensingMLP

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def estimate_roi_from_pos(pos_xy, res=16):
    """感知位置 (x,y 归一化 [-1,1]) → 估计 ROI 体素（目标附近块）。"""
    roi = np.zeros((res, res, res), dtype=np.float32)
    cx = int((pos_xy[0] + 1) / 2 * res)
    cy = int((pos_xy[1] + 1) / 2 * res)
    half = 2  # 目标块半径（体素）
    x0, x1 = max(cx - half, 0), min(cx + half + 1, res)
    y0, y1 = max(cy - half, 0), min(cy + half + 1, res)
    roi[x0:x1, y0:y1, res // 2 - 1: res // 2 + 2] = 1.0
    return roi


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ---- 1. 加载感知模型 ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[demo] 感知模型加载: cond_dim={ckpt['cond_dim']}, "
          f"irs={ckpt['irs_mode']}, phase={ckpt['phase_mode']}, wideband={ckpt['wideband']}")

    # ---- 2. 场景 ----
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]
    print(f"[demo] 卫星: {scenario.sat_name}, 过境帧 {len(frames)}, "
          f"目标({scenario.target_lat}N,{scenario.target_lon}E)")

    # ---- 3. 真实目标（随机类别 + ROI） ----
    from data_sat import generate_ground_target_sample
    roi_true, cid_true, ang_true = generate_ground_target_sample()
    pc_true = (np.argwhere(roi_true > 0.5)).astype(np.float32)
    pos_true = pc_true.mean(axis=0) / 16.0 * 2.0 - 1.0   # 体素→归一化质心
    pos_true = pos_true[:2]   # 只取 x,y
    print(f"[demo] 真实目标: {CLASS_NAMES[cid_true]} @ 位置({pos_true[0]:+.2f},{pos_true[1]:+.2f})")

    # ---- 4. 感知（宽带距离像） ----
    rp = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0,
                               align=not args.rp_align)
    rp_t = torch.from_numpy(rp).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, pred_pos = model(rp_t)
    cid_pred = logits.argmax(1).item()
    pos_pred = pred_pos[0].cpu().numpy()
    cls_ok = (cid_pred == cid_true)
    pos_err = np.linalg.norm(pos_pred - pos_true)  # 归一化坐标误差
    print(f"[demo] 感知: 分类={CLASS_NAMES[cid_pred]} (正确:{cls_ok}), "
          f"位置估计({pos_pred[0]:+.2f},{pos_pred[1]:+.2f}), 误差={pos_err:.3f}")

    # ---- 5. 通信：IRS 相位策略对比（跨帧） ----
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    roi_true_t = torch.tensor(roi_true.astype(np.float32)).reshape(-1)
    roi_est = estimate_roi_from_pos(pos_pred)
    roi_est_t = torch.tensor(roi_est.astype(np.float32)).reshape(-1)

    powers = {"oracle": [], "sensed": [], "random": []}
    for t, Ht in enumerate(channels.channels_per_frame):
        # 理想：真实 ROI 优化（上界）
        ph_oracle = opt.optimize_frame(Ht, roi_true_t, X)
        powers["oracle"].append(opt._power(Ht, roi_true_t, X, ph_oracle))
        # 感知闭环：估计 ROI 优化（感知位置驱动）
        ph_sensed = opt.optimize_frame(Ht, roi_est_t, X)
        powers["sensed"].append(opt._power(Ht, roi_true_t, X, ph_sensed))
        # 基线：随机相位
        n_irs = Ht["H_ROI_IRS"].shape[1]
        ph_rand = torch.rand(n_irs) * 2 * np.pi
        powers["random"].append(opt._power(Ht, roi_true_t, X, ph_rand))

    p_rand = float(np.mean(powers["random"]))
    p_sensed = float(np.mean(powers["sensed"]))
    p_oracle = float(np.mean(powers["oracle"]))

    print("\n" + "=" * 66)
    print("通信性能对比（接收功率，越大越好）：")
    print(f"  随机相位 (baseline)   : {p_rand:.4e}")
    print(f"  感知辅助 (sensed)     : {p_sensed:.4e}  ({100*(p_sensed/p_rand-1):+.1f}%)")
    print(f"  理想优化 (oracle)     : {p_oracle:.4e}  ({100*(p_oracle/p_rand-1):+.1f}%)")
    print(f"  感知闭环达成 oracle   : {100*p_sensed/p_oracle:.1f}%")
    print("=" * 66)

    # ---- 6. 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) 位置：真实 vs 感知
    ax = axes[0, 0]
    ax.scatter(*pos_true, c="green", s=120, marker="o", label="True target")
    ax.scatter(*pos_pred, c="crimson", s=120, marker="x", label="Sensed")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("x (normalized)"); ax.set_ylabel("y (normalized)")
    ax.set_title(f"Target Localization (err={pos_err:.2f})")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)

    # (b) 分类结果
    ax = axes[0, 1]
    labels = CLASS_NAMES
    vals = np.zeros(len(labels)); vals[cid_true] = 1.0
    preds = np.zeros(len(labels)); preds[cid_pred] = 1.0
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals, 0.4, color="green", label="True")
    ax.bar(x + 0.2, preds, 0.4, color="crimson", label="Predicted")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
    ax.set_title(f"Target Classification ({'Correct' if cls_ok else 'Wrong'})")
    ax.legend()

    # (c) 通信功率 vs 帧（三种策略）
    ax = axes[1, 0]
    t = np.arange(len(frames))
    for name, color in [("random", "gray"), ("sensed", "crimson"), ("oracle", "steelblue")]:
        ax.plot(t, powers[name], marker="o", ms=4, label=name, color=color)
    ax.set_xlabel("Frame"); ax.set_ylabel("Received power")
    ax.set_title("Communication Power by IRS Strategy")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)

    # (d) 闭环增益摘要
    ax = axes[1, 1]
    names = ["random", "sensed", "oracle"]
    gains = [100 * (np.mean(powers[n]) / p_rand - 1) for n in names]
    colors = ["gray", "crimson", "steelblue"]
    ax.bar(names, gains, color=colors)
    ax.set_ylabel("Gain vs random (%)")
    ax.set_title("Closed-loop Gain Summary")
    for i, g in enumerate(gains):
        ax.text(i, g + 0.5, f"{g:+.1f}%", ha="center")
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    fig.suptitle("Satellite-Ground ISAC: Sensing-Communication Closed Loop Demo", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "demo_result.png")
    fig.savefig(out_png, dpi=150)
    print(f"\n[demo] 结果图: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星-地 ISAC 感知-通信闭环 demo")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="tracked")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--rp_align", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
