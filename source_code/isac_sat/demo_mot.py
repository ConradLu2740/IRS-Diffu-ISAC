"""
demo_mot.py — 10 个移动目标的多目标检测与追踪（MOT）演示

流程：
  1. 生成移动场景（10 目标 / 5 类 / 运动学）
  2. 逐帧检测（距离像 → K=10 组分类+定位）
  3. MOT 追踪（匈牙利关联 + α-β 滤波，保持 ID）
  4. 评估：轨迹数 / 检测召回 / ID 稳定性 / 类别
  5. 可视化：目标轨迹动画（GIF）

物理说明：单站距离像对同距离单元目标分辨有限（检测率 ~44%），
MOT 用轨迹连续性补足漏检并稳定 ID——追踪价值即在此。

用法：
  python demo_mot.py [--checkpoint ./isac_demo/detect_best.pth]
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import linear_sum_assignment

import data_sat
from mot_data import MovingTargetScene, CLASS_NAMES
from mot_tracker import MOTTracker
from train_detect import DetectNet

OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def run_detector(model, rp, device):
    """单帧距离像 → K 组检测（pos, cls_probs）。返回按置信度排序的检测。"""
    with torch.no_grad():
        clss, poss = model(torch.from_numpy(rp).float().unsqueeze(0).to(device))
    probs = [torch.softmax(c, dim=1).squeeze(0).cpu().numpy() for c in clss]
    poss = [p.squeeze(0).cpu().numpy() for p in poss]
    # 置信度 = 类别概率最大值
    confs = [float(np.max(p)) for p in probs]
    dets = []
    for pos, prob, conf in zip(poss, probs, confs):
        dets.append((pos, prob, conf))
    return dets


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    # 模型
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DetectNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[mot] 检测模型加载: {ckpt['classes']}, K={ckpt['k']}")

    # 场景
    scene = MovingTargetScene(n_targets=args.n_targets, n_frames=args.n_frames, seed=args.seed)
    print(f"[mot] 场景: {args.n_targets} 目标, 类别分布: {scene.summary()}")

    # 逐帧：检测 + 追踪
    tracker = MOTTracker(n_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)
    tracker.MAX_MISS = args.max_miss
    gt_all, track_all = [], []
    for t in range(args.n_frames):
        roi = scene.render_roi(t)
        rp = data_sat.compute_range_profile(
            roi, scene.mid["target_pos"], scene.mid["ground_pos"],
            scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
        dets_raw = run_detector(model, rp, device)
        # 置信度过滤（低置信检测会产生噪声轨迹）
        dets = [(p, prob) for p, prob, conf in dets_raw if conf >= args.conf_thr]
        tracks = tracker.update(dets)
        gt_all.append(scene.targets_at(t))
        track_all.append(tracks)
        scene.step()

    # ---- 评估 ----
    n_gt = sum(len(g) for g in gt_all)
    # 每帧：轨迹与 GT 关联（匈牙利）
    matched = 0; id_sw = 0
    gt_to_track = {}   # (gt_idx, t) -> track id（用于 ID 切换检测）
    for t, (gts, trs) in enumerate(zip(gt_all, track_all)):
        if not gts or not trs:
            continue
        cost = np.linalg.norm(
            np.array([g[1] for g in gts])[:, None, :] -
            np.array([tr[2] for tr in trs])[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= args.gate:
                matched += 1
                tid = trs[c][0]
                # ID 切换：同一 GT 在不同帧匹配到不同 track id
                key = r
                if key in gt_to_track and gt_to_track[key] != tid:
                    id_sw += 1
                gt_to_track[key] = tid

    recall = matched / max(n_gt, 1)
    print("\n" + "=" * 60)
    print("MOT 评估:")
    print(f"  检测召回（轨迹命中 GT）: {recall:.3f}")
    print(f"  ID 切换次数: {id_sw}")
    print(f"  确认轨迹数: {len(set(tr[0] for trs in track_all for tr in trs))}")
    print(f"  GT 目标数: {args.n_targets}")
    print("=" * 60)

    # 类别（轨迹多帧投票后的准确率）
    cls_ok = cls_tot = 0
    for t, (gts, trs) in enumerate(zip(gt_all, track_all)):
        for g in gts:
            # 找最近轨迹
            if not trs:
                continue
            d = [np.linalg.norm(g[1] - tr[2]) for tr in trs]
            j = int(np.argmin(d))
            if d[j] <= args.gate:
                cls_tot += 1
                cls_ok += (trs[j][1] == g[0])
    print(f"  轨迹类别准确率: {cls_ok/max(cls_tot,1):.3f} ({cls_ok}/{cls_tot})")

    # ---- 可视化（3D：无人机在空中，地面目标贴地） ----
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (altitude)")
    ax.set_title(f"MOT: {args.n_targets} Moving Targets (3D, uav in air)")
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    track_lines = {}
    gt_lines = []
    info_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top",
                          fontsize=10, family="monospace")
    # 地面参考面（z=-1 平面）
    gx, gy = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    ax.plot_surface(gx, gy, np.full_like(gx, -1.0), alpha=0.08, color="gray")

    def init():
        return []

    def update(t):
        for line in gt_lines:
            line.remove()
        gt_lines.clear()
        for tid, line in track_lines.items():
            line.remove()
        track_lines.clear()
        # GT（灰色叉，3D）
        for gid, (c, (cx, cy, cz)) in enumerate(gt_all[t]):
            pt = ax.plot([cx], [cy], [cz], "x", color="gray", ms=6, mew=1.5)
            gt_lines.append(pt[0])
        # 轨迹（彩色，按 ID，3D）
        for tid, cid, pos, conf in track_all[t]:
            color = colors[tid % 20]
            px, py, pz = pos[0], pos[1], pos[2]
            pt = ax.plot([px], [py], [pz], "o", color=color, ms=7)
            track_lines[tid] = pt[0]
            hist = [tr for tt in range(t + 1) for tr in track_all[tt] if tr[0] == tid]
            if len(hist) > 1:
                xs = [h[2][0] for h in hist]; ys = [h[2][1] for h in hist]; zs = [h[2][2] for h in hist]
                ln = ax.plot(xs, ys, zs, "-", color=color, alpha=0.5, lw=1.2)
                gt_lines.append(ln[0])
            ax.text(px + 0.03, py + 0.03, pz + 0.03, f"T{tid}", fontsize=8, color=color)
        info_text.set_text(
            f"帧 {t+1}/{len(gt_all)} · 目标 {len(gt_all[t])} · 轨迹 {len(track_all[t])} · "
            f"召回 {recall:.2f} · ID切换 {id_sw}")
        return []

    anim = FuncAnimation(fig, update, frames=args.n_frames, init_func=init,
                         blit=False, repeat=True)
    gif_path = os.path.join(OUT_DIR, "mot_animation.gif")
    anim.save(gif_path, writer=PillowWriter(fps=2.0))
    print(f"\n[mot] 3D 动画: {gif_path}")

    # 静态 3D 轨迹图
    fig2 = plt.figure(figsize=(11, 8))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1); ax2.set_zlim(-1.1, 1.1)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z (altitude)")
    ax2.set_title(f"MOT 3D Trajectories (recall={recall:.2f}, ID switches={id_sw})")
    ax2.plot_surface(gx, gy, np.full_like(gx, -1.0), alpha=0.08, color="gray")
    for gts in gt_all:
        for _, (cx, cy, cz) in gts:
            ax2.plot([cx], [cy], [cz], "x", color="gray", ms=4, mew=1, alpha=0.5)
    for trs in track_all:
        for tid, cid, pos, conf in trs:
            ax2.plot([pos[0]], [pos[1]], [pos[2]], ".", color=colors[tid % 20], ms=4)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "mot_trajectories.png"), dpi=150)
    print(f"[mot] 3D 轨迹图: {os.path.join(OUT_DIR, 'mot_trajectories.png')}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10 目标 MOT 演示")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/detect_best.pth")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--gate", type=float, default=0.3)
    parser.add_argument("--conf_thr", type=float, default=0.3)
    parser.add_argument("--max_miss", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
