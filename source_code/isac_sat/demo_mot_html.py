"""
demo_mot_html.py — 生成 3D 交互式 MOT 演示（Plotly.js 单文件 HTML）

用 Plotly.js CDN 渲染 3D 场景：
  - 灰色点：真实目标（每帧）
  - 彩色点 + 轨迹线：MOT 追踪轨迹（按 ID）
  - 交互：拖拽旋转 / 缩放 / 悬停查看

用法：
  python demo_mot_html.py [--checkpoint ./isac_demo/detect_best.pth]
  输出：./isac_demo/mot_3d.html（浏览器打开）
"""

import os
import json
import argparse
import numpy as np
import random
import torch

import data_sat
from mot_data import MovingTargetScene, CLASS_NAMES
from mot_tracker import MOTTracker
from train_detect import DetectNet

OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DetectNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scene = MovingTargetScene(n_targets=args.n_targets, n_frames=args.n_frames, seed=args.seed)
    tracker = MOTTracker(n_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)
    tracker.MAX_MISS = args.max_miss

    gt_all, tr_all = [], []
    for t in range(args.n_frames):
        roi = scene.render_roi(t)
        rp = data_sat.compute_range_profile(
            roi, scene.mid["target_pos"], scene.mid["ground_pos"],
            scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
        with torch.no_grad():
            clss, poss = model(torch.from_numpy(rp).float().unsqueeze(0))
        dets = []
        for c, p in zip(clss, poss):
            prob = torch.softmax(c, dim=1).squeeze(0).numpy()
            if float(np.max(prob)) >= args.conf_thr:
                dets.append((p.squeeze(0).numpy(), prob))
        gt_all.append(scene.targets_at(t))
        tr_all.append(tracker.update(dets))
        scene.step()

    # ---- 构建 Plotly traces ----
    traces = []
    colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
              "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
              "#9a6324", "#800000", "#808000", "#000075", "#808080", "#ffffff"]
    # 1. GT 轨迹（灰，细线）
    for gid in range(len(gt_all[0])):
        xs = [gt_all[t][gid][1][0] for t in range(args.n_frames)]
        ys = [gt_all[t][gid][1][1] for t in range(args.n_frames)]
        zs = [gt_all[t][gid][1][2] for t in range(args.n_frames)]
        traces.append({"type": "scatter3d", "mode": "lines", "name": f"GT{gid}",
                       "x": xs, "y": ys, "z": zs, "line": {"color": "#888", "width": 2}})
    # 2. MOT 轨迹（彩色，线 + 点）
    for tid, color in [(i, colors[i % len(colors)]) for i in range(20)]:
        pts = [(t, tr) for t, trs in enumerate(tr_all) for tr in trs if tr[0] == tid]
        if not pts:
            continue
        xs = [tr[2][0] for _, tr in pts]; ys = [tr[2][1] for _, tr in pts]
        zs = [tr[2][2] for _, tr in pts]
        cls_name = CLASS_NAMES[pts[0][1][1]]
        traces.append({"type": "scatter3d", "mode": "lines+markers", "name": f"T{tid}:{cls_name}",
                       "x": xs, "y": ys, "z": zs,
                       "line": {"color": color, "width": 4},
                       "marker": {"size": 5, "color": color}})
    # 3. 地面参考点（z 最低层）
    traces.append({"type": "scatter3d", "mode": "markers",
                   "name": "ground", "x": [-1, 1], "y": [-1, 1], "z": [-1.02, -1.02],
                   "marker": {"size": 2, "color": "#333", "opacity": 0.3}})

    layout = {
        "title": {"text": f"3D MOT: {args.n_targets} Moving Targets (uav in air, {args.n_frames} frames)"},
        "scene": {"xaxis": {"title": "x"}, "yaxis": {"title": "y"},
                  "zaxis": {"title": "z (altitude)"},
                  "aspectmode": "cube"},
        "showlegend": True,
        "height": 700,
    }
    html = render_html(json.dumps({"traces": traces, "layout": layout}))
    out_path = os.path.join(OUT_DIR, "mot_3d.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[mot_html] 3D 交互演示: {out_path}")
    print(f"[mot_html] 轨迹数: {len(traces)-1} (含 GT), 帧数: {args.n_frames}")


def render_html(data_js):
    return HTML_TEMPLATE.replace("__DATA__", data_js)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>3D 多目标追踪演示</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif;
         background: #0f1420; color: #e8ecf4; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 4px 0; color: #7fd1ff; }
  .tip { font-size: 13px; color: #9aa7bd; margin: 6px 0 12px; }
  #plot { width: 100%; height: 720px; }
</style>
</head>
<body>
  <h1>🛰️ 3D 多目标追踪（MOT）· 10 个移动目标</h1>
  <div class="tip">灰色 = 真实目标轨迹 · 彩色 = 追踪轨迹（按 ID）· 拖拽旋转 / 滚轮缩放 / 悬停查看</div>
  <div id="plot"></div>
<script>
const DATA = __DATA__;
Plotly.newPlot('plot', DATA.traces, DATA.layout, {responsive: true});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 3D 交互式 MOT 演示 HTML")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/detect_best.pth")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--conf_thr", type=float, default=0.3)
    parser.add_argument("--max_miss", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
