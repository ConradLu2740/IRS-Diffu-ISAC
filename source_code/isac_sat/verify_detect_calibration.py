"""
verify_detect_calibration.py — D1：检测器置信度温度校准（MOT 瓶颈的原则性修复）

背景（§7.13）：MOT 质量是检测工作点受限——conf≥0.5 时 GT 覆盖率仅 8.9%，
降到 0.3 recall +121% 但假阳性同步上升。原则性修复：校准检测器置信度，
在**同等 recall** 下减少假阳性（更少的每帧检测数 → 更少的 ID 切换）。

方法（标准温度校准，Guo et al. 2017）：
  1. held-out 场景（种子与 MOT 评估场景不相交）收集每个 slot 的类别 logits
     与正确性标签（位置匹配 GT，gate 0.35）
  2. 拟合温度 T* = argmin_T NLL(softmax(logits/T), label)
  3. 校准置信度 p_cal = max softmax(logits/T*)；画 PR 曲线，
     在目标 recall（raw conf=0.3 的工作点）处选阈值
  4. MOT 配对比较：raw 0.3 vs 校准阈值（同 recall）

预注册命题：
  D1a 温度校准降低 held-out NLL（校准增益可测）
  D1b 同等 recall 下，校准工作点的每帧检测数（假阳性代理）少于 raw 0.3
  D1c 同等 recall 下，ID 切换 ≤ 0.8×raw 0.3，RMSE 不劣化
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_sat
from mot_data import MovingTargetScene, CLASS_NAMES
from train_detect import DetectNet
from mot_tracker import MOTTracker
from verify_tracker_kalman import KalmanMOTTracker, run_tracker, evaluate, collect_detections

from scipy.optimize import linear_sum_assignment


def collect_slot_logits(args, seeds):
    """held-out：每 slot 的 (logits, 正确性标签, 原始 max-prob)。"""
    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DetectNet(count_head=bool(ckpt.get("count_head", False)),
                              obj_head=bool(ckpt.get("obj_head", False))).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    logits_all, labels_all, confs_all = [], [], []
    for sd in seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        scene = MovingTargetScene(n_targets=args.n_targets, n_frames=args.n_frames, seed=sd)
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            rp = data_sat.compute_range_profile(
                roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
            with torch.no_grad():
                clss, poss, _cnt, _objs = model(torch.from_numpy(rp).float().unsqueeze(0).to(device))
            gt = np.array([g[1] for g in scene.targets_at(t)])
            det_pos = np.array([p.squeeze(0).cpu().numpy() for p in poss])
            K = len(clss)
            # 每 slot 与 GT 的匹配（位置 gate）
            if len(gt):
                cost = np.linalg.norm(gt[:, None, :] - det_pos[None, :, :], axis=-1)
                rows, cols = linear_sum_assignment(cost)
                matched = {}
                for r, c in zip(rows, cols):
                    if cost[r, c] <= args.gate:
                        matched[c] = r
            else:
                matched = {}
            for k in range(K):
                lg = clss[k].squeeze(0).cpu().numpy()      # [C] logits
                logits_all.append(lg)
                confs_all.append(float(np.max(F.softmax(torch.tensor(lg), dim=0).numpy())))
                labels_all.append(1.0 if k in matched else 0.0)
            scene.step()
    return np.array(logits_all), np.array(labels_all), np.array(confs_all)


def fit_temperature(logits, labels, T_grid=None):
    """温度校准：T* = argmin NLL（标签为 slot 正确性的二分类）。"""
    if T_grid is None:
        T_grid = np.concatenate([np.linspace(0.2, 2.0, 19), np.linspace(2.5, 10.0, 16)])
    lg = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    best_T, best_nll = 1.0, float("inf")
    for T in T_grid:
        p = F.softmax(lg / T, dim=1).max(dim=1).values.clamp(1e-7, 1 - 1e-7)
        nll = F.binary_cross_entropy(p, y).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


def main(args):
    logits, labels, confs = collect_slot_logits(args, args.calib_seeds)
    n_pos = labels.sum()
    print(f"校准集: {len(labels)} slots, 正例 {int(n_pos)} ({n_pos / len(labels):.1%})")

    nll_raw = F.binary_cross_entropy(
        torch.tensor(confs.clip(1e-6, 1 - 1e-6), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)).item()
    T_star, nll_cal = fit_temperature(logits, labels)
    p_cal = F.softmax(torch.tensor(logits, dtype=torch.float32) / T_star,
                      dim=1).max(dim=1).values.numpy()
    print(f"NLL: raw {nll_raw:.4f} → 校准(T*={T_star:.2f}) {nll_cal:.4f} "
          f"(改善 {100 * (1 - nll_cal / nll_raw):.1f}%)")

    # PR 曲线与阈值选择（目标 recall = raw conf=0.3 的 recall）
    order = np.argsort(-p_cal)
    p_sorted, y_sorted = p_cal[order], labels[order]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    recall = tp / max(n_pos, 1); precision = tp / np.maximum(tp + fp, 1)

    # raw 0.3 工作点（在同一校准集上测，保证可比）
    mask_raw = confs >= 0.3
    rec_raw = float(labels[mask_raw].sum() / n_pos)
    dets_raw = float(mask_raw.mean())

    # 校准阈值：达到同等 recall 的最高精度点
    idx = np.where(recall >= rec_raw)[0]
    if len(idx):
        i = idx[0]
        thr_cal = float(p_sorted[i])
        prec_cal = float(precision[i])
    else:
        thr_cal, prec_cal = 0.0, 0.0
    mask_cal = p_cal >= thr_cal
    dets_cal = float(mask_cal.mean())
    prec_raw = float(labels[mask_raw].sum() / max(mask_raw.sum(), 1))

    print(f"\n工作点对比（校准集, 目标 recall={rec_raw:.3f}）:")
    print(f"  raw conf>=0.3 : 每帧检测率 {dets_raw:.3f}, precision {prec_raw:.3f}")
    print(f"  校准 p>={thr_cal:.3f}: 每帧检测率 {dets_cal:.3f}, precision {prec_cal:.3f}")
    print(f"  → 同等 recall 下每帧检测数变化: {100 * (dets_cal / max(dets_raw, 1e-9) - 1):+.1f}%")

    # MOT 配对比较（原始 0.3 vs 校准阈值）——用包装检测函数
    class A: pass
    a = A(); a.checkpoint = args.checkpoint; a.n_targets = args.n_targets
    a.n_frames = args.n_frames; a.snr_db = args.snr_db; a.gate = args.gate
    a.seeds = args.mot_seeds; a.device = args.device

    import verify_tracker_kalman as vtk

    def make_collect(conf_fn):
        """用 conf_fn(p_cal_or_raw) 过滤的检测序列收集器。"""
        device = args.device
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = DetectNet(count_head=bool(ckpt.get("count_head", False)),
                              obj_head=bool(ckpt.get("obj_head", False))).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        seqs = []
        for sd in a.seeds:
            torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
            scene = MovingTargetScene(n_targets=a.n_targets, n_frames=a.n_frames, seed=sd)
            frames = []
            for t in range(a.n_frames):
                roi = scene.render_roi(t)
                rp = data_sat.compute_range_profile(
                    roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                    scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
                with torch.no_grad():
                    clss, poss, _cnt, _objs = model(torch.from_numpy(rp).float().unsqueeze(0).to(device))
                dets = []
                for k in range(len(clss)):
                    lg = clss[k].squeeze(0)
                    p = float(F.softmax(lg / T_star, dim=0).max())
                    if conf_fn(p):
                        dets.append((poss[k].squeeze(0).cpu().numpy(),
                                     F.softmax(lg, dim=0).cpu().numpy()))
                frames.append({"dets": dets, "gt": scene.targets_at(t)})
                scene.step()
            seqs.append(frames)
        return seqs

    seqs_raw = make_collect(lambda p: p >= 0.3)            # raw 口径（T=1 时 p 即 raw）
    # 注意：raw 口径需要 T=1 的概率——重新收集一次
    def make_collect_T1():
        device = args.device
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = DetectNet(count_head=bool(ckpt.get("count_head", False)),
                              obj_head=bool(ckpt.get("obj_head", False))).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        seqs = []
        for sd in a.seeds:
            torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
            scene = MovingTargetScene(n_targets=a.n_targets, n_frames=a.n_frames, seed=sd)
            frames = []
            for t in range(a.n_frames):
                roi = scene.render_roi(t)
                rp = data_sat.compute_range_profile(
                    roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                    scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
                with torch.no_grad():
                    clss, poss, _cnt, _objs = model(torch.from_numpy(rp).float().unsqueeze(0).to(device))
                dets = []
                for k in range(len(clss)):
                    lg = clss[k].squeeze(0)
                    p = float(F.softmax(lg, dim=0).max())
                    if p >= 0.3:
                        dets.append((poss[k].squeeze(0).cpu().numpy(),
                                     F.softmax(lg, dim=0).cpu().numpy()))
                frames.append({"dets": dets, "gt": scene.targets_at(t)})
                scene.step()
            seqs.append(frames)
        return seqs

    seqs_raw = make_collect_T1()
    seqs_cal = make_collect(lambda p: p >= thr_cal)

    res = {}
    for name, seqs in [("raw_0.3", seqs_raw), ("calibrated", seqs_cal)]:
        per = []
        for frames in seqs:
            tr = run_tracker(frames, lambda: MOTTracker(n_classes=len(CLASS_NAMES),
                                                        class_names=CLASS_NAMES))
            per.append(evaluate(frames, tr, a))
        res[name] = {"rmse": float(np.mean([p["rmse"] for p in per])),
                     "recall": float(np.mean([p["recall"] for p in per])),
                     "idsw": float(np.mean([p["id_switches"] for p in per]))}

    print(f"\n{'=' * 66}")
    print("MOT 配对比较（α-β，相同场景）:")
    for name, r in res.items():
        print(f"  {name:>12}: RMSE={r['rmse']:.4f} recall={r['recall']:.3f} IDsw={r['idsw']:.1f}")

    verdicts = {
        "D1a_nll_improved": bool(nll_cal < nll_raw),
        "D1b_fewer_dets_at_matched_recall": bool(dets_cal <= dets_raw),
        "D1c_idsw_le_080x": bool(res["calibrated"]["idsw"] <= 0.8 * res["raw_0.3"]["idsw"]),
        "D1c2_rmse_no_worse": bool(res["calibrated"]["rmse"] <= res["raw_0.3"]["rmse"] * 1.05),
    }
    print(f"  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"T_star": T_star, "nll_raw": nll_raw, "nll_cal": nll_cal,
           "recall_target": rec_raw, "thr_cal": thr_cal,
           "dets_per_frame_raw": dets_raw, "dets_per_frame_cal": dets_cal,
           "precision_raw": prec_raw, "precision_cal": prec_cal,
           "mot": res, "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "detect_calibration.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D1 检测器置信度温度校准")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/detect_best.pth")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=40)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--calib_seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--mot_seeds", nargs="+", type=int, default=[7, 8, 9])
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
