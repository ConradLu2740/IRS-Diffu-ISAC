"""
verify_detect_multiframe.py — F1：多帧融合检测（MOT 结论指向的"新观测"方向）

背景（§7.17）：MOT 六层诊断结论——检测器内部调优已到极限，进一步收益需要
**新观测**。本实验检验最便宜的"新观测"：连续 M=4 帧距离像堆叠作为检测器
输入（时间上下文：跨帧一致的目标峰 vs 逐帧独立的噪声/虚警）。

预注册命题：
  F1 多帧输入的 P@R=0.67 ≥ 0.65（单帧 D2 为 0.573）
  F2 MOT recall ≥ 0.70 且 ID 切换 ≤ 0.9× 单帧
  F3 若 F1 失败 ⇒ 单帧精度极限不是噪声型虚警导致（时间上下文无效），
     记录幅度后转向双站/多帧复观测方向
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
from train_detect import DetectNet, K_MAX
from mot_tracker import MOTTracker
from verify_tracker_kalman import run_tracker, evaluate

from scipy.optimize import linear_sum_assignment


def load_model(args, name):
    ckpt = torch.load(os.path.join(args.ckpt_dir, name), map_location=args.device)
    model = DetectNet(count_head=bool(ckpt.get("count_head", False)),
                      stack=int(ckpt.get("stack", 1))).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt


def scene_dets(args, seeds, name, stack, thr=0.3, n_avg=1):
    """收集检测（stack>1 多帧堆叠；n_avg>1 同帧多 realizing 平均）。"""
    model, _ = load_model(args, name)
    seqs = []
    rng = random.Random(999)
    for sd in seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        n_t = rng.randint(*args.n_targets_range) if args.n_targets_range else args.n_targets
        scene = MovingTargetScene(n_targets=n_t, n_frames=args.n_frames, seed=sd)
        buf = []
        frames = []
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            acc = None
            for a in range(n_avg):
                rp = data_sat.compute_range_profile(
                    roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                    scene.scenario.wavelength_m, snr_db=args.snr_db,
                    seed=t * 100 + a, align=False)
                acc = rp if acc is None else acc + rp
            rp = acc / n_avg
            buf.append(rp)
            if len(buf) > stack:
                buf.pop(0)
            x = np.stack(buf, axis=0) if len(buf) == stack else \
                np.stack([buf[0]] * (stack - len(buf)) + buf, axis=0)
            with torch.no_grad():
                clss, poss, _cnt = model(torch.from_numpy(x).float().unsqueeze(0).to(args.device))
            dets = []
            for k in range(len(clss)):
                lg = clss[k].squeeze(0)
                p = float(F.softmax(lg, dim=0).max())
                if p >= thr:
                    dets.append((poss[k].squeeze(0).cpu().numpy(),
                                 F.softmax(lg, dim=0).cpu().numpy()))
            frames.append({"dets": dets, "gt": scene.targets_at(t)})
            scene.step()
        seqs.append(frames)
    return seqs


def pr_at_recall(seqs, target=0.67):
    confs, labels = [], []
    for frames in seqs:
        for fr in frames:
            gt = np.array([g[1] for g in fr["gt"]])
            if not fr["dets"]:
                continue
            dt = np.array([d[0] for d in fr["dets"]])
            cost = np.linalg.norm(gt[:, None, :] - dt[None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(cost)
            matched = set(c for r, c in zip(rows, cols) if cost[r, c] <= 0.35)
            for j, d in enumerate(fr["dets"]):
                confs.append(float(np.max(d[1])))
                labels.append(1.0 if j in matched else 0.0)
    if not labels:
        return 0.0
    confs = np.array(confs); labels = np.array(labels)
    n_pos = max(labels.sum(), 1)
    order = np.argsort(-confs)
    tp = np.cumsum(labels[order]); fp = np.cumsum(1 - labels[order])
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.where(recall >= target)[0]
    return float(precision[idx[0]]) if len(idx) else 0.0


def main(args):
    print(f"\n{'=' * 70}")
    print("F1 多帧融合检测（stack=4 vs 单帧，变计数场景 5-12）")
    print(f"{'=' * 70}")

    seqs_single = scene_dets(args, args.calib_seeds, args.base_name, 1)
    if args.mode == "diversity":
        seqs_multi = scene_dets(args, args.calib_seeds, args.f1_name, 1,
                                n_avg=args.stack)
    else:
        seqs_multi = scene_dets(args, args.calib_seeds, args.f1_name, args.stack)
    p_single = pr_at_recall(seqs_single)
    p_multi = pr_at_recall(seqs_multi)
    _mode_tag = f"n_avg={args.stack}" if args.mode == "diversity" else f"stack={args.stack}"
    print(f"P@R=0.67: 基线 {p_single:.3f} → {_mode_tag} {p_multi:.3f}（目标 ≥0.65）")

    mot_single = scene_dets(args, args.mot_seeds, args.base_name, 1)
    if args.mode == "diversity":
        mot_multi = scene_dets(args, args.mot_seeds, args.f1_name, 1, n_avg=args.stack)
    else:
        mot_multi = scene_dets(args, args.mot_seeds, args.f1_name, args.stack)
    res = {}
    for name, seqs in [("single", mot_single), ("multi", mot_multi)]:
        per = []
        for frames in seqs:
            tr = run_tracker(frames, lambda: MOTTracker(n_classes=len(CLASS_NAMES),
                                                        class_names=CLASS_NAMES))
            per.append(evaluate(frames, tr, args))
        res[name] = {"rmse": float(np.mean([p["rmse"] for p in per])),
                     "recall": float(np.mean([p["recall"] for p in per])),
                     "idsw": float(np.mean([p["id_switches"] for p in per]))}
        print(f"  MOT {name:>7}: RMSE={res[name]['rmse']:.4f} "
              f"recall={res[name]['recall']:.3f} IDsw={res[name]['idsw']:.1f}")

    verdicts = {
        "F1_precision_ge_065": bool(p_multi >= 0.65),
        "F1b_gain_over_single": bool(p_multi >= p_single + 0.05),
        "F2_mot_recall_ge_070": bool(res["multi"]["recall"] >= 0.70),
        "F2b_idsw_le_090x": bool(res["multi"]["idsw"] <= 0.9 * res["single"]["idsw"]),
    }
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"p_single": p_single, "p_multi": p_multi, "mot": res,
           "stack": args.stack, "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "detect_multiframe.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 多帧融合检测")
    parser.add_argument("--ckpt_dir", type=str, default="./isac_demo")
    parser.add_argument("--base_name", type=str, default="detect_best_d2.pth")
    parser.add_argument("--f1_name", type=str, default="detect_best_f1.pth")
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--mode", choices=["stack", "diversity"], default="stack")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=40)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--n_targets_range", nargs=2, type=int, default=[5, 12])
    parser.add_argument("--calib_seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--mot_seeds", nargs="+", type=int, default=[7, 8, 9])
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
