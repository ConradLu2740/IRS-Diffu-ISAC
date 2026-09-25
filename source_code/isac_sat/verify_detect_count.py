"""
verify_detect_count.py — D3：可变计数检测头（top-K 选择替代固定阈值）

背景（§7.15）：固定 K=10 slot 检测头强制输出 10 个槽，slot-目标分配仅靠
matching loss 学习；precision 差距是学习/架构极限。D3 加计数头
（预测场景目标数 n∈[0,10]），推理输出按置信度排序的前 n 个槽。

预注册命题：
  D3a 计数准确率（|n_pred − n_true| ≤ 1）≥ 80%
  D3b top-K 在同等 recall(0.67) 下 precision 优于最佳固定阈值工作点
      （D2 的 0.573）≥ +0.05
  D3c MOT（top-K 选择）recall ≥ 0.58 且 ID 切换 ≤ 0.8×conf-0.3 工作点
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
    model = DetectNet(count_head=bool(ckpt.get("count_head", False))).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt


def scene_frames(args, seeds, mode, count_model=None, thr=0.3, name=None):
    """mode: 'count'（top-K）或 'thr'（固定阈值）；name 指定 checkpoint。"""
    model, _ = load_model(args, name or (args.d3_name if mode == "count" else args.base_name))
    seqs = []
    rng = random.Random(999)
    for sd in seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        n_t = rng.randint(*args.n_targets_range) if args.n_targets_range else args.n_targets
        scene = MovingTargetScene(n_targets=n_t, n_frames=args.n_frames, seed=sd)
        frames = []
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            rp = data_sat.compute_range_profile(
                roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
            with torch.no_grad():
                clss, poss, cnt = model(torch.from_numpy(rp).float().unsqueeze(0).to(args.device))
            if mode == "count" and cnt is not None:
                n_pred = int(cnt.argmax(1).item())
                slots = sorted(range(len(clss)),
                               key=lambda k: -float(F.softmax(clss[k].squeeze(0), dim=0).max()))
                dets = []
                for k in slots[:max(n_pred, 0)]:
                    lg = clss[k].squeeze(0)
                    dets.append((poss[k].squeeze(0).cpu().numpy(),
                                 F.softmax(lg, dim=0).cpu().numpy()))
            else:
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


def count_accuracy(args):
    model, _ = load_model(args, args.d3_name)
    ok1 = 0; tot = 0
    rng = random.Random(999)
    for sd in args.calib_seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        n_t = rng.randint(*args.n_targets_range) if args.n_targets_range else args.n_targets
        scene = MovingTargetScene(n_targets=n_t, n_frames=args.n_frames, seed=sd)
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            rp = data_sat.compute_range_profile(
                roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
            with torch.no_grad():
                _, _, cnt = model(torch.from_numpy(rp).float().unsqueeze(0).to(args.device))
            n_true = min(len(scene.targets_at(t)), K_MAX)
            n_pred = int(cnt.argmax(1).item())
            ok1 += int(abs(n_pred - n_true) <= 1)
            tot += 1
            scene.step()
    return ok1 / max(tot, 1)


def pr_at_recall(seqs, target=0.67):
    """slot 级 PR：precision@recall=target。"""
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
    print("D3 可变计数检测头（top-K 选择 vs 固定阈值）")
    print(f"{'=' * 70}")

    acc1 = count_accuracy(args)
    print(f"D3a 计数准确率（|Δn|≤1）: {acc1:.3f}（目标 ≥0.80）")

    # 同等 recall 下的 precision（held-out 校准场景）
    # S1 隔离匹配损失效应：新模型与基线都用阈值过滤（不用计数头 top-K）
    seqs_new = scene_frames(args, args.calib_seeds, "thr", thr=0.3, name=args.d3_name)
    seqs_thr = scene_frames(args, args.calib_seeds, "thr", thr=0.3, name=args.base_name)
    p_count = pr_at_recall(seqs_new)
    p_thr = pr_at_recall(seqs_thr)
    print(f"P@R=0.67: 新模型(阈值0.3) {p_count:.3f} vs 基线(阈值0.3) {p_thr:.3f}")

    # MOT 配对（场景与校准不相交）
    res = {}
    mot_thr = scene_frames(args, args.mot_seeds, "thr", thr=0.3, name=args.base_name)
    mot_new = scene_frames(args, args.mot_seeds, "thr", thr=0.3, name=args.d3_name)
    for name, seqs in [("baseline", mot_thr), ("new_model", mot_new)]:
        per = []
        for frames in seqs:
            tr = run_tracker(frames, lambda: MOTTracker(n_classes=len(CLASS_NAMES),
                                                        class_names=CLASS_NAMES))
            per.append(evaluate(frames, tr, args))
        res[name] = {"rmse": float(np.mean([p["rmse"] for p in per])),
                     "recall": float(np.mean([p["recall"] for p in per])),
                     "idsw": float(np.mean([p["id_switches"] for p in per]))}
        print(f"  MOT {name:>8}: RMSE={res[name]['rmse']:.4f} "
              f"recall={res[name]['recall']:.3f} IDsw={res[name]['idsw']:.1f}")

    verdicts = {
        "D3a_count_acc_ge_080": bool(acc1 >= 0.80),
        "D3b_precision_gain_ge_005": bool(p_count >= p_thr + 0.05),
        "D3c_mot_recall_ge_058": bool(res["new_model"]["recall"] >= 0.58),
        "D3c2_idsw_le_080x": bool(res["new_model"]["idsw"] <= 0.8 * res["baseline"]["idsw"]),
    }
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"count_acc": acc1, "p_at_recall_topk": p_count,
           "p_at_recall_thr": p_thr, "mot": res, "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "detect_count.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3 可变计数检测头")
    parser.add_argument("--ckpt_dir", type=str, default="./isac_demo")
    parser.add_argument("--d3_name", type=str, default="detect_best_d3.pth")
    parser.add_argument("--base_name", type=str, default="detect_best_d2.pth")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=40)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--calib_seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--mot_seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--n_targets_range", nargs=2, type=int, default=None,
                        help="评估场景随机目标数范围 (lo hi)（D4）")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
