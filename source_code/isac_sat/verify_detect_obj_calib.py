"""
verify_detect_obj_calib.py — S3：objectness 阈值校准（检测线收尾）

S2 的 ID 切换反弹（192 vs 基线 178）源于 sigmoid 阈值 0.3 与 max-softmax
阈值 0.3 不是同一工作点。本脚本在 held-out 场景（种子 11-13，与训练/MOT
场景不相交）上收集 objectness 分数与正确性标签，按最大 F1 选阈值，
并对基线模型做同款校准（公平对比），再跑 MOT。

预注册命题：
  S3a 校准后 S2 的 MOT ID 切换 ≤ 158（S1 最佳）且 recall ≥ 0.70
  S3b S2@校准 在 (recall, IDsw, RMSE) 三项中至少两项优于 基线@校准
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
from verify_tracker_kalman import run_tracker, evaluate
from verify_detect_count import load_model, det_conf

from scipy.optimize import linear_sum_assignment


def collect_obj_labels(args, name, seeds):
    """held-out 场景收集 (objectness 分数, 正确性标签)。"""
    model, _ = load_model(args, name)
    confs, labels = [], []
    rng = random.Random(999)
    for sd in seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        n_t = rng.randint(*args.n_targets_range) if args.n_targets_range else args.n_targets
        scene = MovingTargetScene(n_targets=n_t, n_frames=args.n_frames, seed=sd)
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            rp = data_sat.compute_range_profile(
                roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
            with torch.no_grad():
                clss, poss, _cnt, objs = model(
                    torch.from_numpy(rp).float().unsqueeze(0).to(args.device))
            gt = np.array([g[1] for g in scene.targets_at(t)])
            det_pos = np.array([p.squeeze(0).cpu().numpy() for p in poss])
            if len(gt):
                cost = np.linalg.norm(gt[:, None, :] - det_pos[None, :, :], axis=-1)
                rows, cols = linear_sum_assignment(cost)
                matched = set(c for r, c in zip(rows, cols) if cost[r, c] <= args.gate)
            else:
                matched = set()
            for k in range(len(clss)):
                confs.append(det_conf(clss[k].squeeze(0),
                                      None if objs is None else objs[k]))
                labels.append(1.0 if k in matched else 0.0)
            scene.step()
    return np.array(confs), np.array(labels)


def best_f1_threshold(confs, labels):
    """扫描阈值取最大 F1。"""
    order = np.argsort(-confs)
    c, y = confs[order], labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(c[i]), float(f1[i]), float(prec[i]), float(rec[i])


def mot_at(args, name, thr):
    model, _ = load_model(args, name)
    per = []
    rng = random.Random(999)
    for sd in args.mot_seeds:
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
                clss, poss, _cnt, objs = model(
                    torch.from_numpy(rp).float().unsqueeze(0).to(args.device))
            dets = []
            for k in range(len(clss)):
                p = det_conf(clss[k].squeeze(0), None if objs is None else objs[k])
                if p >= thr:
                    dets.append((poss[k].squeeze(0).cpu().numpy(),
                                 F.softmax(clss[k].squeeze(0), dim=0).cpu().numpy()))
            frames.append({"dets": dets, "gt": scene.targets_at(t)})
            scene.step()
        tr_all = run_tracker(frames, lambda: MOTTracker(n_classes=len(CLASS_NAMES),
                                                        class_names=CLASS_NAMES))
        per.append(evaluate(frames, tr_all, args))
    return {"rmse": float(np.mean([p["rmse"] for p in per])),
            "recall": float(np.mean([p["recall"] for p in per])),
            "idsw": float(np.mean([p["id_switches"] for p in per]))}


def main(args):
    print(f"\n{'=' * 70}")
    print("S3 objectness 阈值校准（held-out F1 选阈）")
    print(f"{'=' * 70}")

    c_new, y_new = collect_obj_labels(args, args.s2_name, args.calib_seeds)
    c_base, y_base = collect_obj_labels(args, args.base_name, args.calib_seeds)
    thr_new, f1_new, p_new, r_new = best_f1_threshold(c_new, y_new)
    thr_base, f1_base, p_base, r_base = best_f1_threshold(c_base, y_base)
    print(f"  S2   : 最优阈值 {thr_new:.3f} (F1={f1_new:.3f}, P={p_new:.3f}, R={r_new:.3f})")
    print(f"  base : 最优阈值 {thr_base:.3f} (F1={f1_base:.3f}, P={p_base:.3f}, R={r_base:.3f})")

    res = {}
    for name, ckpt, thr in [("base@0.3", args.base_name, 0.3),
                            ("base@calib", args.base_name, thr_base),
                            ("S2@0.3", args.s2_name, 0.3),
                            ("S2@calib", args.s2_name, thr_new)]:
        res[name] = mot_at(args, ckpt, thr)
        print(f"  MOT {name:>12}: RMSE={res[name]['rmse']:.4f} "
              f"recall={res[name]['recall']:.3f} IDsw={res[name]['idsw']:.1f}")

    verdicts = {
        "S3a_idsw_le_158": bool(res["S2@calib"]["idsw"] <= 158.0),
        "S3a2_recall_ge_070": bool(res["S2@calib"]["recall"] >= 0.70),
        "S3b_dominates_2of3": bool(
            (res["S2@calib"]["recall"] >= res["base@calib"]["recall"]) +
            (res["S2@calib"]["idsw"] <= res["base@calib"]["idsw"]) +
            (res["S2@calib"]["rmse"] <= res["base@calib"]["rmse"]) >= 2),
    }
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"thr_s2": thr_new, "f1_s2": f1_new, "thr_base": thr_base,
           "f1_base": f1_base, "mot": res, "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "obj_calib.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 objectness 阈值校准")
    parser.add_argument("--ckpt_dir", type=str, default="./isac_demo")
    parser.add_argument("--base_name", type=str, default="detect_best_d2.pth")
    parser.add_argument("--s2_name", type=str, default="detect_best_s2.pth")
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
