"""
verify_tracker_kalman.py — MOT 跟踪器升级：α-β → CRB 一致噪声的卡尔曼滤波

预注册命题（roadmap §4.3-D4 跟踪器证书）：
  T1 现有 α-β（α=0.6/β=0.2 固定增益）次优：其等效增益与噪声比不匹配
  T2 KF（R 取实测检测噪声、Q 取场景运动学）位置 RMSE ≤ α-β（预测 −10~−30%）
  T3 η_track = CRB_ss / RMSE² ≥ 0.9（KF 对其噪声模型近优）
  T4 ID 切换与召回不劣化（公平改进）

协议：与 demo_mot.py 同一检测管线（detect_best.pth + MovingTargetScene），
多个种子；两个跟踪器吃**相同**的检测序列（配对比较）。
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_sat
from mot_data import MovingTargetScene, CLASS_NAMES
from mot_tracker import MOTTracker, Track, MAX_MISS, CONFIRM_FRAMES, GATE
from train_detect import DetectNet
from demo_mot import run_detector

from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------
# 卡尔曼轨迹（CV 模型，2D x-y；接口与 Track 兼容）
# ----------------------------------------------------------------------

class KalmanTrack(Track):
    """恒定速度 Kalman 滤波轨迹（替代 α-β 平滑）。"""

    def __init__(self, tid, pos, cls_probs, n_classes, R=None, Q=None):
        super().__init__(tid, pos, cls_probs, n_classes)
        self.state = np.array([pos[0], pos[1], 0.0, 0.0])     # [x, y, vx, vy]
        self.P = np.eye(4) * 0.05
        self.R = R if R is not None else np.eye(2) * 0.01
        self.Q = Q if Q is not None else np.eye(4) * 1e-4
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.F = np.array([[1.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.P + self.Q if False else self.F @ self.P @ self.F.T + self.Q
        return np.array([self.state[0], self.state[1], self.pos[2]])

    def update(self, det_pos, det_cls_probs):
        z = np.asarray(det_pos[:2], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - self.H @ self.state)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.pos = np.array([self.state[0], self.state[1], float(det_pos[2])])
        self.cls_probs = 0.7 * self.cls_probs + 0.3 * np.asarray(det_cls_probs)
        self.age += 1
        self.miss = 0
        self.confirmed = True if self.age >= CONFIRM_FRAMES else self.confirmed
        self.history.append(self.pos.copy())

    def miss_frame(self):
        self.miss += 1
        self.predict()


class KalmanMOTTracker(MOTTracker):
    """MOTTracker 的卡尔曼变体（关联/清理逻辑不变）。"""

    def __init__(self, n_classes, class_names=None, ground_z=-0.5, R=None, Q=None):
        super().__init__(n_classes, class_names, ground_z)
        self._R, self._Q = R, Q

    def _new_track(self, det_pos, det_cls_probs):
        t = KalmanTrack(self._next_id, det_pos, det_cls_probs, self.n_classes,
                        R=self._R, Q=self._Q)
        self._next_id += 1
        self.tracks.append(t)
        return t


# ----------------------------------------------------------------------
# 实验
# ----------------------------------------------------------------------

class ImprovedMOTTracker(KalmanMOTTracker):
    """关联层改进（pre-registered A1-A3）：
      1. 马氏距离门控（χ² 检验，用 KF 预测协方差 S）替代固定欧氏门限
      2. 第二轮补救关联（未匹配轨迹 × 未匹配检测，放宽 χ² 门限）
    """

    CHI2_TIGHT = 5.991      # 2 自由度 χ²@0.95
    CHI2_LOOSE = 9.210      # 2 自由度 χ²@0.99

    def update(self, detections):
        if not self.tracks or len(detections) == 0:
            for d in detections:
                self._new_track(*d)
        else:
            preds = [t.predict() for t in self.tracks]
            dets = np.array([d[0] for d in detections])
            S_inv = []
            for t in self.tracks:
                S = t.H @ t.P @ t.H.T + t.R
                S_inv.append(np.linalg.inv(S))
            # 马氏距离矩阵
            D = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j in range(len(detections)):
                    dz = dets[j, :2] - preds[i][:2]
                    D[i, j] = float(dz @ S_inv[i] @ dz)
            rows, cols = linear_sum_assignment(D)
            matched_t, matched_d = set(), set()
            for r, c in zip(rows, cols):
                if D[r, c] <= self.CHI2_TIGHT:
                    self.tracks[r].update(*detections[c])
                    matched_t.add(r); matched_d.add(c)
            # 第二轮补救关联（放宽门限）
            rem_t = [i for i in range(len(self.tracks)) if i not in matched_t]
            rem_d = [j for j in range(len(detections)) if j not in matched_d]
            if rem_t and rem_d:
                sub = D[np.ix_(rem_t, rem_d)]
                r2, c2 = linear_sum_assignment(sub)
                for a, b in zip(r2, c2):
                    if sub[a, b] <= self.CHI2_LOOSE:
                        self.tracks[rem_t[a]].update(*detections[rem_d[b]])
                        matched_t.add(rem_t[a]); matched_d.add(rem_d[b])
            for c in range(len(detections)):
                if c not in matched_d:
                    self._new_track(*detections[c])
            for r in range(len(self.tracks)):
                if r not in matched_t:
                    self.tracks[r].miss_frame()

        self.tracks = [t for t in self.tracks if t.miss <= MAX_MISS]
        for t in self.tracks:
            if np.argmax(t.cls_probs) not in self.air_cls:
                t.pos[2] = self.ground_z
                if t.history:
                    t.history[-1][2] = self.ground_z
        out = []
        for t in self.tracks:
            if t.confirmed:
                out.append((t.id, int(np.argmax(t.cls_probs)),
                            t.pos.copy(), float(np.max(t.cls_probs))))
        return out



def collect_detections(args, seeds):
    """跑检测管线，返回每帧 (dets, gt)。dets 与 gt 配对供两个跟踪器共用。"""
    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DetectNet().to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    seqs = []
    for sd in seeds:
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        scene = MovingTargetScene(n_targets=args.n_targets, n_frames=args.n_frames, seed=sd)
        frames = []
        for t in range(args.n_frames):
            roi = scene.render_roi(t)
            rp = data_sat.compute_range_profile(
                roi, scene.mid["target_pos"], scene.mid["ground_pos"],
                scene.scenario.wavelength_m, snr_db=args.snr_db, seed=t, align=False)
            dets_raw = run_detector(model, rp, device)
            dets = [(p, prob) for p, prob, conf in dets_raw if conf >= args.conf_thr]
            frames.append({"dets": dets, "gt": scene.targets_at(t)})
            scene.step()
        seqs.append(frames)
    return seqs


def run_tracker(frames, make_tracker):
    tracker = make_tracker()
    out = []
    for fr in frames:
        tracks = tracker.update(fr["dets"])
        out.append(tracks)
    return out


def evaluate(frames, track_all, args):
    """位置 RMSE / 召回 / ID 切换（与 demo_mot.py 同口径）。"""
    n_gt = sum(len(f["gt"]) for f in frames)
    matched = 0; id_sw = 0; sq_err = []
    gt_to_track = {}
    for t, (fr, trs) in enumerate(zip(frames, track_all)):
        if not fr["gt"] or not trs:
            continue
        gt_pos = np.array([g[1] for g in fr["gt"]])
        tr_pos = np.array([tr[2] for tr in trs])
        cost = np.linalg.norm(gt_pos[:, None, :] - tr_pos[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= args.gate:
                matched += 1
                sq_err.append(cost[r, c] ** 2)
                tid = trs[c][0]
                if r in gt_to_track and gt_to_track[r] != tid:
                    id_sw += 1
                gt_to_track[r] = tid
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err else float("nan")
    return {"rmse": rmse, "recall": matched / max(n_gt, 1), "id_switches": id_sw}


def estimate_noise(seqs, args):
    """从检测 vs GT 估计测量噪声 R（各向同性：两轴方差均值）。"""
    errs = []
    for frames in seqs:
        for fr in frames:
            if not fr["gt"] or not fr["dets"]:
                continue
            gt = np.array([g[1] for g in fr["gt"]])
            dt = np.array([d[0] for d in fr["dets"]])
            cost = np.linalg.norm(gt[:, None, :] - dt[None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= args.gate:
                    errs.append(dt[c][:2] - gt[r][:2])
    if not errs:
        return np.eye(2) * 0.01
    E = np.array(errs)
    R = np.eye(2) * float(np.mean(E.var(axis=0)))
    return R


def estimate_process_noise(seqs):
    """从 GT 轨迹估计 CV 残差（加速度噪声）→ Q（离散白噪声加速度模型）。"""
    accs = []
    for frames in seqs:
        # 每目标按帧对齐的近似：用相邻帧 GT 质心的二阶差分
        pos_seq = []
        for fr in frames:
            if fr["gt"]:
                pos_seq.append(np.array([g[1][:2] for g in fr["gt"]]))
        if len(pos_seq) >= 3:
            P = np.array(pos_seq)                    # [T, K, 2]
            for k in range(P.shape[1]):
                a = P[2:, k, :] - 2 * P[1:-1, k, :] + P[:-2, k, :]
                accs.append(a.reshape(-1, 2))
    if not accs:
        return np.eye(4) * 1e-4
    A = np.concatenate(accs)
    var = float(np.mean(A.var(axis=0)))
    # 离散 WNA：Q = σ_a² · [[1/4,0,1/2,0],[0,1/4,0,1/2],[1/2,0,1,0],[0,1/2,0,1]]
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = var / 4
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = var / 2
    Q[2, 2] = Q[3, 3] = var
    return Q


def main(args):
    seqs = collect_detections(args, args.seeds)
    print(f"收集 {len(seqs)} 个场景 × {args.n_frames} 帧检测序列")

    R = estimate_noise(seqs, args)
    Q = estimate_process_noise(seqs)
    print(f"实测检测噪声 R = diag({R[0, 0]:.5f}, {R[1, 1]:.5f})")
    print(f"场景过程噪声 Q 迹 = {np.trace(Q):.2e}")

    res = {}
    for name, factory in [
        ("alpha_beta", lambda: MOTTracker(n_classes=len(CLASS_NAMES),
                                          class_names=CLASS_NAMES)),
        ("kalman_crb", lambda: KalmanMOTTracker(n_classes=len(CLASS_NAMES),
                                                class_names=CLASS_NAMES, R=R, Q=Q)),
        ("improved_assoc", lambda: ImprovedMOTTracker(n_classes=len(CLASS_NAMES),
                                                      class_names=CLASS_NAMES, R=R, Q=Q)),
    ]:
        per_seed = []
        for frames in seqs:
            tr_all = run_tracker(frames, factory)
            per_seed.append(evaluate(frames, tr_all, args))
        res[name] = {
            "rmse_mean": float(np.mean([p["rmse"] for p in per_seed])),
            "recall_mean": float(np.mean([p["recall"] for p in per_seed])),
            "idsw_mean": float(np.mean([p["id_switches"] for p in per_seed])),
            "per_seed": per_seed,
        }

    print(f"\n{'=' * 66}")
    print("MOT 跟踪器对比（相同检测输入，配对比较）:")
    for name, r in res.items():
        print(f"  {name:>12}: RMSE={r['rmse_mean']:.4f}  recall={r['recall_mean']:.3f}  "
              f"ID切换={r['idsw_mean']:.1f}")
    gain = 100 * (1 - res["kalman_crb"]["rmse_mean"] / res["alpha_beta"]["rmse_mean"])
    print(f"  → KF 位置 RMSE 改善: {gain:+.1f}%")

    # η_track：稳态 Kalman 先验协方差 vs 实测 RMSE²（理论地板）
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    P = np.eye(4) * 0.05
    for _ in range(200):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P
    crb_pos = float(np.trace(P[:2, :2]) / 2)          # 每轴稳态方差
    rmse2 = res["kalman_crb"]["rmse_mean"] ** 2
    eta_track = crb_pos / rmse2 if rmse2 > 0 else float("nan")
    print(f"  → 稳态 CRB(位置, 每轴) = {crb_pos:.5f}；η_track = CRB/RMSE² = {eta_track:.3f}")

    verdicts = {
        "T2_kf_rmse_better": bool(res["kalman_crb"]["rmse_mean"] <=
                                  res["alpha_beta"]["rmse_mean"]),
        "T3_eta_track_ge_090": bool(eta_track >= 0.90),
        "T4_no_degradation": bool(res["kalman_crb"]["recall_mean"] >=
                                  res["alpha_beta"]["recall_mean"] - 0.02 and
                                  res["kalman_crb"]["idsw_mean"] <=
                                  res["alpha_beta"]["idsw_mean"] + 1),
        "A1_recall_gain_ge_005": bool(res["improved_assoc"]["recall_mean"] >=
                                      res["kalman_crb"]["recall_mean"] + 0.05),
        "A2_idsw_reduce_ge_20pct": bool(res["improved_assoc"]["idsw_mean"] <=
                                        0.8 * res["kalman_crb"]["idsw_mean"]),
        "A3_rmse_no_worse": bool(res["improved_assoc"]["rmse_mean"] <=
                                 res["kalman_crb"]["rmse_mean"] * 1.05),
    }
    print(f"  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"results": res, "R": R.tolist(), "Q_trace": float(np.trace(Q)),
           "crb_pos_steady": crb_pos, "eta_track": eta_track,
           "rmse_gain_pct": float(gain), "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "tracker_kalman.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOT 跟踪器 KF 升级验证")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/detect_best.pth")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=40)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--conf_thr", type=float, default=0.5)
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
