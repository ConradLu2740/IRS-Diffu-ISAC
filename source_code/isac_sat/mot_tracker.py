"""
mot_tracker.py — 多目标追踪（MOT）

检测-关联-追踪：
  - 输入：每帧检测（K 组位置 + 类别概率）
  - 关联：匈牙利算法（预测位置 vs 检测位置，基于恒定速度模型）
  - 轨迹：ID 保持、类别多帧投票、位置 α-β 平滑
  - 输出：轨迹列表（ID / 类别 / 位置序列 / 置信度）

价值：单帧检测率有限（单站距离像），轨迹连续性补足漏检并稳定 ID。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_MISS = 5          # 轨迹丢失帧数上限
CONFIRM_FRAMES = 2    # 确认轨迹所需帧数
GATE = 0.35           # 关联门限（归一化距离）


class Track:
    def __init__(self, tid, pos, cls_probs, n_classes):
        self.id = tid
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)   # 3D 速度 (x, y, z)
        self.cls_probs = np.array(cls_probs, dtype=float)  # [n_classes]
        self.age = 1
        self.miss = 0
        self.confirmed = False
        self.history = [self.pos.copy()]
        self.n_classes = n_classes

    def predict(self):
        return self.pos + self.vel

    def update(self, det_pos, det_cls_probs):
        # α-β 平滑（α=0.6, β=0.2）
        pred = self.predict()
        resid = det_pos - pred
        self.pos = pred + 0.6 * resid
        self.vel = self.vel + 0.2 * resid
        self.cls_probs = 0.7 * self.cls_probs + 0.3 * det_cls_probs
        self.age += 1
        self.miss = 0
        self.confirmed = True if self.age >= CONFIRM_FRAMES else self.confirmed
        self.history.append(self.pos.copy())

    def miss_frame(self):
        self.miss += 1
        self.pos = self.predict()

    def class_name(self, class_names):
        return class_names[int(np.argmax(self.cls_probs))]


class MOTTracker:
    def __init__(self, n_classes, class_names=None):
        self.n_classes = n_classes
        self.class_names = class_names
        self.tracks = []
        self._next_id = 0
        self.metrics = {"id_switches": 0, "track_frags": 0}

    # ------------------------------------------------------------------
    def _new_track(self, det_pos, det_cls_probs):
        t = Track(self._next_id, det_pos, det_cls_probs, self.n_classes)
        self._next_id += 1
        self.tracks.append(t)
        return t

    # ------------------------------------------------------------------
    def update(self, detections):
        """detections: list of (pos [x,y], cls_probs [C])。

        返回：确认轨迹列表 [(id, class_id, pos, confidence)]。
        """
        # 1. 预测 + 关联
        if not self.tracks or len(detections) == 0:
            for d in detections:
                self._new_track(*d)
        else:
            preds = np.array([t.predict() for t in self.tracks])
            dets = np.array([d[0] for d in detections])
            cost = np.linalg.norm(preds[:, None, :] - dets[None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(cost)

            matched_track = set(); matched_det = set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= GATE:
                    self.tracks[r].update(*detections[c])
                    matched_track.add(r); matched_det.add(c)

            # 未匹配检测 → 新轨迹
            for c in range(len(detections)):
                if c not in matched_det:
                    self._new_track(*detections[c])
            # 未匹配轨迹 → 丢失
            for r in range(len(self.tracks)):
                if r not in matched_track:
                    self.tracks[r].miss_frame()

        # 2. 清理超时轨迹
        self.tracks = [t for t in self.tracks if t.miss <= MAX_MISS]

        # 3. 返回确认轨迹
        out = []
        for t in self.tracks:
            if t.confirmed:
                out.append((t.id, int(np.argmax(t.cls_probs)),
                            t.pos.copy(), float(np.max(t.cls_probs))))
        return out
