"""最近邻关联 + 匀速（CV）预测最小跟踪器（smoke 级）。

与 isac_sat/mot_tracker.py（匈牙利关联 + 多数表决）互补：
本实现是教学级最小参考，展示关联-预测-维护的最小闭环。
"""

import numpy as np


class NearestNeighborTracker:
    def __init__(self, gate_m: float = 5.0, dt: float = 1.0):
        self.gate = float(gate_m)
        self.dt = float(dt)
        self.tracks = {}   # track_id -> {"pos": [3], "vel": [3], "hits": int, "miss": int}
        self._next_id = 0

    def step(self, detections: np.ndarray):
        """输入 [M, 3] 检测位置（米），更新轨迹；返回 {track_id: pos}。"""
        detections = np.atleast_2d(np.asarray(detections, dtype=float))
        used = set()
        for tid, tr in self.tracks.items():
            pred = tr["pos"] + tr["vel"] * self.dt
            if len(detections) == 0:
                tr["miss"] += 1
                continue
            d = np.linalg.norm(detections - pred, axis=1)
            j = int(np.argmin(d))
            if d[j] <= self.gate and j not in used:
                # 位置更新 + 速度估计（一阶差分）
                tr["vel"] = (detections[j] - tr["pos"]) / self.dt
                tr["pos"] = detections[j].copy()
                tr["hits"] += 1
                tr["miss"] = 0
                used.add(j)
            else:
                tr["miss"] += 1
        # 新轨迹
        for j in range(len(detections)):
            if j not in used:
                self.tracks[self._next_id] = {
                    "pos": detections[j].copy(), "vel": np.zeros(3),
                    "hits": 1, "miss": 0}
                self._next_id += 1
        # 删除长期丢失轨迹
        self.tracks = {t: v for t, v in self.tracks.items() if v["miss"] <= 3}
        return {tid: tr["pos"].copy() for tid, tr in self.tracks.items()}
