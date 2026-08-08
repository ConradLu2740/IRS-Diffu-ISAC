"""
mot_data.py — 多移动目标场景生成（多目标追踪数据源）

场景：LEO 卫星感知 ROI 内 N 个移动目标（5 类：car/uav/bicycle/pedestrian/train）。
- 每帧目标位置更新（类别速度 + 边界反弹）
- 每帧合成 ROI 体素（目标模板放当前位置）
- 提供每帧距离像（宽带）作为检测特征

类别速度（城市/近空典型值，m/s）：
  pedestrian 1.4 · bicycle 5 · uav 8 · car 12 · train 25
"""

import numpy as np
import setup_sat as ss
from data_sat import (GROUND_TARGET_TEMPLATES, make_roi_local,
                      compute_range_profile, WIDEBAND_K)

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
SPEEDS_MPS = {"pedestrian": 1.4, "bicycle": 5.0, "uav": 8.0, "car": 12.0, "train": 25.0}


def _template_by_name(name):
    for n, maker in GROUND_TARGET_TEMPLATES:
        if n == name:
            return maker().astype(np.float32)
    raise KeyError(name)


class MovingTarget:
    """单个移动目标：类别 + 位置（体素坐标）+ 速度（体素/帧）。"""

    def __init__(self, cls_id, pos, vel, roi_res=16):
        self.cls_id = cls_id
        self.cls_name = CLASS_NAMES[cls_id]
        self.pos = np.array(pos, dtype=float)   # ROI 体素坐标（连续）
        self.vel = np.array(vel, dtype=float)   # 体素/帧
        self.roi_res = roi_res
        self.template = _template_by_name(self.cls_name)

    def step(self):
        """更新位置，边界反弹。"""
        self.pos = self.pos + self.vel
        # 反弹（目标模板 8³，位置范围 [0, res-8]）
        low, high = 0.0, self.roi_res - 8.0
        for d in range(2):
            if self.pos[d] < low:
                self.pos[d] = low + (low - self.pos[d])
                self.vel[d] = abs(self.vel[d])
            elif self.pos[d] > high:
                self.pos[d] = high - (self.pos[d] - high)
                self.vel[d] = -abs(self.vel[d])

    def place(self, roi, t):
        """把目标模板放到 ROI 当前帧位置。"""
        x, y = int(round(self.pos[0])), int(round(self.pos[1]))
        z = 4  # 目标贴地（z 固定）
        x = min(max(x, 0), self.roi_res - 8)
        y = min(max(y, 0), self.roi_res - 8)
        roi[x:x + 8, y:y + 8, z:z + 8] = np.maximum(
            roi[x:x + 8, y:y + 8, z:z + 8], self.template)
        return roi

    def center(self):
        """目标中心（归一化 [-1,1]）。"""
        return (self.pos[:2] + 4.0) / self.roi_res * 2.0 - 1.0


class MovingTargetScene:
    """N 个移动目标的场景：帧序列 + ROI 渲染 + 距离像。"""

    def __init__(self, n_targets=10, roi_res=16, n_frames=16, dt_s=1.0,
                 seed=42, tle_lines=None, fc_hz=None):
        self.n_targets = n_targets
        self.roi_res = roi_res
        self.n_frames = n_frames
        self.dt_s = dt_s
        rng = np.random.RandomState(seed)

        # 场景几何（真实轨道）
        self.scenario = ss.SatISACScenario(
            tle_lines=tle_lines or ss.ISS_TLE,
            fc_hz=fc_hz or ss.FC_HZ, tau=n_frames)
        self.frames = self.scenario.build_frames()
        self.mid = self.frames[len(self.frames) // 2]

        # 目标初始化：随机类别 + 随机位置 + 类别速度
        self.targets = []
        for i in range(n_targets):
            cls_id = rng.randint(len(GROUND_TARGET_TEMPLATES))
            cls_name = CLASS_NAMES[cls_id]
            pos = rng.uniform(0, roi_res - 8, 2)
            speed_vox = SPEEDS_MPS[cls_name] * dt_s / 5.0   # 5m/体素
            ang = rng.uniform(0, 2 * np.pi)
            vel = speed_vox * np.array([np.cos(ang), np.sin(ang)])
            self.targets.append(MovingTarget(cls_id, pos, vel, roi_res))

    # ------------------------------------------------------------------
    def render_roi(self, t):
        """合成第 t 帧 ROI 体素（所有目标当前位置）。"""
        roi = np.zeros((self.roi_res, self.roi_res, self.roi_res), dtype=np.float32)
        for tg in self.targets:
            roi = tg.place(roi, t)
        return roi

    def targets_at(self, t):
        """第 t 帧目标 (cls_id, center_xy) 列表。"""
        return [(tg.cls_id, tg.center()) for tg in self.targets]

    def step(self):
        for tg in self.targets:
            tg.step()

    # ------------------------------------------------------------------
    def range_profile_sequence(self, snr_db=20.0):
        """生成整个场景的距离像序列（每帧）。

        返回 (rps [T, K], ground_truth [T, N, (cls, cx, cy)])
        """
        rps, gts = [], []
        for t in range(self.n_frames):
            roi = self.render_roi(t)
            rp = compute_range_profile(
                roi, self.mid["target_pos"], self.mid["ground_pos"],
                self.scenario.wavelength_m, snr_db=snr_db, seed=t, align=False)
            rps.append(rp)
            gts.append(self.targets_at(t))
            self.step()
        return np.array(rps), gts

    # ------------------------------------------------------------------
    def summary(self):
        counts = {}
        for tg in self.targets:
            counts[tg.cls_name] = counts.get(tg.cls_name, 0) + 1
        return counts
