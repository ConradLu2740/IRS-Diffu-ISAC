"""分段重构控制器：量化"重构速率 vs 信道相干时间"权衡。

每 update_interval 帧才重新配置一次相位，中间帧沿用旧配置。
与 isac_sat/phase_optimizer_sat.py::optimize_sequence 的 K-sweep 同构，
但与具体 RIS 模型解耦：ContinuousPhaseRis / BinaryPhaseRis 均可插入。
"""

import numpy as np

from .base import RisModel


class SegmentedTracking:
    def __init__(self, ris: RisModel, update_interval: int = 1):
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.ris = ris
        self.update_interval = int(update_interval)

    def run(self, gains_per_frame: list, power_fn=None) -> dict:
        """gains_per_frame: 每帧的有效增益列表 [T][N]。

        power_fn(gains, phases) → float：可选的功率评估函数，
        默认用 base.coherent_power。返回 per-frame 相位与功率。
        """
        if power_fn is None:
            from .base import coherent_power as power_fn
        phases_per_frame, powers = [], []
        current = None
        for t, g in enumerate(gains_per_frame):
            if current is None or t % self.update_interval == 0:
                current = self.ris.configure(g)
            phases_per_frame.append(current)
            powers.append(power_fn(g, current))
        return {"phases": np.array(phases_per_frame), "powers": np.array(powers)}
