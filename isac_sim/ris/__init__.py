# =============================================================================
# ris — RIS 模型层
#
# ContinuousPhaseRis  连续相位解析对齐（默认；与 isac_sat/phase_optimizer_sat 同构）
# BinaryPhaseRis      1-bit 离散相位（理想连续解的最近邻量化，smoke 级）
# SegmentedTracking   分段重构控制器（重构速率 vs 相干时间权衡，复用任意 RIS 实现）
# =============================================================================

from .base import RisModel, coherent_power
from .continuous import ContinuousPhaseRis
from .binary import BinaryPhaseRis
from .segmented import SegmentedTracking

__all__ = ["RisModel", "coherent_power", "ContinuousPhaseRis", "BinaryPhaseRis",
           "SegmentedTracking"]
