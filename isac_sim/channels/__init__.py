# =============================================================================
# channels — 信道模型层
#
# L0 FreeSpaceChannel  自由空间远场（默认；与 isac_sat/data_sat.get_channel_mat 同构）
# L1 RicianChannel     莱斯衰落（LOS + 散射簇，平均功率与 L0 对齐）
# L2 ThreeGppNtnChannel 3GPP TR 38.811 NTN 对齐（P1 规划中，暂未实现）
#
# 约定：所有矩阵元素为复数链路增益（无量纲），距离输入单位为米。
# =============================================================================

from .base import ChannelModel
from .free_space import FreeSpaceChannel
from .rician import RicianChannel

__all__ = ["ChannelModel", "FreeSpaceChannel", "RicianChannel"]
