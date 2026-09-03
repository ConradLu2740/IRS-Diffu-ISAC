# =============================================================================
# channels — 信道模型层
#
# L0 FreeSpaceChannel  自由空间远场（默认；与 isac_sat/data_sat.get_channel_mat 同构）
# L1 RicianChannel     莱斯衰落（LOS + 散射簇，平均功率与 L0 对齐）
# L2 ThreeGppNtnChannel 3GPP TR 38.811 NTN 对齐（P1 规划中，暂未实现）
#
# L2.1 SionnaCdlChannel（已实现，可选依赖 sionna>=2.0）
#      Sionna 2.x CDL 标准信道对照（TR 38.901 CDL-D 代理），
#      懒加载：from isac_sim.channels.sionna_cdl import SionnaCdlChannel
#      约定：所有矩阵元素为复数链路增益（无量纲），距离输入单位为米。
# =============================================================================

from .base import ChannelModel
from .free_space import FreeSpaceChannel
from .rician import RicianChannel

__all__ = ["ChannelModel", "FreeSpaceChannel", "RicianChannel"]