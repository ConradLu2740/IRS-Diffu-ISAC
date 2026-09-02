# =============================================================================
# sensing — 感知算法层
#
# CA-Cfar1D  一维 CA-CFAR（单元平均恒虚警，向量实现）
# Planned: 2D-CFAR / MUSIC / ESPRIT / ML 适配器（迁移自 isac_sat baseline_classic）
#
# 场景级感知（扩散重建、HRRP 分类/定位）仍见 source_code/isac_sat。
# =============================================================================

from .cfar import CA_Cfar1D

__all__ = ["CA_Cfar1D"]
