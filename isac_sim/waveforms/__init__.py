# =============================================================================
# waveforms — 波形层
#
# OfdmWaveform  多载波 OFDM 感知波形（默认；对齐 isac_sat 宽带 HRRP 形态）
# OtfsWaveform  高动态 NTN 候选波形（P3 规划中）
# =============================================================================

from .base import Waveform
from .ofdm import OfdmWaveform

__all__ = ["Waveform", "OfdmWaveform"]
