# =============================================================================
# waveforms — 波形层
#
# OfdmWaveform  多载波 OFDM 感知波形（默认；对齐 isac_sat 宽带 HRRP 形态）
# OtfsWaveform  高动态 NTN 候选波形（DD 域二维扭曲卷积 / Zak 变换视角）
# AfdmWaveform  高动态 NTN 候选波形（DAFT/IDAFT chirp 多载波）
# =============================================================================

from .base import Waveform
from .ofdm import OfdmWaveform
from .otfs import OtfsWaveform
from .afdm import AfdmWaveform

__all__ = ["Waveform", "OfdmWaveform", "OtfsWaveform", "AfdmWaveform"]
