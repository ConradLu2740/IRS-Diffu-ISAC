# =============================================================================
# comm — 通信链路层（补齐"通信侧只有功率增益"的短板）
#
# QpskAwgnLink  QPSK over AWGN 最小链路（smoke 级，BER 对照理论曲线）
# Planned: 高阶 QAM / 简单编码 / 频谱效率指标
# =============================================================================

from .base import DigitalModulation
from .qpsk import QpskAwgnLink

__all__ = ["DigitalModulation", "QpskAwgnLink"]
