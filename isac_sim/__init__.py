# =============================================================================
# isac_sim — ISAC 全流程分层仿真参考库（骨架）
#
# 设计原则（与 isac_sat 参考实现一致）：
#   1. 分层可插拔：channel / waveform / RIS / comm / sensing / tracking
#      每层只依赖下层抽象，任何一层都可被独立替换或单独复用。
#   2. 核心零依赖：默认实现仅依赖 numpy（scipy 可选），不依赖 torch ——
#      让无 GPU / 无深度学习环境的同行也能跑通全部经典层。
#   3. smoke 级验收：每个模块入 main 前必须有物理 sanity check +
#      一条 make 命令 + CI 冒烟（缺一不收）。
#   4. 参考实现：source_code/isac_sat（星-地闭环）与 legacy（扩散重建）
#      保持不动，作为场景级参考应用；后续逐步迁移到本包接口上。
#
# 技术栈覆盖计划（见 stacks/README.md）：
#   - Python / numpy 主栈（本包）
#   - Sionna 交叉验证（数值对齐报告，P4）
#   - MATLAB 参考实现对照（关键模块）
# =============================================================================

from . import channels, comm, ris, sensing, tracking, waveforms

__version__ = "0.1.0"
