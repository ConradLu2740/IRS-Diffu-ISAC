# stacks — 技术栈交叉验证

ISAC 仿真参考库的可信度不只来自自身自洽，还来自**跨栈数值对齐**。
规划如下（P4 启动，按模块逐个做）：

## Sionna 交叉验证（TensorFlow）

- 范围：FreeSpaceChannel / RicianChannel 与 Sionna RT / 内置信道模型
  在相同几何与载频下的链路增益、多普勒谱对齐。
- 交付：`sionna_crosscheck.py`（对齐报告 + 最大相对误差断言），
  误差阈值入 CI。
- 安装：独立 extra（`pip install isac-sim[sionna]` 计划中），
  不污染主栈 numpy 依赖。

## MATLAB 参考实现对照

- 范围：关键模块各一份 MATLAB 参考脚本（FreeSpace / CA-CFAR / QPSK BER），
  数值与 Python 主栈导出的固定测试向量对齐（`tests/golden/*.npz`）。
- 目的：覆盖 ISAC 领域大量 MATLAB 生态同行，降低迁移门槛。
- 交付：`matlab/` 目录 + 对齐说明。

## 原则

- 主栈永远是 Python/numpy（零 torch 依赖）。
- 交叉验证是**单向对照**：外部栈结果不改主栈行为，只出报告。
- 对齐失败优先怀疑约定差异（FFT 平移、增益因子、噪声方差定义），
  并在两边注释里写明。
