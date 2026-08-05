# 贡献指南

感谢你对 IRS-Diffu-ISAC 感兴趣！任何形式的贡献都欢迎：bug 报告、功能建议、代码、文档。

## 快速开始

```bash
# 克隆 + 环境
git clone https://github.com/ConradLu2740/IRS-Diffu-ISAC.git
cd IRS-Diffu-ISAC
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 验证环境正常
cd source_code/isac_sat && ../../.venv/bin/python verify_sat.py
```

## 代码结构

```
source_code/
├── isac_sat/     # 星-地 ISAC + 感知 + demo（活跃开发区）
└── legacy/       # 原项目（RIS + 扩散模型重建，归档）
```

- 新功能优先放在 `isac_sat/`（模块化：物理层/数据/感知/通信/demo）
- 引用 legacy 模块时用 `sys.path` 注入（见现有脚本模式）

## 提交规范

- commit message 用常规格式：`feat: ...` / `fix: ...` / `docs: ...` / `refactor: ...`
- 修改后跑 `verify_sat.py`（物理验证）确保不破坏基础
- 新脚本/功能补充说明到 `space_isac_design.md`

## 如何贡献

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feat/your-feature`
3. 提交改动（遵循规范）
4. 发起 Pull Request，说明改动内容与验证结果

## 反馈问题

- Bug / 功能建议 → [Issues](https://github.com/ConradLu2740/IRS-Diffu-ISAC/issues)
- 使用问题 → 附上运行环境与报错输出

## 项目方向

- 太空 ISAC（ISAC-NTN）：LEO/GEO 卫星感知 + 通信
- RIS 动态配置：跟踪、重构速率权衡
- 感知-通信闭环：感知结果驱动无线资源配置
- 真实数据：SDR 接口、半实物验证
