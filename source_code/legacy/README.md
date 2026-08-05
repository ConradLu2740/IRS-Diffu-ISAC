# legacy/ — 原项目（RIS 辅助 ISAC + 扩散模型 3D 点云重建）

> 归档的原项目代码。星-地 ISAC 扩展见 `../isac_sat/`。
> 本目录脚本保留原项目实验能力，星-地扩展通过 `sys.path` 注入引用其中模块。

## 文件

| 文件 | 用途 |
|------|------|
| `setup.py` | 仿真参数与信道预计算（近程场景） |
| `data.py` / `data_no_irs.py` | IRS / 无 IRS 数据生成 |
| `models.py` / `models_unet.py` | PointVAE / DiT / UNet 模型 |
| `train.py` / `train_no.py` / `train_r.py` / `train_unified.py` | 训练脚本 |
| `phase_optimizer.py` | 静态 IRS 相位优化（坐标上升） |
| `run_experiments.py` / `analyze_results.py` / `generate_results_json.py` | 实验调度与统计 |
| `compute_stats.py` / `generate_charts.py` | 统计与图表 |
| `smoke_test.py` | 全链路最小复现 |
| `experiment_results.json` | 历史实验结果 |

## 运行

```bash
cd source_code/legacy
../.venv/bin/python train.py        # 原项目训练
../.venv/bin/python smoke_test.py   # 全链路 smoke
```

> 注意：原项目脚本未做结构整理，保持原样以兼容历史运行方式。
