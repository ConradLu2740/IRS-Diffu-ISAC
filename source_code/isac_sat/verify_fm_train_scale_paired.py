"""
verify_fm_train_scale_paired.py — FM 训练预算扫描的**同批次逐云配对重评**

动机（审稿意见驱动）：verify_fm_train_scale.py 的 run_cell 在训练消耗全局 RNG
**之后**才取测试批，而 SatROIDataset.__getitem__ 是现采样（generate_ground_target_sample
用全局 random.choice/uniform，_frame_phases 用 torch.rand），因此各单元格记录的聚合 CD
实际上是在**各自不同**的 8 云测试批上计算的——跨预算比较混入了批噪声，种子间的符号翻转
可能部分是 n=8 的检验功效问题而非真实反转。

本脚本用已保存的最终权重（*_latest.pth）在**每个种子一个固定构造的公共批次**上重评
全部单元格（单元格间重种子固定 x0，使 GT 云与噪声双重配对），输出逐云 CD 与相对
512/60 基单元格的配对差，并按预注册口径判读。

预注册判读口径（P_paired，运行前记录于 JSON）：
  对每个种子，以 512/60 为基，逐云配对差 Δ_i = CD_i(budget) − CD_i(base)：
  - median(Δ) ≤ −0.02 且改善云数 ≥ 6/8 → "consistent improvement"
  - median(Δ) ≥ +0.02 且恶化云数 ≥ 6/8 → "consistent degradation"
  - 其余（|median| ≤ 0.02 或符号混合）→ "no consistent scale effect at n=8"
  若 top-2 |Δ| 占 Σ|Δ| ≥ 60%，显式记录 "aggregate sign driven by <=2 clouds"。
  P_scale 在配对口径下的等价判定：两个种子都 consistent improvement 才 PASS。

与原 fm_train_scale.json 的关系：原文件保留为**训练运行记录**（每格各自的随机批次）；
本文件是配对口径下的**承重比较**。两者 CD 绝对值不同属预期（批次不同），
论文引用以本文件为准，并在正文说明原聚合值为各自批次上的单次观测。

用法：
  py verify_fm_train_scale_paired.py                  # 重评全部已有单元格
  py verify_fm_train_scale_paired.py --seeds 42       # 单种子
  未训练的单元格（缺 *_latest.pth）自动记 "not completed"，可断点重跑。
"""

import os
import json
import time
import random
import hashlib
import argparse
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import chamfer_distance_loss
from fm_utils import sample_conditional_FM

PROPOSITION = (
    "P_paired (registered before running): on a per-seed common test batch "
    "(fixed construction, shared x0 across cells), the paired per-cloud CD "
    "difference vs the (512,60) base cell reads 'consistent improvement' "
    "(median <= -0.02 and >= 6/8 clouds improving) on BOTH seeds 42 and 43. "
    "A median within +/-0.02, or a mixed sign pattern, reads 'no consistent "
    "scale effect at n=8' regardless of the aggregate sign; an aggregate sign "
    "flip dominated by <= 2 clouds (top-2 |delta| share >= 60%) is recorded as "
    "outlier-driven, not as 'scaling harmful'."
)

X0_SEED_OFFSET = 9000  # 每格采样前重种子，固定 x0


class _PairView(torch.utils.data.Dataset):
    """把 (pc, cond, *rest) 数据集包装成 (pc, cond) 对（宽带特征丢弃），
    与 verify_fm_train_scale.py 的 run_cell 保持一致。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        out = self.base[i]
        return out[0], out[1]


def budget_tag(samples, epochs):
    return f"b{samples}_{epochs}"


def parse_budget(s):
    a, b = s.split(":")
    return int(a), int(b)


def per_cloud_cd(pc_gt, pc_hat):
    """逐云 CD（与 chamfer_distance_loss 的批均值一致：批值 = 逐云均值）。"""
    cds = [chamfer_distance_loss(pc_gt[i:i + 1], pc_hat[i:i + 1]).item()
           for i in range(pc_gt.shape[0])]
    batch_cd = chamfer_distance_loss(pc_gt, pc_hat).item()
    assert abs(float(np.mean(cds)) - batch_cd) < 1e-5, (
        f"per-cloud mean {np.mean(cds)} != batch CD {batch_cd}")
    return [round(c, 6) for c in cds], batch_cd


def build_common_batch(args, seed):
    """每个种子一个固定构造的公共测试批（构造前重种子，首批 8 云）。"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tle = ss.STARLINK_TLE if args.sat == "starlink" else ss.ISS_TLE
    scenario = ss.SatISACScenario(tau=args.tau, tle_lines=tle, sat_name=args.sat)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=args.device)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=args.device, tau=args.tau,
                            phase_mode=args.phase_mode, cond_feat=args.cond_feat)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(args.device)
    cond = cond[:args.n_eval].to(args.device)
    h = hashlib.sha1()
    h.update(pc_gt.detach().cpu().numpy().tobytes())
    h.update(cond.detach().cpu().numpy().tobytes())
    return pc_gt, cond, h.hexdigest()[:12]


def load_cell(args, seed, tag):
    """加载某单元格的最终权重；缺失返回 None。"""
    d = os.path.join(args.save_dir.format(seed=seed), tag)
    ce_p = os.path.join(d, "condenc_fm_latest.pth")
    vn_p = os.path.join(d, "vnet_fm_latest.pth")
    if not (os.path.exists(ce_p) and os.path.exists(vn_p)):
        return None
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=args.cond_dim,
                                  hidden_size=128, out_emb=256).to(args.device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(args.device)
    condenc.load_state_dict(torch.load(ce_p, map_location=args.device))
    vnet.load_state_dict(torch.load(vn_p, map_location=args.device))
    return condenc, vnet


def eval_cell(args, seed, tag, vae, z_mean, z_std, pc_gt, cond):
    """在公共批次上重评一个单元格（重种子固定 x0），返回逐云 CD 与聚合。"""
    loaded = load_cell(args, seed, tag)
    if loaded is None:
        return {"tag": tag, "status": "not completed",
                "note": "final checkpoints (*_latest.pth) not found; cell not trained yet"}
    condenc, vnet = loaded
    torch.manual_seed(X0_SEED_OFFSET + seed)
    random.seed(X0_SEED_OFFSET + seed)
    np.random.seed(X0_SEED_OFFSET + seed)
    pc_hat = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                   device=args.device, cfg_scale=args.cfg,
                                   nfe=args.nfe, solver=args.solver)
    cds, batch_cd = per_cloud_cd(pc_gt, pc_hat)
    return {"tag": tag, "status": "ok", "per_cloud_cd": cds,
            "cd_mean": round(batch_cd, 6)}


def read_delta(deltas):
    """预注册判读口径（见 PROPOSITION）。"""
    d = np.asarray(deltas, dtype=float)
    med = float(np.median(d))
    n_imp = int((d < 0).sum())
    n_wor = int((d > 0).sum())
    absd = np.abs(d)
    top2 = float(np.sort(absd)[-2:].sum() / absd.sum()) if absd.sum() > 0 else 0.0
    if med <= -0.02 and n_imp >= 6:
        reading = "consistent improvement"
    elif med >= 0.02 and n_wor >= 6:
        reading = "consistent degradation"
    else:
        reading = "no consistent scale effect at n=8"
    outlier = bool(top2 >= 0.60)
    return {"median_delta": round(med, 6), "n_improved": n_imp, "n_worsened": n_wor,
            "top2_abs_delta_share": round(top2, 4), "reading": reading,
            "aggregate_sign_outlier_driven": outlier}


def run_seed(args, seed):
    pc_gt, cond, fp = build_common_batch(args, seed)
    print(f"[seed {seed}] common batch fingerprint={fp}, n={pc_gt.shape[0]}")
    vae_src = os.path.join(args.vae_src.format(seed=seed), args.irs_mode)
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(args.device)
    vae.load_state_dict(torch.load(os.path.join(vae_src, "vae_best.pth"),
                                   map_location=args.device))
    stats = torch.load(os.path.join(vae_src, "latent_stats.pth"),
                       map_location=args.device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]

    with torch.no_grad():
        mu, _ = vae.encode(pc_gt)
        pc_vae, _ = vae.decode(mu)
    oracle_cds, oracle_cd = per_cloud_cd(pc_gt, pc_vae)

    cells, paired = {}, {}
    base_cds = None
    for s, e in args.budgets:
        tag = budget_tag(s, e)
        res = eval_cell(args, seed, tag, vae, z_mean, z_std, pc_gt, cond)
        cells[tag] = res
        if res["status"] != "ok":
            continue
        if base_cds is None:
            base_cds = res["per_cloud_cd"]
            base_tag = tag
        else:
            deltas = [round(a - b, 6) for a, b in zip(res["per_cloud_cd"], base_cds)]
            paired[tag] = {"base": base_tag, "delta_per_cloud": deltas,
                           **read_delta(deltas)}
        print(f"[seed {seed}] {tag}: CD(mean on common batch)={res['cd_mean']:.4f}"
              + (f"  Δ vs {base_tag}: median={paired[tag]['median_delta']:+.4f}, "
                 f"{paired[tag]['n_improved']}/8 improved, "
                 f"reading={paired[tag]['reading']}" if tag in paired else ""))
    readings = [v["reading"] for v in paired.values()]
    verdict = ("PASS" if readings and all(r == "consistent improvement" for r in readings)
               else "not PASS")
    return {"batch_fingerprint": fp, "oracle_per_cloud_cd": oracle_cds,
            "oracle_cd_mean": round(oracle_cd, 6), "cells": cells,
            "paired_vs_base": paired, "seed_verdict": verdict}


def main(args):
    t0 = time.time()
    out_path = args.out
    prev = None
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
    out = {
        "experiment": "fm_train_scale_paired",
        "title": ("Paired per-cloud re-evaluation of the FM training-budget sweep "
                  "on a per-seed common test batch"),
        "motivation": (
            "run_cell draws the test batch AFTER training consumes the global RNG "
            "while SatROIDataset.__getitem__ samples on the fly (random.choice/"
            "uniform + torch.rand), so each cell's recorded aggregate CD was computed "
            "on a different 8-cloud batch; cross-budget comparison is confounded by "
            "batch noise. This file is the load-bearing paired comparison: one fixed "
            "common batch per seed, shared x0 across cells, per-cloud CD."),
        "proposition": PROPOSITION,
        "proposition_registered_before_run": True,
        "config": {
            "seeds": args.seeds,
            "budgets": [f"{s}/{e}" for s, e in args.budgets],
            "irs_mode": args.irs_mode, "cond_feat": args.cond_feat,
            "phase_mode": args.phase_mode, "test_data": args.test_data,
            "n_eval": args.n_eval, "num_points": args.num_points,
            "tau": args.tau, "dit_depth": args.depth,
            "vae_frozen_from": args.vae_src,
            "checkpoints": args.save_dir + "/<seed>/<tag>/*_latest.pth (final weights)",
            "sampling": {"nfe": args.nfe, "cfg_scale": args.cfg,
                         "solver": args.solver,
                         "x0_seed_offset": X0_SEED_OFFSET},
            "device": args.device,
        },
        "seeds": {},
        "verdicts": {},
    }
    if prev and prev.get("seeds"):
        out["seeds"].update(prev["seeds"])
    for seed in args.seeds:
        out["seeds"][str(seed)] = run_seed(args, seed)
    verdicts = {str(sd): out["seeds"][str(sd)]["seed_verdict"] for sd in args.seeds}
    out["verdicts"] = {
        "P_scale_paired_PASS_requires_both_seeds_consistent_improvement": verdicts,
        "overall": "PASS" if all(v == "PASS" for v in verdicts.values()) else "not PASS",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n配对重评完成（{time.time() - t0:.1f}s）→ {out_path}")
    print(f"裁决: {json.dumps(out['verdicts'], ensure_ascii=False)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FM 训练预算扫描：同批次逐云配对重评")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--budgets", nargs="+", default=["512:60", "1024:100", "2048:200"])
    p.add_argument("--vae_src", type=str, default="./sat_model_scale_{seed}")
    p.add_argument("--save_dir", type=str, default="./sat_model_m3b_{seed}")
    p.add_argument("--out", type=str, default="./sat_model_cmp/fm_train_scale_paired.json")
    p.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    p.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                   default="hrrp")
    p.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    p.add_argument("--test_data", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_eval", type=int, default=8)
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--tau", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--nfe", type=int, default=1)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    p.add_argument("--roi_res", type=int, default=16)
    p.add_argument("--sat", choices=["iss", "starlink"], default="iss")
    args = p.parse_args()
    args.budgets = [parse_budget(b) for b in args.budgets]
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # cond_dim 由公共批决定（hrrp 广播条件下 = K）
    pc_gt, cond, fp = build_common_batch(args, args.seeds[0])
    args.cond_dim = cond.shape[-1]
    print(f"Device: {args.device}, cond_dim={args.cond_dim}, seeds={args.seeds}")
    main(args)
