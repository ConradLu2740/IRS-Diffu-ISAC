"""
verify_fm_train_scale.py — FM 训练预算扫描：端到端质量差距是否随训练规模向 VAE oracle 闭合

背景：FMShape 论文把残余质量差距（FMShape CD ~0.23 vs VAE oracle ~0.009）
归因于潜空间容量与训练规模而非生成目标，但这一归因目前只有断言、没有证据。
既有实验已排除/部分排除其他解释：
  - verify_vae_scale.py（M1 vs scale）：VAE 训练规模 50ep/256 → 200ep/1024 后
    VAE 天花板下降 ~48%（V1/V4 PASS），但 FM(NFE=10) CD 未改善（V2 FAIL）；
  - verify_gen_scale.py（M3，gen_scale.json）：VAE 固定时生成训练 100ep → 400ep
    （4×）后 FM NFE=1 CD 不降反升（Ga/Gb FAIL，间隙比 27–45× 不缩小）。
本实验把变量换成 **FM 训练预算（样本数 × epoch 数）** 的联合缩放：
  (512, 60) 论文设定 → (1024, 100) → (2048, 200)，种子 {42, 43}，
  其余全部固定：HRRP 条件、phase_mode=random、fresh-data 协议（不物化）、
  同一 DiT 容量（2 blocks / 256 hidden / 8 heads）与条件编码器、CFG w=2、
  NFE=1 Euler 采样、同一测试批构造协议（test_data=32 取首批 8 个）、
  VAE 固定复用 sat_model_scale_<seed>（200ep/1024，与 M3/gen_scale 同口径），
  只变 FM 侧训练预算，隔离"生成模型训练规模"这一单一变量。

预注册命题（运行前记录，先写 JSON 骨架再训练）：
  P_scale: FMShape NFE=1 CD 随训练预算单调非增，
           且 oracle-gap ratio（FM NFE=1 CD / VAE oracle CD）单调非增。
  判定口径与 verify_gen_scale.py 一致：两个种子都满足才算 PASS；
  任一单元格未完成（如触发计时护栏被停）则记 false 并显式标注 incomplete。

用法：
  py verify_fm_train_scale.py --plan_only                       # 只写预注册骨架
  py verify_fm_train_scale.py --seeds 42 --budgets 512:60       # 单单元格（可断点续跑）
  py verify_fm_train_scale.py                                   # 跑全部 3×2 单元格
  py verify_fm_train_scale.py --mark_incomplete 42:b2048_200    # 计时护栏停机后显式登记
"""

import os
import json
import time
import argparse
import numpy as np
import random
import torch
import sys
from torch.utils.data import DataLoader

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import matplotlib
matplotlib.use("Agg")

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import chamfer_distance_loss
from fm_utils import train_1D_FM, sample_conditional_FM
from eval_sat import f_score, voxel_iou

PROPOSITION = (
    "P_scale: FMShape NFE=1 CD is monotonically non-increasing in the FM training "
    "budget (samples, epochs) = (512,60) -> (1024,100) -> (2048,200), AND the "
    "oracle-gap ratio (FM NFE=1 CD / VAE oracle CD) is monotonically non-increasing. "
    "Verdict PASS requires both trends to hold on every seed (42, 43)."
)

DESIGN_NOTES = [
    "Only the FM (generative) training budget varies; DiT capacity (2 blocks/256 hidden/8 heads), "
    "condition encoder, CFG w=2, NFE=1 Euler sampling, HRRP condition, phase_mode=random and the "
    "fresh-data protocol are identical across cells.",
    "Stage 1 is frozen: the PointVAE (200ep/1024) and its latent whitening stats are reused from "
    "sat_model_scale_<seed>/sat via the same --vae_ckpt protocol as compare_gen.py / M3, so the "
    "VAE oracle is a fixed per-seed reference and the ratio denominator does not drift with budget.",
    "Evaluation mirrors compare_gen.py / eval_sat.py: first batch (8 samples) of the fresh test "
    "loader (test_data=32), CD / F-Score@0.1 / F-Score@0.2 / voxel IoU on the same batch; the "
    "VAE oracle is the VAE reconstruction (posterior mean) of the same batch.",
    "FM models are evaluated at the final in-memory weights (same as compare_gen.py), not the "
    "best-test-loss checkpoint.",
]


class _PairView(torch.utils.data.Dataset):
    """把 (pc, cond, *rest) 数据集包装成 (pc, cond) 对（宽带特征丢弃）。"""

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


def eval_metrics(pc_gt, pc_hat, roi_res):
    cd = chamfer_distance_loss(pc_gt, pc_hat).item()
    fs1 = float(np.mean([f_score(pc_gt[i], pc_hat[i], 0.1) for i in range(pc_gt.shape[0])]))
    fs2 = float(np.mean([f_score(pc_gt[i], pc_hat[i], 0.2) for i in range(pc_gt.shape[0])]))
    iou = float(np.mean([voxel_iou(pc_gt[i], pc_hat[i], roi_res) for i in range(pc_gt.shape[0])]))
    return {"cd": cd, "fs_0.1": fs1, "fs_0.2": fs2, "iou": iou}


def run_cell(args, seed, samples, epochs):
    """训练一个 (预算, 种子) 单元格并返回结果行。"""
    tag = budget_tag(samples, epochs)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = args.device
    print(f"\n{'=' * 74}")
    print(f"[seed {seed} | budget {samples}/{epochs}] FM 训练预算单元格开始")
    print(f"{'=' * 74}")

    tle = ss.STARLINK_TLE if args.sat == "starlink" else ss.ISS_TLE
    scenario = ss.SatISACScenario(tau=args.tau, tle_lines=tle, sat_name=args.sat)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device)
    train_ds = SatROIDataset(samples, channels, num_points=args.num_points,
                             device=device, tau=args.tau, phase_mode=args.phase_mode,
                             cond_feat=args.cond_feat)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            cond_feat=args.cond_feat)
    train_loader = DataLoader(_PairView(train_ds), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    cond_dim = train_ds[0][1].shape[-1]
    print(f"[seed {seed} | {tag}] cond_dim={cond_dim}, train={len(train_ds)}, test={len(test_ds)}")

    # ---- Stage 1 固定：复用 sat_model_scale_<seed> 的 VAE + 白化统计 ----
    vae_src = os.path.join(args.vae_src.format(seed=seed), args.irs_mode)
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(vae_src, "vae_best.pth"),
                                   map_location=device))
    stats = torch.load(os.path.join(vae_src, "latent_stats.pth"), map_location=device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]
    print(f"[seed {seed} | {tag}] Stage 1 (frozen): PointVAE + latent_stats 复用自 {vae_src}")

    # ---- Stage 2：Flow Matching（容量与 M3/gen_scale 完全一致）----
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    save_dir = os.path.join(args.save_dir.format(seed=seed), tag)
    os.makedirs(save_dir, exist_ok=True)
    train_1D_FM(vae, condenc, vnet, train_loader, test_loader,
                z_mean, z_std, device=device, epochs=epochs,
                lr_cond=args.lr_cond, posterior_sample=args.posterior_sample,
                save_dir=save_dir)

    # ---- 同一测试批评估（协议同 compare_gen.py / eval_sat.py）----
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt[:args.n_eval].to(device)
    cond = cond[:args.n_eval].to(device)

    pc_hat = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                   device=device, cfg_scale=args.cfg, nfe=args.nfe,
                                   solver=args.solver)
    fm_m = eval_metrics(pc_gt, pc_hat, args.roi_res)
    print(f"[seed {seed} | {tag}] FM NFE={args.nfe}: CD={fm_m['cd']:.6f} "
          f"FS@0.1={fm_m['fs_0.1']:.4f} FS@0.2={fm_m['fs_0.2']:.4f} IoU={fm_m['iou']:.4f}")

    with torch.no_grad():
        mu, _ = vae.encode(pc_gt)
        pc_vae, _ = vae.decode(mu)
    oracle = eval_metrics(pc_gt, pc_vae, args.roi_res)
    print(f"[seed {seed} | {tag}] VAE oracle: CD={oracle['cd']:.6f} "
          f"FS@0.1={oracle['fs_0.1']:.4f} IoU={oracle['iou']:.4f}")

    runtime = time.time() - t0
    return {
        "seed": seed,
        "budget": f"{samples}/{epochs}",
        "train_data": samples,
        "gen_epochs": epochs,
        "tag": tag,
        "status": "ok",
        "runtime_sec": round(runtime, 1),
        "runtime_min": round(runtime / 60.0, 2),
        "fm_nfe1": fm_m,
        "vae_oracle": oracle,
        "gap_ratio": fm_m["cd"] / oracle["cd"],
        "save_dir": save_dir.replace("\\", "/"),
    }


def planned_rows(seeds, budgets):
    rows = []
    for sd in seeds:
        for s, e in budgets:
            rows.append({"seed": sd, "budget": f"{s}/{e}", "train_data": s,
                         "gen_epochs": e, "tag": budget_tag(s, e),
                         "status": "pending"})
    return rows


def config_dict(args, seeds, budgets):
    return {
        "seeds": seeds,
        "budgets": [f"{s}/{e}" for s, e in budgets],
        "irs_mode": args.irs_mode,
        "cond_feat": args.cond_feat,
        "phase_mode": args.phase_mode,
        "fresh_data": True,
        "materialized": False,
        "train_data_per_budget": [s for s, _ in budgets],
        "fm_epochs_per_budget": [e for _, e in budgets],
        "test_data": args.test_data,
        "batch_size": args.batch_size,
        "n_eval": args.n_eval,
        "num_points": args.num_points,
        "tau": args.tau,
        "dit_depth": args.depth,
        "dit_hidden": 256,
        "dit_heads": 8,
        "condenc_hidden": 128,
        "condenc_out_emb": 256,
        "lr_cond": args.lr_cond,
        "posterior_sample": bool(args.posterior_sample),
        "vae_fixed": "sat_model_scale_<seed>/sat (200ep/1024, --vae_ckpt protocol)",
        "sampling": {"nfe": args.nfe, "cfg_scale": args.cfg, "solver": args.solver},
        "roi_res": args.roi_res,
        "sat": args.sat,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def grid_from_rows(rows):
    """从 JSON 行推导完整网格（种子 × 预算），避免单次调用只覆盖当前单元格。"""
    seeds = sorted({r["seed"] for r in rows})
    budgets = sorted({(r["train_data"], r["gen_epochs"]) for r in rows})
    return seeds, budgets


def compute_verdicts(rows):
    """P_scale 判定：每种子内按预算升序，FM NFE=1 CD 与 gap ratio 均单调非增。"""
    seeds, budgets = grid_from_rows(rows)
    order = sorted(range(len(budgets)), key=lambda i: (budgets[i][0], budgets[i][1]))
    per_seed = {}
    for sd in seeds:
        cell = {r["tag"]: r for r in rows if r["seed"] == sd}
        seq = [cell.get(budget_tag(*budgets[i])) for i in order]
        ok = [r for r in seq if r and r.get("status") == "ok"]
        complete = len(ok) == len(seq)
        cd_mono = all(ok[i]["fm_nfe1"]["cd"] >= ok[i + 1]["fm_nfe1"]["cd"] - 1e-12
                      for i in range(len(ok) - 1)) if len(ok) >= 2 else None
        gap_mono = all(ok[i]["gap_ratio"] >= ok[i + 1]["gap_ratio"] - 1e-12
                       for i in range(len(ok) - 1)) if len(ok) >= 2 else None
        per_seed[f"seed{sd}"] = {
            "cells_completed": f"{len(ok)}/{len(seq)}",
            "fm1_cd": [r["fm_nfe1"]["cd"] for r in ok],
            "gap_ratio": [r["gap_ratio"] for r in ok],
            "fm1_cd_monotone_nonincreasing": cd_mono,
            "gap_ratio_monotone_nonincreasing": gap_mono,
        }
    done = [per_seed[f"seed{sd}"] for sd in seeds]
    all_complete = all(d["cells_completed"] == f"{len(budgets)}/{len(budgets)}" for d in done)
    verdicts = {
        "P_scale_fm1_cd_monotone_nonincreasing": bool(
            all_complete and all(d["fm1_cd_monotone_nonincreasing"] for d in done)),
        "P_scale_gap_ratio_monotone_nonincreasing": bool(
            all_complete and all(d["gap_ratio_monotone_nonincreasing"] for d in done)),
        "per_seed": per_seed,
        "all_cells_completed": bool(all_complete),
    }
    if not all_complete:
        missing = [f"{sd}:{budget_tag(*b)}" for sd in seeds for b in budgets
                   if not any(r["seed"] == sd and r["tag"] == budget_tag(*b)
                              and r.get("status") == "ok" for r in rows)]
        verdicts["note"] = ("incomplete: cells not completed (timing guard): "
                            + ", ".join(missing))
    falsified = [sd for sd in seeds
                 if per_seed[f"seed{sd}"]["fm1_cd_monotone_nonincreasing"] is False
                 or per_seed[f"seed{sd}"]["gap_ratio_monotone_nonincreasing"] is False]
    if falsified:
        verdicts["note"] = ((verdicts.get("note", "") + " | " if verdicts.get("note") else "")
                            + "falsified on completed cells (non-monotone in budget): seeds "
                            + ", ".join(str(s) for s in falsified))
    return verdicts


def print_summary(rows, verdicts):
    seeds, budgets = grid_from_rows(rows)
    print(f"\n{'=' * 78}")
    print("FM 训练预算扫描汇总（NFE=1, CFG w=2, HRRP, fresh-data；CD/FS/IoU 为 8 样本公共测试批）")
    print(f"{'=' * 78}")
    print(f"{'seed':>6} {'budget':>10} {'CD':>8} {'FS@0.1':>8} {'FS@0.2':>8} {'IoU':>8} "
          f"{'oracleCD':>9} {'gap':>7} {'min':>7} {'status':>14}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['seed']:>6} {r['budget']:>10} {'-':>8} {'-':>8} {'-':>8} {'-':>8} "
                  f"{'-':>9} {'-':>7} {'-':>7} {r['status']:>14}")
            continue
        m, o = r["fm_nfe1"], r["vae_oracle"]
        print(f"{r['seed']:>6} {r['budget']:>10} {m['cd']:>8.4f} {m['fs_0.1']:>8.4f} "
              f"{m['fs_0.2']:>8.4f} {m['iou']:>8.4f} {o['cd']:>9.4f} {r['gap_ratio']:>6.0f}x "
              f"{r['runtime_min']:>7.1f} {'ok':>14}")
    print("-" * 78)
    print(f"  裁决: {json.dumps(verdicts, indent=2)}")


def main(args):
    seeds = args.seeds
    budgets = [parse_budget(b) for b in args.budgets]

    out = load_json(args.out)
    if out is None:
        out = {"rows": planned_rows(seeds, budgets)}
    else:
        have = {(r["seed"], r["tag"]) for r in out["rows"]}
        for r in planned_rows(seeds, budgets):
            if (r["seed"], r["tag"]) not in have:
                out["rows"].append(r)

    out["experiment"] = "fm_train_scale"
    out["title"] = ("FM training-budget sweep: does the end-to-end quality gap to the "
                    "VAE oracle close with FM training scale?")
    out["proposition"] = PROPOSITION
    out["proposition_registered_before_run"] = True
    out["design_notes"] = DESIGN_NOTES
    seeds_full, budgets_full = grid_from_rows(out["rows"])
    out["config"] = config_dict(args, seeds_full, budgets_full)
    out["verdicts"] = compute_verdicts(out["rows"])
    save_json(args.out, out)
    print(f"预注册/进度已写入: {args.out}")

    if args.plan_only:
        print("--plan_only：仅写预注册骨架，不训练。")
        return

    if args.mark_incomplete:
        sd_s, tag = args.mark_incomplete.split(":")
        for r in out["rows"]:
            if r["seed"] == int(sd_s) and r["tag"] == tag:
                r["status"] = "not completed"
                r["note"] = ("stopped by the timing guard (>40 min for the (2048,200) cell) "
                             "or otherwise interrupted; see run log for elapsed time")
        out["verdicts"] = compute_verdicts(out["rows"])
        save_json(args.out, out)
        print(f"已标记 not completed: seed {sd_s} {tag}")
        print_summary(out["rows"], out["verdicts"])
        return

    for sd in seeds:
        for s, e in budgets:
            tag = budget_tag(s, e)
            row = next((r for r in out["rows"] if r["seed"] == sd and r["tag"] == tag), None)
            if row is not None and row.get("status") == "ok":
                print(f"[skip] seed {sd} {tag} 已完成（CD={row['fm_nfe1']['cd']:.6f}）")
                continue
            res = run_cell(args, sd, s, e)
            if row is None:
                out["rows"].append(res)
            else:
                row.update(res)
            out["verdicts"] = compute_verdicts(out["rows"])
            save_json(args.out, out)
            print(f"结果已更新: {args.out}")

    print_summary(out["rows"], out["verdicts"])
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM 训练预算扫描（P_scale 预注册实验）")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--budgets", nargs="+", default=["512:60", "1024:100", "2048:200"],
                        help="训练预算 samples:epochs 列表")
    parser.add_argument("--vae_src", type=str, default="./sat_model_scale_{seed}",
                        help="固定 VAE 复用目录模板（--vae_ckpt 协议）")
    parser.add_argument("--save_dir", type=str, default="./sat_model_m3b_{seed}",
                        help="checkpoint 根目录模板（非 sat_model_scale_<seed>，防冲突）")
    parser.add_argument("--out", type=str, default="./sat_model_cmp/fm_train_scale.json")
    parser.add_argument("--plan_only", action="store_true", help="只写预注册骨架")
    parser.add_argument("--mark_incomplete", type=str, default=None,
                        help="标记某单元格未完成，如 42:b2048_200")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                        default="hrrp")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr_cond", type=float, default=1e-4)
    parser.add_argument("--posterior_sample", type=int, default=1)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    parser.add_argument("--roi_res", type=int, default=16)
    parser.add_argument("--sat", choices=["iss", "starlink"], default="iss")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, seeds={args.seeds}, budgets={args.budgets}")
    main(args)
