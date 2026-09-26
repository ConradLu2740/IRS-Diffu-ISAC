"""
verify_m3_paired.py — M3（生成训练 100ep → 400ep，VAE 固定）的同批次逐云配对重评

动机：verify_gen_scale.py 只是聚合器——它比较的 sat_model_scale_<seed>
（生成 100ep 基线）与 sat_model_m3_<seed>（400ep）各自目录下的 compare_gen.json
由**独立运行**产生，每份的测试批都是训练消耗全局 RNG 之后现采样的
（SatROIDataset.__getitem__ 用全局 random/torch.rand），因此 M3 的
"4× 训练不降 CD" 结论同样混有批噪声，Ga/Gb 的证伪是软的。

本脚本在两个模型集之间做**同批次、同 x0** 的逐云配对重评：
公共批构造与 verify_fm_train_scale_paired.py 完全一致（每种子重种子后取首批 8 云，
指纹记录），VAE 固定复用 sat_model_scale_<seed>/sat 的缩放版（与 M3 协议一致），
FM 取各自的 *_latest.pth 最终权重，NFE ∈ {1, 10}，CFG w=2，Euler。

判读口径沿用 P_paired（见 verify_fm_train_scale_paired.PROPOSITION）：
median(Δ) ≤ −0.02 且 ≥6/8 云改善 → consistent improvement；
median(Δ) ≥ +0.02 且 ≥6/8 云恶化 → consistent degradation；
其余 → no consistent effect at n=8。
另报原始 Ga 口径（NFE=10 均值 CD 降幅 ≥20%？）在配对批次上的值。
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from verify_fm_train_scale_paired import (
    build_common_batch, per_cloud_cd, read_delta, budget_tag,
)

MODEL_DIR = "./sat_model_{tag}_{seed}/sat"   # tag ∈ {scale, m3}
VAE_DIR = "./sat_model_scale_{seed}/sat"      # 两集共用同一缩放版 VAE（M3 协议）


def load_fm(args, tag, seed, vae, z_mean, z_std, pc_gt, cond, nfe):
    from models import AdvancedCondEncoder, LatentDiT1D_CrossAttn
    from fm_utils import sample_conditional_FM
    d = MODEL_DIR.format(tag=tag, seed=seed)
    ce_p, vn_p = os.path.join(d, "condenc_fm_latest.pth"), os.path.join(d, "vnet_fm_latest.pth")
    if not (os.path.exists(ce_p) and os.path.exists(vn_p)):
        return None
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=args.cond_dim,
                                  hidden_size=128, out_emb=256).to(args.device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(args.device)
    condenc.load_state_dict(torch.load(ce_p, map_location=args.device))
    vnet.load_state_dict(torch.load(vn_p, map_location=args.device))
    torch.manual_seed(9000 + seed)   # 与 P_scale 配对重评同一 x0 种子
    import random as _random
    _random.seed(9000 + seed)
    np.random.seed(9000 + seed)
    pc_hat = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                   device=args.device, cfg_scale=args.cfg,
                                   nfe=nfe, solver=args.solver)
    cds, mean_cd = per_cloud_cd(pc_gt, pc_hat)
    return {"per_cloud_cd": cds, "cd_mean": round(mean_cd, 6)}


def run_seed(args, seed):
    from models import PointVAE
    pc_gt, cond, fp = build_common_batch(args, seed)
    print(f"[seed {seed}] common batch fingerprint={fp}")
    vae_src = VAE_DIR.format(seed=seed)
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(args.device)
    vae.load_state_dict(torch.load(os.path.join(vae_src, "vae_best.pth"),
                                   map_location=args.device))
    stats = torch.load(os.path.join(vae_src, "latent_stats.pth"), map_location=args.device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]
    with torch.no_grad():
        mu, _ = vae.encode(pc_gt)
        pc_vae, _ = vae.decode(mu)
    _, oracle_cd = per_cloud_cd(pc_gt, pc_vae)

    res = {"batch_fingerprint": fp, "oracle_cd_mean": round(oracle_cd, 6), "cells": {}}
    for nfe in args.nfes:
        base = load_fm(args, "scale", seed, vae, z_mean, z_std, pc_gt, cond, nfe)
        m3 = load_fm(args, "m3", seed, vae, z_mean, z_std, pc_gt, cond, nfe)
        if base is None or m3 is None:
            res["cells"][f"nfe{nfe}"] = {"status": "not completed",
                                         "note": "missing final checkpoints"}
            continue
        deltas = [round(a - b, 6) for a, b in zip(m3["per_cloud_cd"], base["per_cloud_cd"])]
        reading = read_delta(deltas)
        drop_pct = 100.0 * (base["cd_mean"] - m3["cd_mean"]) / base["cd_mean"]
        res["cells"][f"nfe{nfe}"] = {
            "status": "ok",
            "base_100ep": base, "m3_400ep": m3,
            "delta_per_cloud": deltas, **reading,
            "mean_cd_drop_pct": round(drop_pct, 2),
            "gap_ratio_base": round(base["cd_mean"] / oracle_cd, 2),
            "gap_ratio_m3": round(m3["cd_mean"] / oracle_cd, 2),
        }
        r = res["cells"][f"nfe{nfe}"]
        print(f"[seed {seed}] NFE={nfe}: base(100ep) CD={base['cd_mean']:.4f} → "
              f"m3(400ep) CD={m3['cd_mean']:.4f} ({drop_pct:+.1f}%), "
              f"median Δ={reading['median_delta']:+.4f}, "
              f"{reading['n_improved']}/8 improved, reading={reading['reading']}")
    readings = [c["reading"] for c in res["cells"].values() if c.get("status") == "ok"]
    res["seed_verdict"] = ("PASS" if readings and all(r == "consistent improvement"
                                                      for r in readings) else "not PASS")
    return res


def main(args):
    t0 = time.time()
    pc_gt, cond, fp = build_common_batch(args, args.seeds[0])
    args.cond_dim = cond.shape[-1]
    out = {
        "experiment": "m3_paired",
        "title": "M3 (generative training 100ep -> 400ep, VAE fixed): paired per-cloud "
                 "re-evaluation on a common batch",
        "motivation": ("verify_gen_scale.py aggregates two independently produced "
                       "compare_gen.json files (sat_model_scale_<seed> vs "
                       "sat_model_m3_<seed>); each drew its own post-training test batch "
                       "from the global RNG, so the original Ga/Gb falsification is "
                       "batch-confounded. This file is the paired comparison on one fixed "
                       "common batch per seed with shared x0, same protocol as "
                       "verify_fm_train_scale_paired.py."),
        "proposition_registered_before_run": True,
        "proposition": ("Paired reading of M3 on a common batch: 400ep improves over "
                        "100ep iff median(Δ) <= -0.02 and >= 6/8 clouds improve, at each "
                        "of NFE 1 and 10; otherwise 'no consistent effect at n=8'. The "
                        "original Ga criterion (mean CD drop >= 20% at NFE=10) is also "
                        "reported on the paired batch."),
        "config": {"seeds": args.seeds, "nfes": args.nfes, "vae_frozen_from": VAE_DIR,
                   "model_dirs": MODEL_DIR, "cfg": args.cfg,
                   "x0_seed_offset": 9000, "device": args.device},
        "seeds": {}, "verdicts": {},
    }
    for sd in args.seeds:
        out["seeds"][str(sd)] = run_seed(args, sd)
    out["verdicts"] = {str(sd): out["seeds"][str(sd)]["seed_verdict"] for sd in args.seeds}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nM3 配对重评完成（{time.time() - t0:.1f}s）→ {args.out}")
    print(f"裁决: {json.dumps(out['verdicts'], ensure_ascii=False)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="M3 同批次逐云配对重评")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--nfes", nargs="+", type=int, default=[1, 10])
    p.add_argument("--out", type=str, default="./sat_model_cmp/m3_paired.json")
    p.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    p.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                   default="narrowband")  # M3/scale 当次运行即 narrowband（LSTM 输入 61 维）
    p.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    p.add_argument("--test_data", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_eval", type=int, default=8)
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--tau", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    p.add_argument("--roi_res", type=int, default=16)
    p.add_argument("--sat", choices=["iss", "starlink"], default="iss")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, seeds={args.seeds}, nfes={args.nfes}")
    main(args)
