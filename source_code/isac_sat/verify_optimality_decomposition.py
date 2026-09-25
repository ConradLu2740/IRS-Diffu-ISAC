"""
verify_optimality_decomposition.py — 闭环最优性分解（数学证书：最优性间隙）

把 README 里孤立的两个数字（闭环达成 oracle 73.3%、闭式达成数值上界 ~77%）
放进**同一条可复现链路**上做精确分解：

  系统实际达到的功率
      P_sensed  = 闭式相位(感知估计位置)        ← 系统现状（两阶段贪心）
  分解因子
      η_sense   = P_sensed / P_oracle_cf        ← 感知误差因子
      η_design  = P_oracle_cf / P_global        ← 相位设计因子（闭式 vs 全局最优）
  真全局最优（同场景可计算）
      P_global  = 坐标上升(真实位置, 多起点)     ← 数值全局上界参考

恒等式：η_sense × η_design = P_sensed / P_global（单场景精确成立）
→ 系统距离真全局最优的百分比 = 总达成率；两个因子各自可归因，
  任何"涨点"宣传都能被分解到具体环节（这就是最优性间隙证书）。

协议：多种子（每种子换真实目标样本 + 随机相位基线），与 demo.py 同一
场景/模型/相位设计器；数值上界用 optimize_frame_numeric（坐标上升）。

用法：python verify_optimality_decomposition.py --n_seeds 8
"""

import os
import json
import argparse
import numpy as np
import random
import torch

import setup_sat as ss
from data_sat import (SatScenarioChannels, compute_range_profile, _SIGNAL1,
                      generate_ground_target_sample)
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing import SensingMLP
from demo import estimate_roi_from_pos


def run_seed(seed, args):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]

    roi_true, cid_true, _ = generate_ground_target_sample()
    pc_true = (np.argwhere(roi_true > 0.5)).astype(np.float32)
    pos_true = pc_true.mean(axis=0) / 16.0 * 2.0 - 1.0

    rp = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0,
                               align=not args.rp_align, sat_ecef=mid["sat_pos"])
    rp_t = torch.from_numpy(rp).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, pred_pos = model(rp_t)
    pos_pred = pred_pos[0].cpu().numpy()
    pos_err = float(np.linalg.norm(pos_pred - pos_true[:2]))

    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    roi_true_t = torch.tensor(roi_true.astype(np.float32)).reshape(-1)
    roi_est_t = torch.tensor(estimate_roi_from_pos(pos_pred).astype(np.float32)).reshape(-1)

    p_rand, p_sensed, p_oracle_cf, p_global = [], [], [], []
    for t, Ht in enumerate(channels.channels_per_frame):
        ph_rand = torch.rand(Ht["H_ROI_IRS"].shape[1]) * 2 * np.pi
        p_rand.append(opt._power(Ht, roi_true_t, X, ph_rand))

        ph_sensed = opt.optimize_frame(Ht, roi_est_t, X)
        p_sensed.append(opt._power(Ht, roi_true_t, X, ph_sensed))

        ph_cf = opt.optimize_frame(Ht, roi_true_t, X)
        p_oracle_cf.append(opt._power(Ht, roi_true_t, X, ph_cf))

        _, p_num = opt.optimize_frame_numeric(Ht, roi_true_t, X, seed=t)
        p_global.append(p_num)

    res = {k: float(np.mean(v)) for k, v in
           [("random", p_rand), ("sensed", p_sensed),
            ("oracle_cf", p_oracle_cf), ("global_numeric", p_global)]}
    res["pos_err"] = pos_err
    res["cls_ok"] = bool(logits.argmax(1).item() == cid_true)
    return res


def main(args):
    rows = [run_seed(s, args) for s in range(args.n_seeds)]
    keys = ["random", "sensed", "oracle_cf", "global_numeric"]
    agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
               "std": float(np.std([r[k] for r in rows]))} for k in keys}
    eta_sense = [r["sensed"] / r["oracle_cf"] for r in rows]
    eta_design = [r["oracle_cf"] / r["global_numeric"] for r in rows]
    eta_total = [r["sensed"] / r["global_numeric"] for r in rows]

    print(f"\n{'=' * 74}")
    print("闭环最优性分解（多种子，均值±标准差）:")
    for k in keys:
        print(f"  {k:>14}: {agg[k]['mean']:.4e} ± {agg[k]['std']:.4e}")
    print("-" * 74)
    print(f"  η_sense  (感知误差因子)  : {np.mean(eta_sense):.3f} ± {np.std(eta_sense):.3f}")
    print(f"  η_design(相位设计因子)   : {np.mean(eta_design):.3f} ± {np.std(eta_design):.3f}")
    print(f"  总达成率 = η_sense×η_design: {np.mean(eta_total):.3f} ± {np.std(eta_total):.3f}")
    print("-" * 74)
    print(f"  系统 vs 随机基线提升      : {100 * (agg['sensed']['mean'] / agg['random']['mean'] - 1):+.1f}%")
    print(f"  真全局上界 vs 随机基线提升: {100 * (agg['global_numeric']['mean'] / agg['random']['mean'] - 1):+.1f}%")
    print(f"  感知定位平均误差          : {np.mean([r['pos_err'] for r in rows]):.3f}")
    print(f"{'=' * 74}")

    # Bootstrap 95% CI（对种子重采样）
    rng = np.random.default_rng(0)
    n = len(rows)
    def boot(x):
        x = np.asarray(x)
        means = [x[rng.integers(0, n, n)].mean() for _ in range(2000)]
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    ci = {k: boot(v) for k, v in
          [("eta_sense", eta_sense), ("eta_design", eta_design), ("eta_total", eta_total)]}
    print("Bootstrap 95% CI:")
    for k, (lo, hi) in ci.items():
        print(f"  {k:>10}: [{lo:.3f}, {hi:.3f}]")

    out = {
        "powers": agg,
        "bootstrap_ci95": ci,
        "eta_sense": {"mean": float(np.mean(eta_sense)), "std": float(np.std(eta_sense))},
        "eta_design": {"mean": float(np.mean(eta_design)), "std": float(np.std(eta_design))},
        "eta_total": {"mean": float(np.mean(eta_total)), "std": float(np.std(eta_total))},
        "n_seeds": args.n_seeds,
        "identity_check": float(np.mean(eta_sense) * np.mean(eta_design)),
    }
    json_path = os.path.join(os.path.dirname(args.checkpoint), "optimality_decomposition.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="闭环最优性分解")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--rp_align", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=8)
    args = parser.parse_args()
    main(args)
