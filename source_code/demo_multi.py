"""
demo_multi.py — 多目标感知 + IRS 指向闭环演示

ROI 内 1-2 个目标 → 感知（K=2 组分类+定位）→ IRS 指向检测到的目标 → 通信增益。

工程说明：
  - 感知模型由 train_sensing_multi.py 训练（checkpoint: sensing_multi_best.pth）
  - 多目标信号在单站观测下混合，分类难度高（物理限制），定位/检测可用
"""

import os
import argparse
import numpy as np
import torch

import setup_sat as ss
from data_sat import (SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      compute_range_profile, _SIGNAL1, generate_multi_target_sample)
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing_multi import SensingMLPMulti, K_MAX

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLPMulti(in_dim=ckpt["feat_dim"], k=ckpt["k"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)

    # ---- 多目标场景 ----
    roi, targets = generate_multi_target_sample(n_max=2)
    print(f"[demo_multi] 真实目标 {len(targets)} 个:")
    for cid, (cx, cy) in targets:
        print(f"  - {CLASS_NAMES[cid]} @ ({cx:+.2f}, {cy:+.2f})")

    # ---- 感知 ----
    rp = compute_range_profile(roi, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0, align=False)
    with torch.no_grad():
        clss, poss = model(torch.from_numpy(rp).float().unsqueeze(0))
    pred_pos = torch.stack(poss).squeeze(1).numpy()
    pred_cls = torch.stack(clss).squeeze(1).argmax(1).numpy()
    order = np.argsort(pred_pos[:, 0])
    pred_pos = pred_pos[order]; pred_cls = pred_cls[order]

    print("\n[demo_multi] 感知输出:")
    det = 0
    for k in range(K_MAX):
        print(f"  - 目标{k}: {CLASS_NAMES[pred_cls[k]]} @ ({pred_pos[k,0]:+.2f}, {pred_pos[k,1]:+.2f})")
    # 检测统计
    for cid, (cx, cy) in targets:
        if any(np.linalg.norm(p - np.array([cx, cy])) < 0.5 for p in pred_pos):
            det += 1
    print(f"[demo_multi] 检测: {det}/{len(targets)}")

    # ---- IRS 指向检测到的目标（用第一个目标的估计位置构建 ROI） ----
    roi_true_t = torch.tensor(roi.astype(np.float32)).reshape(-1)
    res = 16
    # 用感知到的第一个目标位置
    px, py = pred_pos[0]
    roi_est = np.zeros((res, res, res), dtype=np.float32)
    cx = int((px + 1) / 2 * res); cy = int((py + 1) / 2 * res)
    roi_est[max(cx - 2, 0):min(cx + 3, res), max(cy - 2, 0):min(cy + 3, res), 7:9] = 1.0
    roi_est_t = torch.tensor(roi_est.astype(np.float32)).reshape(-1)

    powers = {"random": [], "sensed": [], "oracle": []}
    for Ht in channels.channels_per_frame:
        n_irs = Ht["H_ROI_IRS"].shape[1]
        powers["random"].append(opt._power(Ht, roi_true_t, X, torch.rand(n_irs) * 2 * np.pi))
        powers["sensed"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_est_t, X)))
        powers["oracle"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_true_t, X)))

    p_rand = float(np.mean(powers["random"]))
    p_sen = float(np.mean(powers["sensed"]))
    p_ora = float(np.mean(powers["oracle"]))
    print("\n" + "=" * 62)
    print("通信性能（IRS 指向感知目标）：")
    print(f"  random : {p_rand:.4e}")
    print(f"  sensed : {p_sen:.4e}  ({100*(p_sen/p_rand-1):+.1f}%)")
    print(f"  oracle : {p_ora:.4e}  ({100*(p_ora/p_rand-1):+.1f}%)")
    print(f"  达成率 : {100*p_sen/p_ora:.0f}%")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多目标感知-通信闭环")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_multi_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
