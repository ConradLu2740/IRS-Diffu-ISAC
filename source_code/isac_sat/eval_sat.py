"""
eval_sat.py — 星-地 ISAC 评估：CD / F-Score / Voxel IoU + 可视化

评估已训练好的模型（sat_model/{mode}/ 下的 checkpoint）：
  - Chamfer Distance (CD): 基础重建误差
  - F-Score (τ=0.1/0.2):    点云结构正确性（precision/recall 调和平均）
  - Voxel IoU:              体素占用一致率（与感知/检测叙事直接相关）
  - 可视化:                 GT vs 采样点云 3D 对比 + 指标汇总

用法：
  python eval_sat.py --modes none sat ground --save_dir ./sat_model
"""

import os
import json
import argparse
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader

_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import DDPMScheduler, sample_conditional_1D, chamfer_distance_loss

TAUS = [0.1, 0.2]


# ----------------------------------------------------------------------
# 指标实现
# ----------------------------------------------------------------------

def f_score(gt, pred, tau=0.1):
    """F-Score：距离 < tau 的点对比例（precision/recall 调和平均）。

    gt, pred: [N, 3] 点云（归一化坐标）
    """
    d = torch.cdist(gt.unsqueeze(0), pred.unsqueeze(0)).squeeze(0)  # [N, N]
    prec = (d.min(dim=1)[0] < tau).float().mean().item()
    rec = (d.min(dim=0)[0] < tau).float().mean().item()
    if prec + rec < 1e-12:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def voxelize(pc, res=16, lo=-1.0, hi=1.0):
    """把 [-1,1] 点云量化为 [res]³ 体素占据。"""
    idx = ((pc - lo) / (hi - lo) * res).long().clamp(0, res - 1)
    vox = torch.zeros((res, res, res), dtype=torch.bool, device=pc.device)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vox


def voxel_iou(gt, pred, res=16):
    """体素 IoU。"""
    vg = voxelize(gt, res)
    vp = voxelize(pred, res)
    inter = (vg & vp).sum().item()
    union = (vg | vp).sum().item()
    return inter / max(union, 1)


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------

def visualize_pairs(pc_gts, pc_hats, mode, save_dir, n_show=3):
    """GT vs 预测点云 3D 对比。pc_gts: list[tensor]"""
    n = min(n_show, len(pc_gts))
    fig = plt.figure(figsize=(4 * n, 8))
    for i in range(n):
        for j, (pc, title) in enumerate([(pc_gts[i], "GT"), (pc_hats[i], "Pred")]):
            ax = fig.add_subplot(2, n, i + 1 + j * n, projection="3d")
            p = pc.detach().cpu().numpy()
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.2, alpha=0.7,
                       c="steelblue" if j == 0 else "crimson")
            ax.set_title(f"{title} #{i}")
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"IRS mode = {mode}: GT vs Sampled Point Cloud")
    fig.tight_layout()
    path = os.path.join(save_dir, f"compare_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_box(metrics, save_dir):
    """CD 分布箱线图（各模式）。"""
    modes = list(metrics.keys())
    cds = [metrics[m]["cd"]["list"] for m in modes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(cds, labels=modes)
    ax.set_ylabel("Chamfer Distance")
    ax.set_title("Sampling CD Distribution by IRS Mode")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(save_dir, "cd_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 主评估
# ----------------------------------------------------------------------

def evaluate_mode(mode, args):
    device = args.device
    print(f"\n[{mode}] 加载模型并评估...")
    save_dir = os.path.join(args.save_dir, mode)
    if not os.path.exists(os.path.join(save_dir, "vae_best.pth")):
        print(f"[{mode}] 缺少 checkpoint，跳过。")
        return None

    # 场景 + 测试数据
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=mode, device=device)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    cond_dim = channels.frame_cond_dim()

    # 模型
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(save_dir, "vae_best.pth"), map_location=device))
    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    condenc.load_state_dict(torch.load(os.path.join(save_dir, "condenc_best.pth"), map_location=device))
    epsnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                   depth=args.depth, num_heads=8).to(device)
    epsnet.load_state_dict(torch.load(os.path.join(save_dir, "epsnet_best.pth"), map_location=device))
    stats = torch.load(os.path.join(save_dir, "latent_stats.pth"), map_location=device)
    z_mean, z_std = stats["z_mean"], stats["z_std"]
    sched = DDPMScheduler(T=args.T, device=device)

    # 采样 + 指标
    metrics = {"cd": [], "fs_0.1": [], "fs_0.2": [], "iou": []}
    pc_gts, pc_hats = [], []
    with torch.no_grad():
        for pc_gt, cond in test_loader:
            pc_gt = pc_gt[:args.n_eval].to(device)
            cond = cond[:args.n_eval].to(device)
            pc_hat = sample_conditional_1D(vae, condenc, epsnet, sched, cond,
                                           z_mean, z_std, device=device, cfg_scale=2.0)
            for i in range(pc_gt.shape[0]):
                cd = chamfer_distance_loss(pc_gt[i:i+1], pc_hat[i:i+1]).item()
                metrics["cd"].append(cd)
                metrics["fs_0.1"].append(f_score(pc_gt[i], pc_hat[i], 0.1))
                metrics["fs_0.2"].append(f_score(pc_gt[i], pc_hat[i], 0.2))
                metrics["iou"].append(voxel_iou(pc_gt[i], pc_hat[i], args.roi_res))
            pc_gts.append(pc_gt)
            pc_hats.append(pc_hat)

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "list": v}
               for k, v in metrics.items()}

    # 可视化
    vis_path = visualize_pairs(pc_gts[0], pc_hats[0], mode, args.save_dir, n_show=3)
    print(f"[{mode}] 图已保存: {vis_path}")
    return summary


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    for mode in args.modes:
        summary = evaluate_mode(mode, args)
        if summary is not None:
            results[mode] = summary

    if not results:
        print("没有可评估的模型 checkpoint，先运行 train_sat.py 训练。")
        return

    # 汇总表
    print("\n" + "=" * 78)
    print(f"{'模式':<10}{'CD↓':>12}{'F-Score@0.1↑':>14}{'F-Score@0.2↑':>14}{'Voxel IoU↑':>12}")
    print("-" * 78)
    for mode, m in results.items():
        print(f"{mode:<10}{m['cd']['mean']:12.4f}{m['fs_0.1']['mean']:14.4f}"
              f"{m['fs_0.2']['mean']:14.4f}{m['iou']['mean']:12.4f}")
    print("=" * 78)

    # 箱线图
    box_path = plot_metric_box(results, args.save_dir)
    print(f"CD 箱线图: {box_path}")

    # 保存 json
    out = {mode: {k: {"mean": v["mean"], "std": v["std"]} for k, v in m.items()}
           for mode, m in results.items()}
    json_path = os.path.join(args.save_dir, "eval_metrics.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"指标已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星-地 ISAC 评估")
    parser.add_argument("--modes", nargs="+", choices=["none", "sat", "ground"],
                        default=["none", "sat", "ground"])
    parser.add_argument("--save_dir", type=str, default="./sat_model")
    parser.add_argument("--test_data", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--roi_res", type=int, default=16)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random",
                        help="IRS 相位模式（需与训练一致）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}, modes={args.modes}, T={args.T}")
    main(args)
