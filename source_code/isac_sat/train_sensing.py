"""
train_sensing.py — 训练感知模型（目标分类 + 定位）

为闭环 demo 提供实时感知器：
  - 输入：窄带通信接收信号特征 cond [tau, cond_dim]（快，CPU 实时）
  - 输出：类别（6 类）+ 目标位置（x,y 归一化坐标，来自点云质心）
  - 独立头：分类头 / 定位头（互不干扰）

工程导向：固定训练集（可复现）、保存 checkpoint、评估准确率/定位误差。
"""

import os
import math
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels, GROUND_TARGET_TEMPLATES

N_CLASSES = len(GROUND_TARGET_TEMPLATES)
CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]


class SensingMLP(nn.Module):
    """感知模型：分类 + 定位（独立头）。"""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, N_CLASSES))
        self.pos_head = nn.Sequential(
            nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, cond):
        feat = self.shared(cond.flatten(1))
        return self.cls_head(feat), self.pos_head(feat)


def make_fixed(dataset, n, wideband=False):
    samples = [dataset[i] for i in range(n)]
    if wideband:
        conds = torch.stack([s[2] for s in samples])   # 距离像特征
    else:
        conds = torch.stack([s[1] for s in samples])   # 窄带 cond 特征
    cls = torch.tensor([s[3] if wideband else s[2] for s in samples], dtype=torch.long)
    pos = torch.stack([s[0].mean(dim=0) for s in samples])[:, :2]
    return TensorDataset(conds, cls, pos)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    pos_err = 0.0
    for cond, cid, pos in loader:
        cond = cond.to(device)
        logits, pred_pos = model(cond)
        correct += (logits.argmax(1).cpu() == cid).sum().item()
        total += len(cid)
        pos_err += (pred_pos.cpu() - pos).abs().mean().item() * 2  # 归一化→实际范围
    return correct / total, pos_err / len(loader)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}, irs={args.irs_mode}, phase={args.phase_mode}")

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    cond_dim = channels.frame_cond_dim()
    print(f"cond_dim={cond_dim}")

    print(f"预生成训练/测试样本 (wideband={args.wideband})...")
    tr_ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                          device=device, tau=args.tau, phase_mode=args.phase_mode,
                          with_label=True, target_source="ground", wideband=args.wideband,
                          rp_align=not args.rp_align)
    te_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                          device=device, tau=args.tau, phase_mode=args.phase_mode,
                          with_label=True, target_source="ground", wideband=args.wideband,
                          rp_align=not args.rp_align)
    tr = DataLoader(make_fixed(tr_ds, args.train_data, wideband=args.wideband),
                    batch_size=args.batch_size, shuffle=True)
    te = DataLoader(make_fixed(te_ds, args.test_data, wideband=args.wideband),
                    batch_size=args.batch_size)

    feat_dim = 512 if args.wideband else args.tau * cond_dim
    model = SensingMLP(in_dim=feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for cond, cid, pos in tr:
            cond = cond.to(device); cid = cid.to(device); pos = pos.to(device)
            logits, pred_pos = model(cond)
            loss = F.cross_entropy(logits, cid) + args.pos_weight * F.mse_loss(pred_pos, pos)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        acc, pos_err = evaluate(model, te, device)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep+1:3d}/{args.epochs} loss={tot/len(tr):.4f} | "
                  f"cls acc={acc:.3f} pos err={pos_err:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(),
                        "cond_dim": cond_dim, "tau": args.tau,
                        "irs_mode": args.irs_mode, "phase_mode": args.phase_mode,
                        "wideband": args.wideband, "feat_dim": feat_dim},
                       os.path.join(args.save_dir, "sensing_best.pth"))

    print(f"\n[sensing] 最佳: cls acc={best_acc:.3f}, pos err={pos_err:.4f}")
    print(f"[sensing] checkpoint: {args.save_dir}/sensing_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练感知模型（分类+定位）")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="tracked")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--train_data", type=int, default=600)
    parser.add_argument("--test_data", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_points", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--rp_align", action="store_true", help="距离像质心对齐（默认 False：保留位置给定位）")
    parser.add_argument("--wideband", action="store_true", help="用宽带距离像特征（更精确）")
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./isac_demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
