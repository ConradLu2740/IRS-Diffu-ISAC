"""
task_sat.py — 星-地 ISAC 感知任务升级：目标分类 + 姿态估计

通感一体核心：只用通信接收信号的特征（幅值dB/相位/IRS相位/动态/RCS功率）
做感知，不假设理想 CSI、不用 GT 点云参与推理。

物理基础：
  - 类别 → RCS/形状差异 → 回波功率与多径模式不同
  - 姿态 → 散射体分布变化 → 接收信号幅度/相位模式随姿态变化
  - 微多普勒 → 运动部件（旋翼/车辆）调制回波（RATR 方法）
  - 物理结论（已验证）：窄带观测下 6 类目标分类上界 ~0.35-0.45，
    姿态在窄带下不敏感（MAE≈90°）；tracked 相位（感知赋形）优于 random

通信工程利用：
  - 特征用 dB 尺度（功率/幅度对数表示，通信标准）
  - 对比 phase_mode random vs tracked：量化"RIS 感知波束赋形"价值

评估协议：固定测试集（seed 固定）+ 在线训练（数据增强）。
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import setup_sat as ss
from data_sat import (SatROIDataset, SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      WIDEBAND_K, ISAR_M)
from models import AdvancedCondEncoder

N_CLASSES = len(GROUND_TARGET_TEMPLATES)
CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]


def ss_wb_dim():
    return WIDEBAND_K


def ss_isar_dim():
    return ISAR_M * WIDEBAND_K


class MLPBackbone(nn.Module):
    """MLP 骨干（双分支）：cond 分支 + 距离像分支 → 融合 → 特征。"""

    def __init__(self, in_dim, rp_dim=0, hidden=256):
        super().__init__()
        self.rp_dim = rp_dim
        self.rp_net = None
        self.pose_rp_net = None
        if rp_dim > 0:
            self.rp_net = nn.Sequential(
                nn.Linear(rp_dim, 64), nn.BatchNorm1d(64), nn.ReLU())
            # 姿态独立编码器（不与分类共享，避免分类主导）
            self.pose_rp_net = nn.Sequential(
                nn.Linear(rp_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(in_dim + (64 if rp_dim > 0 else 0), hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        )
        self.out_dim = hidden // 2

    def forward(self, cond, rp=None):
        x = cond.flatten(1) if cond is not None else torch.zeros(rp.size(0), 0)
        r = None
        rp_pose = None
        if rp is not None and self.rp_net is not None:
            if rp.dim() > 2:
                rp = rp.flatten(1)      # [B, M*K] ISAR 序列展平
            r = self.rp_net(rp)          # [B, 64] 分类用 rp 编码
            rp_pose = self.pose_rp_net(rp)  # [B, 64] 姿态独立编码
            x = torch.cat([x, r], dim=1)
        fused = self.net(x)
        return fused, rp_pose             # (融合特征, 姿态专用特征)


class CondEncBackbone(nn.Module):
    """AdvancedCondEncoder 骨干（可选，LSTM+Transformer）。"""

    def __init__(self, seq_len, input_size, out_emb=256):
        super().__init__()
        self.enc = AdvancedCondEncoder(seq_len=seq_len, input_size=input_size,
                                       hidden_size=128, out_emb=out_emb)
        self.out_dim = out_emb

    def forward(self, cond):
        c_seq = self.enc(cond)               # [B, seq, out_emb]
        return c_seq.mean(dim=1)             # [B, out_emb]


class PerceptionHead(nn.Module):
    """感知头：分类用融合特征，姿态用 rp 特征（任务分离，避免分类主导）。"""

    def __init__(self, in_dim, rp_dim=0, n_classes=N_CLASSES, hidden=128):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        pose_in = rp_dim if rp_dim > 0 else in_dim
        self.pose_head = nn.Sequential(
            nn.Linear(pose_in, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, feat, rp_feat=None):
        logits = self.cls_head(feat)
        pose_sc = self.pose_head(rp_feat if rp_feat is not None else feat)
        return logits, pose_sc


def pose_target(angle_deg):
    theta = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def angle_error_deg(pred_sc, angle_deg):
    pred_theta = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    gt_theta = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
    diff = (pred_theta - gt_theta + math.pi) % (2 * math.pi) - math.pi
    return torch.abs(torch.rad2deg(diff))


def make_fixed_samples(dataset, n, wideband=False):
    """预生成固定样本列表（可复现评估）。"""
    samples = [dataset[i] for i in range(n)]
    pcs = torch.stack([s[0] for s in samples])
    conds = torch.stack([s[1] for s in samples])
    cls = torch.tensor([s[2 + int(wideband)] for s in samples], dtype=torch.long)
    ang = torch.tensor([s[3 + int(wideband)] for s in samples], dtype=torch.float32)
    if wideband:
        rps = torch.stack([s[2] for s in samples])
        return TensorDataset(pcs, conds, rps, cls, ang)
    return TensorDataset(pcs, conds, cls, ang)


def make_online_train_loader(channels, args, device):
    """每 epoch 生成新训练样本（数据增强）。"""
    ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                       device=device, tau=args.tau, phase_mode=args.phase_mode,
                       with_label=True, target_source="ground",
                       wideband=args.wideband, wideband_snr_db=args.wideband_snr_db,
                       isar=args.isar)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True)


@torch.no_grad()
def evaluate(model, head, loader, device, wideband=False, rp_only=False):
    model.eval(); head.eval()
    correct, total, mae_sum = 0, 0, 0.0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for batch in loader:
        if wideband:
            pc, cond, rp, cid, angle = batch
            feat = model(None if rp_only else cond.to(device), rp.to(device))
        else:
            pc, cond, cid, angle = batch
            feat = model(cond.to(device))
        fused, rp_feat = feat if isinstance(feat, tuple) else (feat, None)
        logits, pose_sc = head(fused, rp_feat)
        pred = logits.argmax(dim=1)
        for i in range(len(cid)):
            cm[cid[i].item(), pred[i].item()] += 1
        correct += (pred.cpu() == cid).sum().item()
        total += len(cid)
        mae_sum += angle_error_deg(pose_sc, angle).mean().item() * len(angle)
    return correct / total, mae_sum / total, cm


def plot_confusion(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Target Classification Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}, irs={args.irs_mode}, phase={args.phase_mode}, "
          f"backbone={args.backbone}, classes={CLASS_NAMES}")

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    cond_dim = channels.frame_cond_dim()
    print(f"cond_dim={cond_dim}")

    # ---- 固定测试集（可复现） ----
    print(f"预生成测试样本 ({args.test_data}), wideband={args.wideband}, isar={args.isar}...")
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            with_label=True, target_source="ground",
                            wideband=args.wideband, wideband_snr_db=args.wideband_snr_db,
                            isar=args.isar)
    test_fixed = make_fixed_samples(test_ds, args.test_data, wideband=args.wideband or args.isar)
    test_loader = DataLoader(test_fixed, batch_size=args.batch_size, shuffle=False)

    # ---- 模型 ----
    if args.isar:
        rp_dim = ss_isar_dim()
    elif args.wideband:
        rp_dim = ss_wb_dim()
    else:
        rp_dim = 0
    cond_used = args.tau * cond_dim if not args.rp_only else 0
    if args.backbone == "mlp":
        model = MLPBackbone(in_dim=cond_used, rp_dim=rp_dim)
    else:
        model = CondEncBackbone(seq_len=args.tau, input_size=cond_dim)
    head = PerceptionHead(in_dim=model.out_dim,
                          rp_dim=64 if (args.wideband or args.isar) else 0,
                          n_classes=N_CLASSES).to(device)
    model.to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc, best_mae = 0.0, 999.0
    for ep in range(args.epochs):
        model.train(); head.train()
        train_loader = make_online_train_loader(channels, args, device)
        tot_loss = tot_cls = tot_pose = 0.0
        for batch in train_loader:
            if args.wideband or args.isar:
                pc, cond, rp, cid, angle = batch
                cond = cond.to(device); rp = rp.to(device)
            else:
                pc, cond, cid, angle = batch
                cond = cond.to(device)
            cid = cid.to(device); angle = angle.to(device)
            feat = model(None if args.rp_only else cond,
                         rp if (args.wideband or args.isar) else None)
            fused, rp_feat = feat if isinstance(feat, tuple) else (feat, None)
            logits, pose_sc = head(fused, rp_feat)
            loss_cls = F.cross_entropy(logits, cid)
            loss_pose = F.mse_loss(pose_sc, pose_target(angle).to(device))
            loss = loss_cls + args.pose_weight * loss_pose
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); tot_cls += loss_cls.item(); tot_pose += loss_pose.item()

        acc, mae, cm = evaluate(model, head, test_loader, device,
                                wideband=args.wideband or args.isar,
                                rp_only=args.rp_only)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep+1:3d}/{args.epochs} loss={tot_loss/len(train_loader):.4f} "
                  f"| test acc={acc:.3f} pose MAE={mae:.1f}°")
        if acc > best_acc:
            best_acc, best_mae = acc, mae
            torch.save({"model": model.state_dict(), "head": head.state_dict()},
                       os.path.join(args.save_dir, f"perception_best_{args.phase_mode}.pth"))
            np.save(os.path.join(args.save_dir, f"confusion_{args.phase_mode}.npy"), cm)
            plot_confusion(cm, os.path.join(args.save_dir, f"confusion_{args.phase_mode}.png"))

    print(f"\n[{args.irs_mode}|{args.phase_mode}|{args.backbone}] 最佳: "
          f"acc={best_acc:.3f}, pose MAE={best_mae:.1f}°")
    return {"irs": args.irs_mode, "phase": args.phase_mode,
            "backbone": args.backbone, "acc": best_acc, "mae_deg": best_mae}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星-地 ISAC 目标分类 + 姿态估计")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="tracked")
    parser.add_argument("--backbone", choices=["mlp", "condenc"], default="mlp")
    parser.add_argument("--train_data", type=int, default=300)
    parser.add_argument("--test_data", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_points", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pose_weight", type=float, default=0.5)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./sat_perception")
    parser.add_argument("--bs_ant", type=int, default=4, help="卫星天线数")
    parser.add_argument("--ue_ant", type=int, default=4, help="地面站天线数")
    parser.add_argument("--isar", action="store_true", help="使用 ISAR 距离-时间序列（目标转动）")
    parser.add_argument("--rp_only", action="store_true", help="只用距离像特征（姿态最佳）")
    parser.add_argument("--wideband", action="store_true", help="使用宽带距离像特征")
    parser.add_argument("--wideband_snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
