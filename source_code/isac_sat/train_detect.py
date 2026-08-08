"""
train_detect.py — 10 目标检测器训练（分类 + 定位）

输入：单帧宽带距离像 [K=512]
输出：K=10 组 (类别 5 类 + 位置 x,y)
训练：预测组按位置排序与真实目标（按 x 排序）匹配，联合 loss。

数据：MovingTargetScene（N 个移动目标场景）的帧级样本。
物理：单站距离像对同距离单元目标分辨有限，检测器输出最佳 K 组，
      轨迹连续性由 MOT 追踪模块（mot_tracker.py）补足。

用法：
  python train_detect.py [--n_scenes 40 --epochs 60]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mot_data import MovingTargetScene, CLASS_NAMES
from data_sat import WIDEBAND_K

N_CLASSES = len(CLASS_NAMES)
K_MAX = 10


class DetectNet(nn.Module):
    """距离像 → K 组检测（共享编码 + K×(分类头+定位头)）。"""

    def __init__(self, in_dim=WIDEBAND_K, k=K_MAX, hidden=512):
        super().__init__()
        self.k = k
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        )
        self.cls_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, N_CLASSES))
             for _ in range(k)])
        self.pos_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 3))  # 3D
             for _ in range(k)])

    def forward(self, x):
        feat = self.shared(x)
        clss = [h(feat) for h in self.cls_heads]
        poss = [h(feat) for h in self.pos_heads]
        return clss, poss


def match_loss(clss, poss, targets, device):
    """预测 K 组与真实目标匹配（按 x 排序），联合 loss。"""
    B = poss[0].shape[0]
    loss_cls = loss_pos = 0.0
    for b in range(B):
        tg = sorted(targets[b], key=lambda t: t[1][0])
        n_t = len(tg)
        pred_pos = torch.stack([p[b] for p in poss])
        pred_cls = torch.stack([c[b] for c in clss])
        order = torch.argsort(pred_pos[:, 0])
        pred_pos, pred_cls = pred_pos[order], pred_cls[order]
        for k in range(K_MAX):
            if k < n_t:
                cid, (cx, cy, cz) = tg[k]
                loss_cls = loss_cls + F.cross_entropy(
                    pred_cls[k].unsqueeze(0), torch.tensor([cid], device=device))
                loss_pos = loss_pos + F.mse_loss(
                    pred_pos[k], torch.tensor([cx, cy, cz], dtype=torch.float32, device=device))
            else:
                # 空槽：推远位置 + 均匀类别（弱正则）
                loss_pos = loss_pos + 4.0 * F.mse_loss(
                    pred_pos[k], torch.tensor([2.0, 2.0, 2.0], device=device))
    return loss_cls / B, loss_pos / B


def build_dataset(n_scenes, n_frames, seed0, snr_db=20.0):
    """生成 n_scenes 个移动场景 → 帧级训练样本。"""
    rps_all, tg_all = [], []
    for s in range(n_scenes):
        scene = MovingTargetScene(n_targets=10, n_frames=n_frames, seed=seed0 + s)
        rps, gts = scene.range_profile_sequence(snr_db=snr_db)
        rps_all.append(rps)
        tg_all.extend(gts)
    rps = np.concatenate(rps_all, axis=0)
    return rps, tg_all


@torch.no_grad()
def evaluate(model, rps, targets, device, iou_thr=0.25):
    model.eval()
    total_t = detected = cls_ok = 0
    pos_err = 0.0
    clss, poss = model(torch.from_numpy(rps).float().to(device))
    B = rps.shape[0]
    for b in range(B):
        tg = sorted(targets[b], key=lambda t: t[1][0])
        pred_pos = torch.stack([p[b] for p in poss]).cpu().numpy()
        pred_cls = torch.stack([c[b] for c in clss]).argmax(1).cpu().numpy()
        order = np.argsort(pred_pos[:, 0])
        pred_pos, pred_cls = pred_pos[order], pred_cls[order]
        for k in range(min(K_MAX, len(tg))):
            cid, (cx, cy, cz) = tg[k]
            total_t += 1
            if np.linalg.norm(pred_pos[k] - np.array([cx, cy, cz])) < iou_thr:
                detected += 1
                cls_ok += (pred_cls[k] == cid)
                pos_err += np.linalg.norm(pred_pos[k] - np.array([cx, cy, cz]))
    return (detected / max(total_t, 1), cls_ok / max(total_t, 1),
            pos_err / max(total_t, 1))


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}, classes={CLASS_NAMES}, K={K_MAX}")

    print(f"生成训练数据 ({args.n_scenes} 场景 × {args.n_frames} 帧)...")
    tr_rps, tr_tg = build_dataset(args.n_scenes, args.n_frames, args.seed, args.snr_db)
    te_rps, te_tg = build_dataset(8, args.n_frames, args.seed + 1000, args.snr_db)
    print(f"训练样本: {tr_rps.shape[0]}, 测试样本: {te_rps.shape[0]}")

    model = DetectNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    best_det = 0.0
    n = tr_rps.shape[0]
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = torch.from_numpy(tr_rps[idx]).float().to(device)
            clss, poss = model(x)
            lc, lp = match_loss(clss, poss, [tr_tg[j] for j in idx.tolist()], device)
            loss = lc + args.pos_weight * lp
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        det, cls_acc, pos_e = evaluate(model, te_rps, te_tg, device)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep+1:3d}/{args.epochs} loss={tot:.4f} | "
                  f"detect={det:.3f} cls={cls_acc:.3f} pos_err={pos_e:.3f}")
        if det > best_det:
            best_det = det
            torch.save({"model": model.state_dict(), "k": K_MAX,
                        "n_classes": N_CLASSES, "classes": CLASS_NAMES},
                       os.path.join(args.save_dir, "detect_best.pth"))

    print(f"\n[detect] 最佳: detect={best_det:.3f}, cls={cls_acc:.3f}, pos_err={pos_e:.3f}")
    print(f"[detect] checkpoint: {args.save_dir}/detect_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10 目标检测器训练")
    parser.add_argument("--n_scenes", type=int, default=40)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--save_dir", type=str, default="./isac_demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
