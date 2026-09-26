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
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment

from mot_data import MovingTargetScene, CLASS_NAMES
from data_sat import WIDEBAND_K

N_CLASSES = len(CLASS_NAMES)
K_MAX = 10


class DetectNet(nn.Module):
    """距离像 → K 组检测（共享编码 + K×(分类头+定位头)）+ 计数头（D3）。

    计数头预测场景目标数 n ∈ [0, K_MAX]；推理时输出按置信度排序的前 n 个
    槽（top-K 选择），替代固定阈值过滤——可变计数检测头。
    """

    def __init__(self, in_dim=WIDEBAND_K, k=K_MAX, hidden=512, count_head=True, stack=1,
                 obj_head=False):
        super().__init__()
        self.k = k
        self.count_head = count_head
        self.obj_head = obj_head
        self.stack = stack
        self.shared = nn.Sequential(
            nn.Linear(in_dim * stack, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
        )
        self.cls_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, N_CLASSES))
             for _ in range(k)])
        self.pos_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 3))  # 3D
             for _ in range(k)])
        if count_head:
            self.count = nn.Sequential(
                nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Linear(64, k + 1))
        if obj_head:
            self.obj_heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1))
                 for _ in range(k)])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)          # [B, stack*K] 展平（stack=1 兼容 [B,K]/[B,1,K]）
        feat = self.shared(x)
        clss = [h(feat) for h in self.cls_heads]
        poss = [h(feat) for h in self.pos_heads]
        cnt = self.count(feat) if self.count_head else None
        objs = [h(feat).squeeze(-1) for h in self.obj_heads] if self.obj_head else None
        return clss, poss, cnt, objs


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


def match_loss_hungarian(clss, poss, targets, device, objs=None):
    """DETR 式集合预测损失：每样本匈牙利匹配（位置代价）后对匹配对施加
    CE+MSE，未匹配槽推远。与 match_loss（x 排序配对）的区别：分配是
    最优双射而非排序启发式——直接消解槽位分配歧义（§7.19 诊断的根因）。"""
    B = poss[0].shape[0]
    loss_cls = loss_pos = loss_obj = 0.0
    for b in range(B):
        tg = targets[b]
        n_t = len(tg)
        pred_pos = torch.stack([p[b] for p in poss])      # [K, 3]
        pred_cls = torch.stack([c[b] for c in clss])      # [K, C]
        if objs is not None:
            pred_obj = torch.stack([o[b] for o in objs])  # [K]
            obj_target = torch.zeros_like(pred_obj)
        if n_t > 0:
            gt_pos = torch.tensor([t[1] for t in tg], dtype=torch.float32, device=device)
            gt_cls = torch.tensor([t[0] for t in tg], device=device)
            with torch.no_grad():
                cost = torch.cdist(pred_pos, gt_pos).cpu().numpy()
                rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                loss_cls = loss_cls + F.cross_entropy(
                    pred_cls[r].unsqueeze(0), gt_cls[c:c + 1])
                loss_pos = loss_pos + F.mse_loss(pred_pos[r], gt_pos[c])
                if objs is not None:
                    obj_target[r] = 1.0
            matched = set(rows.tolist())
        else:
            matched = set()
        for k in range(K_MAX):
            if k not in matched:
                loss_pos = loss_pos + 4.0 * F.mse_loss(
                    pred_pos[k], torch.tensor([2.0, 2.0, 2.0], device=device))
        if objs is not None:
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(pred_obj, obj_target)
    if objs is not None:
        return loss_cls / B, loss_pos / B, loss_obj / B
    return loss_cls / B, loss_pos / B, None


def build_dataset(n_scenes, n_frames, seed0, snr_db=20.0, n_targets_range=None, stack=1,
                  n_avg=1):
    """生成 n_scenes 个移动场景 → 帧级训练样本。

    n_targets_range: (lo, hi) 时每场景随机目标数（D4 变计数 benchmark）；
    None 时固定 10（旧行为）。
    """
    rps_all, tg_all = [], []
    rng = random.Random(seed0)
    for s in range(n_scenes):
        n_t = rng.randint(*n_targets_range) if n_targets_range else 10
        scene = MovingTargetScene(n_targets=n_t, n_frames=n_frames, seed=seed0 + s)
        rps, gts = scene.range_profile_sequence(snr_db=snr_db, stack=stack, n_avg=n_avg)
        rps_all.append(rps)
        tg_all.extend(gts)
    rps = np.concatenate(rps_all, axis=0)
    return rps, tg_all


@torch.no_grad()
def evaluate(model, rps, targets, device, iou_thr=0.25):
    model.eval()
    total_t = detected = cls_ok = 0
    pos_err = 0.0
    _x = torch.from_numpy(rps).float().to(device)
    if _x.dim() == 2:
        _x = _x.unsqueeze(1)
    clss, poss, _cnt, _objs = model(_x)
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
    torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}, classes={CLASS_NAMES}, K={K_MAX}")

    print(f"生成训练数据 ({args.n_scenes} 场景 × {args.n_frames} 帧)...")
    ntr = tuple(args.n_targets_range) if args.n_targets_range else None
    tr_rps, tr_tg = build_dataset(args.n_scenes, args.n_frames, args.seed, args.snr_db,
                                  n_targets_range=ntr, stack=args.stack, n_avg=args.n_avg)
    te_rps, te_tg = build_dataset(8, args.n_frames, args.seed + 1000, args.snr_db,
                                  n_targets_range=ntr, stack=args.stack, n_avg=args.n_avg)
    print(f"训练样本: {tr_rps.shape[0]}, 测试样本: {te_rps.shape[0]}")

    model = DetectNet(count_head=True, stack=args.stack,
                      obj_head=(args.match == "hungarian")).to(device)
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
            if args.stack == 1:
                x = x.unsqueeze(1)              # [B, 1, K] 与堆叠口径一致
            clss, poss, cnt, objs = model(x)
            tgs = [tr_tg[j] for j in idx.tolist()]
            lo = None
            if args.match == "hungarian":
                lc, lp, lo = match_loss_hungarian(clss, poss, tgs, device, objs=objs)
            else:
                lc, lp = match_loss(clss, poss, tgs, device)
            loss = lc + args.pos_weight * lp
            if lo is not None:
                loss = loss + args.obj_weight * lo
            if cnt is not None:
                n_true = torch.tensor([min(len(tr_tg[j]), K_MAX) for j in idx.tolist()],
                                      device=device)
                loss = loss + args.count_weight * F.cross_entropy(cnt, n_true)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        det, cls_acc, pos_e = evaluate(model, te_rps, te_tg, device)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep+1:3d}/{args.epochs} loss={tot:.4f} | "
                  f"detect={det:.3f} cls={cls_acc:.3f} pos_err={pos_e:.3f}")
        if det > best_det:
            best_det = det
            torch.save({"model": model.state_dict(), "k": K_MAX, "count_head": True,
                        "stack": args.stack, "obj_head": (args.match == "hungarian"),
                        "n_classes": N_CLASSES, "classes": CLASS_NAMES},
                       os.path.join(args.save_dir, args.save_name))

    print(f"\n[detect] 最佳: detect={best_det:.3f}, cls={cls_acc:.3f}, pos_err={pos_e:.3f}")
    print(f"[detect] checkpoint: {args.save_dir}/{args.save_name}")


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
    parser.add_argument("--save_name", type=str, default="detect_best.pth",
                        help="checkpoint 文件名（D2 等对比实验用）")
    parser.add_argument("--count_weight", type=float, default=0.5,
                        help="计数头 loss 权重（D3）")
    parser.add_argument("--n_targets_range", nargs=2, type=int, default=None,
                        help="每场景随机目标数范围 (lo hi)（D4 变计数 benchmark）")
    parser.add_argument("--stack", type=int, default=1,
                        help="多帧堆叠输入（F1：时间上下文检测，1=单帧）")
    parser.add_argument("--n_avg", type=int, default=1,
                        help="每帧平均的独立 realizing 数（N1：测量分集，1=单 realizing）")
    parser.add_argument("--match", choices=["sorted", "hungarian"], default="sorted",
                        help="slot-目标分配：sorted（旧行为，x 排序）或 hungarian（DETR 式可微集合预测）")
    parser.add_argument("--obj_weight", type=float, default=1.0,
                        help="objectness 头 BCE 权重（S2）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
