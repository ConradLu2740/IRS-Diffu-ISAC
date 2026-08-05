"""
train_sensing_multi.py — 多目标感知模型训练（分类 + 定位，最多 K 个目标）

工程场景：ROI 内 1-2 个目标，感知模型输出 K=2 组 (类别, 位置)。
匹配：预测组按位置排序与真实目标（按 x 排序）匹配，计算联合 loss。
评估：检测率 / 分类准确率 / 定位误差。

用法：
  python train_sensing_multi.py [--wideband]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import setup_sat as ss
from data_sat import (SatROIDataset, SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      WIDEBAND_K)

N_CLASSES = len(GROUND_TARGET_TEMPLATES)
CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
K_MAX = 2


class SensingMLPMulti(nn.Module):
    """多目标感知：共享编码 + K 组 (分类头, 定位头)。"""

    def __init__(self, in_dim, k=K_MAX, hidden=256):
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
            [nn.Sequential(nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 2))
             for _ in range(k)])

    def forward(self, x):
        feat = self.shared(x.flatten(1))
        clss = [h(feat) for h in self.cls_heads]
        poss = [h(feat) for h in self.pos_heads]
        return clss, poss


def build_fixed(dataset, n, wideband):
    samples = [dataset[i] for i in range(n)]
    pcs = torch.stack([s[0] for s in samples])
    if wideband:
        feats = torch.stack([s[2] for s in samples])
        tg = [s[3] for s in samples]
    else:
        feats = torch.stack([s[1] for s in samples])
        tg = [s[2] for s in samples]
    return pcs, feats, tg


def match_loss(clss, poss, targets, device):
    """预测 K 组与真实目标匹配：真实目标按 x 排序，预测组也按 x 排序。"""
    B = poss[0].shape[0]
    loss_cls = loss_pos = 0.0
    for b in range(B):
        tg = targets[b]  # [(cid, (cx,cy)), ...]
        tg = sorted(tg, key=lambda t: t[1][0])
        n_t = len(tg)
        # 预测组按 x 排序
        pred_pos = torch.stack([p[b] for p in poss])          # [K, 2]
        pred_cls = torch.stack([c[b] for c in clss])          # [K, 6]
        order = torch.argsort(pred_pos[:, 0])
        pred_pos = pred_pos[order]; pred_cls = pred_cls[order]
        for k in range(K_MAX):
            if k < n_t:
                cid, (cx, cy) = tg[k]
                tgt_cls = torch.tensor(cid, device=device, dtype=torch.long)
                tgt_pos = torch.tensor([cx, cy], device=device)
                loss_cls = loss_cls + F.cross_entropy(pred_cls[k].unsqueeze(0), tgt_cls.unsqueeze(0))
                loss_pos = loss_pos + F.mse_loss(pred_pos[k], tgt_pos)
            else:
                # 空目标：推远位置 + 均匀类别（弱正则）
                loss_pos = loss_pos + 4.0 * F.mse_loss(pred_pos[k], torch.tensor([2.0, 2.0], device=device))
    return loss_cls / B, loss_pos / B


@torch.no_grad()
def evaluate(model, feats, targets, device, wideband):
    model.eval()
    total_t = detected = cls_ok = 0
    pos_err = 0.0
    B = feats.shape[0]
    clss, poss = model(feats.to(device))
    for b in range(B):
        tg = sorted(targets[b], key=lambda t: t[1][0])
        pred_pos = torch.stack([p[b] for p in poss]).cpu().numpy()
        pred_cls = torch.stack([c[b] for c in clss]).argmax(1).cpu().numpy()
        order = np.argsort(pred_pos[:, 0])
        pred_pos = pred_pos[order]; pred_cls = pred_cls[order]
        for k in range(min(K_MAX, len(tg))):
            cid, (cx, cy) = tg[k]
            total_t += 1
            # 检测：预测位置在范围内（~2σ）
            if np.linalg.norm(pred_pos[k] - np.array([cx, cy])) < 0.5:
                detected += 1
                cls_ok += (pred_cls[k] == cid)
                pos_err += np.linalg.norm(pred_pos[k] - np.array([cx, cy]))
    return detected / max(total_t, 1), cls_ok / max(total_t, 1), pos_err / max(total_t, 1)


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}, irs={args.irs_mode}, wideband={args.wideband}, K={K_MAX}")

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    feat_dim = WIDEBAND_K if args.wideband else args.tau * channels.frame_cond_dim()

    tr_ds = SatROIDataset(args.train_data, channels, num_points=args.num_points,
                          device=device, tau=args.tau, phase_mode=args.phase_mode,
                          with_label=True, target_source="ground", wideband=args.wideband,
                          rp_align=not args.rp_align, multi=True)
    te_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                          device=device, tau=args.tau, phase_mode=args.phase_mode,
                          with_label=True, target_source="ground", wideband=args.wideband,
                          rp_align=not args.rp_align, multi=True)
    tr_feats, tr_tg = build_fixed(tr_ds, args.train_data, args.wideband)[1:]
    te_feats, te_tg = build_fixed(te_ds, args.test_data, args.wideband)[1:]

    model = SensingMLPMulti(in_dim=feat_dim, k=K_MAX).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    best_det = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        perm = torch.randperm(args.train_data)
        for i in range(0, args.train_data, args.batch_size):
            idx = perm[i:i + args.batch_size]
            clss, poss = model(tr_feats[idx].to(device))
            lc, lp = match_loss(clss, poss, [tr_tg[j] for j in idx.tolist()], device)
            loss = lc + args.pos_weight * lp
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        det, cls_acc, pos_e = evaluate(model, te_feats, te_tg, device, args.wideband)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep+1:3d}/{args.epochs} loss={tot:.4f} | "
                  f"detect={det:.3f} cls={cls_acc:.3f} pos_err={pos_e:.3f}")
        if det > best_det:
            best_det = det
            torch.save({"model": model.state_dict(), "feat_dim": feat_dim,
                        "wideband": args.wideband, "k": K_MAX,
                        "irs_mode": args.irs_mode},
                       os.path.join(args.save_dir, "sensing_multi_best.pth"))

    print(f"\n[multi] 最佳: detect={best_det:.3f}, cls={cls_acc:.3f}, pos_err={pos_e:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多目标感知训练")
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
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--wideband", action="store_true")
    parser.add_argument("--rp_align", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./isac_demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
