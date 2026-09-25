"""
diagnose_condenc_collapse.py — 条件编码器坍塌诊断（G15 第一步）

在受控复训中跟踪：
  1. condenc 输出的 per-sample std（跨条件变化度）随 epoch 的演化
  2. 打乱条件后 condenc 输出的相对变化（敏感度）
  3. 流向 condenc 参数的梯度范数（是否还有学习信号）
  4. vnet 输出对条件变化的敏感度（末端影响）
  5. FM 损失本身（坍塌是否损害损失——判断无条件是否已近优）

协议与 compare_gen.py 的 FM 训练完全一致（同数据/超参/lr），只加监控。
"""

import os
import sys
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import train_PointVAE, estimate_latent_stats
from fm_utils import CFMScheduler, T_EMB_SCALE


@torch.no_grad()
def probe(condenc, vnet, cond, device):
    """condenc 输出 std / 打乱敏感度 / vnet 末端敏感度。"""
    condenc.eval(); vnet.eval()
    c = condenc(cond)
    c_shuf = condenc(cond[torch.randperm(cond.size(0))])
    x = torch.randn((cond.size(0), 256), device=device)
    t = torch.zeros(cond.size(0), device=device)
    v = vnet(x, t, c)
    v_shuf = vnet(x, t, c_shuf)
    condenc.train(); vnet.train()
    return {
        "out_std": float(c.std()),
        "shuffled_rel": float((c - c_shuf).norm() / c.norm()),
        "vnet_cond_sens": float((v - v_shuf).norm() / v.norm()),
    }


def main(args):
    device = args.device
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    train_loader = DataLoader(SatROIDataset(args.train_data, channels,
                                            num_points=args.num_points, device=device,
                                            tau=args.tau, phase_mode=args.phase_mode),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(SatROIDataset(args.test_data, channels,
                                           num_points=args.num_points, device=device,
                                           tau=args.tau, phase_mode=args.phase_mode),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    cond_dim = channels.frame_cond_dim()

    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    train_PointVAE(vae, train_loader, test_loader, device=device,
                   epochs=args.vae_epochs, lr=1e-3, kl_weight=args.kl_weight,
                   kl_warmup_epochs=args.kl_warmup, save_dir=args.save_dir)
    z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device)

    condenc = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                  hidden_size=128, out_emb=256).to(device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    opt = torch.optim.Adam([
        {"params": condenc.parameters(), "lr": args.lr_cond},
        {"params": vnet.parameters(), "lr": args.lr_v}
    ])
    sched = CFMScheduler()

    # 固定监控批次
    mon_pc, mon_cond = next(iter(test_loader))
    mon_cond = mon_cond[:16].to(device)

    print(f"{'ep':>4} {'loss':>8} {'out_std':>9} {'shuf_rel':>10} {'vnet_sens':>10} "
          f"{'|g|cond':>9} {'|g|vnet':>9}")
    for ep in range(1, args.epochs + 1):
        condenc.train(); vnet.train()
        tot = 0.0
        g_cond = g_vnet = 0.0
        for pc, cond in train_loader:
            cond = cond.to(device); B = cond.size(0)
            drop = (torch.rand(B, 1, 1, device=device) > args.cond_drop_prob).float()
            cond_d = cond * drop
            with torch.no_grad():
                mu, _ = vae.encode(pc.to(device))
                x1 = (mu - z_mean) / z_std
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            xt, v_target = sched.interpolate(x0, x1, t)
            c = condenc(cond_d)
            v_hat = vnet(xt, t * T_EMB_SCALE, c)
            loss = F.mse_loss(v_hat, v_target)
            opt.zero_grad(); loss.backward()
            g_cond += sum(p.grad.norm().item() for p in condenc.parameters()
                          if p.grad is not None)
            g_vnet += sum(p.grad.norm().item() for p in vnet.parameters()
                          if p.grad is not None)
            opt.step()
            tot += loss.item()
        if ep % args.mon_every == 0 or ep == 1:
            pr = probe(condenc, vnet, mon_cond, device)
            n = len(train_loader)
            print(f"{ep:>4} {tot/n:>8.4f} {pr['out_std']:>9.4f} {pr['shuffled_rel']:>10.2e} "
                  f"{pr['vnet_cond_sens']:>10.2e} {g_cond/n:>9.2e} {g_vnet/n:>9.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="condenc 坍塌诊断")
    parser.add_argument("--save_dir", type=str, default="./sat_model_g15_diag")
    parser.add_argument("--train_data", type=int, default=256)
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--vae_epochs", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--kl_warmup", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--lr_cond", type=float, default=1e-3)
    parser.add_argument("--lr_v", type=float, default=1e-4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--mon_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
