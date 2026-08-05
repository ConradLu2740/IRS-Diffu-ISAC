# smoke_test.py — 最小规模全链路验证（跑通即可，不做完整训练）
# 复用 train.py 的函数，用小参数：少量样本 / 1 epoch / 小 T / 浅 DiT
import os
import torch
from torch.utils.data import DataLoader

import setup
from setup import precompute_channels
from data import ROIPairedDataset
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import DDPMScheduler, train_PointVAE, train_1D_DDPM, estimate_latent_stats, sample_conditional_1D

device = "cpu"
print(f"Using device: {device}")

# ---- 小参数 ----
train_data, test_data, batch_size, num_points = 32, 8, 8, 512
vae_epochs, ddpm_epochs, T = 1, 1, 100

# ---- 1. 预计算信道 ----
print(">>> [1/6] precompute channels")
H_dict = precompute_channels(device=device)

# ---- 2. 数据集 ----
print(f">>> [2/6] build datasets ({train_data} train / {test_data} test)")
train_dataset = ROIPairedDataset(n_samples=train_data, H_dict=H_dict, num_points=num_points, device=device)
test_dataset = ROIPairedDataset(n_samples=test_data, H_dict=H_dict, num_points=num_points, device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

pc, cond = next(iter(train_loader))
print(f"    sample shapes: point_cloud={tuple(pc.shape)}, cond={tuple(cond.shape)}")

# ---- 3. 模型实例化 ----
print(">>> [3/6] instantiate models")
vae = PointVAE(num_points=num_points, z_dim=256).to(device)
condenc = AdvancedCondEncoder(seq_len=setup.Tau, input_size=88, hidden_size=128, out_emb=256).to(device)
epsnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256, depth=2, num_heads=4).to(device)
sched = DDPMScheduler(T=T, device=device)
print(f"    VAE params={sum(p.numel() for p in vae.parameters())/1e6:.2f}M, "
      f"CondEnc params={sum(p.numel() for p in condenc.parameters())/1e6:.2f}M, "
      f"DiT params={sum(p.numel() for p in epsnet.parameters())/1e6:.2f}M")

# ---- 4. 阶段一：PointVAE ----
print(f">>> [4/6] PointVAE training (epochs={vae_epochs})")
os.makedirs("./smoke_model", exist_ok=True)
train_PointVAE(vae, train_loader, test_loader, device=device, epochs=vae_epochs, lr=1e-3, save_dir="./smoke_model")

# ---- 5. 潜在统计 + 阶段二：LDM ----
print(">>> [5/6] latent stats + LDM training")
z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device)
train_1D_DDPM(vae, condenc, epsnet, sched, train_loader, test_loader,
              z_mean, z_std, device=device, epochs=ddpm_epochs, save_dir="./smoke_model")

# ---- 6. 条件采样推理 ----
print(f">>> [6/6] conditional sampling (T={T} steps, CFG=2.0)")
pc_gt, cond = next(iter(test_loader))
pc_hat = sample_conditional_1D(vae, condenc, epsnet, sched, cond[:1].to(device),
                               z_mean, z_std, device=device, cfg_scale=2.0)
print(f"    GT shape={tuple(pc_gt[:1].shape)}, sampled shape={tuple(pc_hat.shape)}")

# 简单评估：GT 与采样点云的距离（仅验证链路，不代表质量）
with torch.no_grad():
    gt = pc_gt[:1].to(device)
    cd = torch.mean(torch.min(torch.cdist(gt, pc_hat), dim=-1)[0]) + torch.mean(torch.min(torch.cdist(pc_hat, gt), dim=-1)[0])
print(f"    approx CD (GT vs sampled) = {cd.item():.6f}")
print("\nSMOKE TEST PASSED: full pipeline (data -> VAE -> LDM -> sampling -> eval) OK")
