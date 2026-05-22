import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import setup
from setup import precompute_channels
from data import ROIPairedDataset
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn


class DDPMScheduler:
    def __init__(self, T=400, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_abar = torch.sqrt(self.abar)
        self.sqrt_1m_abar = torch.sqrt(1 - self.abar)

    def q_sample(self, z0, t, noise):
        a = self.sqrt_abar[t].view(-1, 1)
        b = self.sqrt_1m_abar[t].view(-1, 1)
        return a * z0 + b * noise


def chamfer_distance_loss(p1, p2):
    """
    p1, p2: [B, N, 3]
    """
    p1 = p1.unsqueeze(2)  # [B, N, 1, 3]
    p2 = p2.unsqueeze(1)  # [B, 1, N, 3]
    dist = torch.sum((p1 - p2) ** 2, dim=-1)  # [B, N, N]
    min_dist_1 = torch.min(dist, dim=2)[0]    # [B, N]
    min_dist_2 = torch.min(dist, dim=1)[0]    # [B, N]
    return torch.mean(min_dist_1) + torch.mean(min_dist_2)


def point_vae_loss(x, reconstructed_x, mu, logvar, kl_weight=1e-6):
    cd_loss = chamfer_distance_loss(x, reconstructed_x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = cd_loss + kl_weight * kl_loss
    return total, cd_loss.detach(), kl_loss.detach()


def save_loss_curve(train_losses, test_losses, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_vae(vae, loader, device="cuda", kl_weight=1e-6):
    vae.eval()
    total_loss = 0.0
    total_cd = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for pc, _ in loader:
            pc = pc.to(device)
            rec_pc, mu, logvar, _ = vae(pc)
            loss, cd_loss, kl_loss = point_vae_loss(pc, rec_pc, mu, logvar, kl_weight=kl_weight)

            total_loss += loss.item()
            total_cd += cd_loss.item()
            total_kl += kl_loss.item()

    n = len(loader)
    return total_loss / n, total_cd / n, total_kl / n


def train_PointVAE(
    vae,
    train_loader,
    test_loader,
    device="cuda",
    epochs=100,
    lr=1e-3,
    kl_weight=1e-6,
    save_dir="./model"
):
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    history = {
        "train_total": [],
        "test_total": [],
        "train_cd": [],
        "test_cd": [],
        "train_kl": [],
        "test_kl": []
    }

    # 初始化最佳 loss 为无穷大
    best_test_loss = float('inf')

    for ep in range(1, epochs + 1):
        vae.train()
        train_total = 0.0
        train_cd = 0.0
        train_kl = 0.0

        for pc, _ in train_loader:
            pc = pc.to(device)

            rec_pc, mu, logvar, _ = vae(pc)
            loss, cd_loss, kl_loss = point_vae_loss(pc, rec_pc, mu, logvar, kl_weight=kl_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += loss.item()
            train_cd += cd_loss.item()
            train_kl += kl_loss.item()

        train_total /= len(train_loader)
        train_cd /= len(train_loader)
        train_kl /= len(train_loader)

        test_total, test_cd, test_kl = evaluate_vae(
            vae, test_loader, device=device, kl_weight=kl_weight
        )

        history["train_total"].append(train_total)
        history["test_total"].append(test_total)
        history["train_cd"].append(train_cd)
        history["test_cd"].append(test_cd)
        history["train_kl"].append(train_kl)
        history["test_kl"].append(test_kl)

        print(
            f"[Point-VAE] epoch {ep}/{epochs} "
            f"train_total={train_total:.6f} test_total={test_total:.6f} "
            f"train_cd={train_cd:.6f} test_cd={test_cd:.6f} "
            f"train_kl={train_kl:.6f} test_kl={test_kl:.6f}"
        )

        # 1. 保存最好的一次
        if test_total < best_test_loss:
            best_test_loss = test_total
            torch.save(vae.state_dict(), os.path.join(save_dir, "vae_best.pth"))
            print(f">>> [Best] 已更新 VAE 最佳模型参数 (Epoch {ep}, Loss: {best_test_loss:.6f})")

        # 2. 每隔50次保存一次
        if ep % 50 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch_{ep}.pth"))
            print(f">>> 已保存 VAE 模型参数 (Epoch {ep})")

    # 3. 保存最后一次
    torch.save(vae.state_dict(), os.path.join(save_dir, "vae_latest.pth"))
    np.save(os.path.join(save_dir, "vae_history.npy"), history, allow_pickle=True)
    print(">>> 已保存 VAE 最终模型参数 (vae_latest.pth)")
    print(">>> 已保存 VAE 训练历史 (vae_history.npy)")

    save_loss_curve(
        history["train_total"],
        history["test_total"],
        title="PointVAE Loss Curve",
        save_path=os.path.join(save_dir, "vae_loss_curve.png")
    )
    print(">>> 已保存 VAE loss 曲线图 (vae_loss_curve.png)")

    return history


@torch.no_grad()
def estimate_latent_stats(vae, loader, device="cuda", max_batches=200):
    vae.eval()
    zs = []

    for i, (pc, _) in enumerate(loader):
        if i >= max_batches:
            break
        mu, _ = vae.encode(pc.to(device))
        zs.append(mu.detach())

    z = torch.cat(zs, dim=0)
    z_mean = z.mean()
    z_std = z.std() + 1e-8
    print(f"[Latent Stats] 1D Global Mean: {z_mean.item():.4f}, Std: {z_std.item():.4f}")
    return z_mean, z_std


def evaluate_1D_DDPM(vae, condenc, epsnet, sched, loader, z_mean, z_std, device="cuda"):
    vae.eval()
    condenc.eval()
    epsnet.eval()

    total_loss = 0.0

    with torch.no_grad():
        for pc, cond in loader:
            pc = pc.to(device)
            cond = cond.to(device)
            B = cond.size(0)

            mu, _ = vae.encode(pc)
            z0 = (mu - z_mean) / z_std

            t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = sched.q_sample(z0, t, noise)

            c = condenc(cond)
            eps_hat = epsnet(zt, t, c)
            loss = F.mse_loss(eps_hat, noise)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_1D_DDPM(
    vae,
    condenc,
    epsnet,
    sched,
    train_loader,
    test_loader,
    z_mean,
    z_std,
    device="cuda",
    epochs=300,
    lr_cond=1e-3,
    lr_eps=1e-4,
    cond_drop_prob=0.1,
    save_dir="./model"
):
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    condenc.train()
    epsnet.train()

    opt = torch.optim.Adam([
        {"params": condenc.parameters(), "lr": lr_cond},
        {"params": epsnet.parameters(), "lr": lr_eps}
    ])

    history = {
        "train_total": [],
        "test_total": []
    }

    # 初始化最佳 loss 为无穷大
    best_test_loss = float('inf')

    for ep in range(1, epochs + 1):
        condenc.train()
        epsnet.train()

        train_total = 0.0

        for pc, cond in train_loader:
            pc = pc.to(device)
            cond = cond.to(device)
            B = cond.size(0)

            # classifier-free guidance 训练
            drop_mask = (torch.rand(B, 1, 1, device=device) > cond_drop_prob).float()
            cond_drop = cond * drop_mask

            with torch.no_grad():
                mu, _ = vae.encode(pc)
                z0 = (mu - z_mean) / z_std

            t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = sched.q_sample(z0, t, noise)

            c = condenc(cond_drop)
            eps_hat = epsnet(zt, t, c)

            loss = F.mse_loss(eps_hat, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += loss.item()

        train_total /= len(train_loader)
        test_total = evaluate_1D_DDPM(
            vae, condenc, epsnet, sched, test_loader, z_mean, z_std, device=device
        )

        history["train_total"].append(train_total)
        history["test_total"].append(test_total)

        print(
            f"[1D-DDPM] epoch {ep}/{epochs} "
            f"train_total={train_total:.6f} test_total={test_total:.6f}"
        )

        # 1. 保存最好的一次
        if test_total < best_test_loss:
            best_test_loss = test_total
            torch.save(condenc.state_dict(), os.path.join(save_dir, "condenc_best.pth"))
            torch.save(epsnet.state_dict(), os.path.join(save_dir, "epsnet_best.pth"))
            print(f">>> [Best] 已更新 DDPM 最佳模型参数 (Epoch {ep}, Loss: {best_test_loss:.6f})")

        # 2. 每隔50次保存一次
        if ep % 50 == 0:
            torch.save(condenc.state_dict(), os.path.join(save_dir, f"condenc_epoch_{ep}.pth"))
            torch.save(epsnet.state_dict(), os.path.join(save_dir, f"epsnet_epoch_{ep}.pth"))
            print(f">>> 已保存 DDPM 模型参数 (Epoch {ep})")

    # 3. 保存最后一次
    torch.save(condenc.state_dict(), os.path.join(save_dir, "condenc_latest.pth"))
    torch.save(epsnet.state_dict(), os.path.join(save_dir, "epsnet_latest.pth"))
    np.save(os.path.join(save_dir, "ldm_history.npy"), history, allow_pickle=True)
    print(">>> 已保存 DDPM 最终模型参数 (condenc_latest.pth, epsnet_latest.pth)")
    print(">>> 已保存 DDPM 训练历史 (ldm_history.npy)")

    save_loss_curve(
        history["train_total"],
        history["test_total"],
        title="Latent Diffusion Loss Curve",
        save_path=os.path.join(save_dir, "ldm_loss_curve.png")
    )
    print(">>> 已保存 LDM loss 曲线图 (ldm_loss_curve.png)")

    return history


@torch.no_grad()
def sample_conditional_1D(vae, condenc, epsnet, sched, cond, z_mean, z_std, device="cuda", cfg_scale=2.0):
    vae.eval()
    condenc.eval()
    epsnet.eval()

    B = cond.size(0)
    zt = torch.randn((B, 256), device=device)

    c = condenc(cond.to(device))
    c_null = condenc(torch.zeros_like(cond).to(device))

    for ti in reversed(range(sched.T)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)

        eps_cond = epsnet(zt, t, c)
        eps_uncond = epsnet(zt, t, c_null)
        eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        beta = sched.betas[ti]
        alpha = sched.alphas[ti]
        abar = sched.abar[ti]

        mean = (1.0 / torch.sqrt(alpha)) * (
            zt - (beta / torch.sqrt(1 - abar + 1e-8)) * eps_hat
        )

        if ti > 0:
            zt = mean + torch.sqrt(beta) * torch.randn_like(zt)
        else:
            zt = mean

    z0 = zt * z_std + z_mean
    pc_hat, _ = vae.decode(z0)
    return pc_hat


def main(
    device="cuda",
    train_data=6400,
    test_data=1600,
    batch_size=32,
    num_points=2048
):
    # =========================================================
    # 1. 预计算物理信道
    # =========================================================
    H_dict = precompute_channels(device=device)

    # =========================================================
    # 2. 构造训练集和测试集
    # =========================================================
    train_dataset = ROIPairedDataset(
        n_samples=train_data,
        H_dict=H_dict,
        num_points=num_points,
        device=device
    )
    test_dataset = ROIPairedDataset(
        n_samples=test_data,
        H_dict=H_dict,
        num_points=num_points,
        device=device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # =========================================================
    # 3. 创建保存目录
    # =========================================================
    model_dir = "./model"
    out_dir = "./outputs_npy"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================
    # 4. 实例化模型
    # =========================================================
    vae = PointVAE(num_points=num_points, z_dim=256).to(device)
    condenc = AdvancedCondEncoder(
        seq_len=setup.Tau,
        input_size=88,
        hidden_size=128,
        out_emb=256
    ).to(device)
    epsnet = LatentDiT1D_CrossAttn(
        z_dim=256,
        cond_emb=256,
        hidden_size=256,
        depth=4,
        num_heads=8
    ).to(device)
    sched = DDPMScheduler(T=1000, device=device)

    # =========================================================
    # 5. 训练 VAE
    # =========================================================
    print("Starting Point-VAE training...")
    vae_history = train_PointVAE(
        vae=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=500,
        lr=1e-3,
        kl_weight=1e-6,
        save_dir=model_dir
    )

    # =========================================================
    # 6. VAE 核对
    # =========================================================
    pc_gt, _ = next(iter(test_loader))
    pc_gt = pc_gt.to(device)[:1]

    vae.eval()
    with torch.no_grad():
        mu_gt, _ = vae.encode(pc_gt)
        pc_vae_reconstruct, _ = vae.decode(mu_gt)

    pc_gt_check_physical = pc_gt[0].detach().cpu().numpy().astype(np.float32) * 0.8 + 0.8
    pc_vae_check_physical = pc_vae_reconstruct[0].detach().cpu().numpy().astype(np.float32) * 0.8 + 0.8

    np.save(os.path.join(out_dir, "PC_GT_Check.npy"), pc_gt_check_physical)
    np.save(os.path.join(out_dir, "PC_VAE_Reconstruct_Check.npy"), pc_vae_check_physical)
    print(">>> 已保存 GT 与 VAE 重建核对点云。")

    # =========================================================
    # 7. 统计潜在空间分布
    # =========================================================
    z_mean, z_std = estimate_latent_stats(vae, train_loader, device=device)
    torch.save({"z_mean": z_mean, "z_std": z_std}, os.path.join(model_dir, "latent_stats.pth"))
    print(">>> 已保存隐空间统计量 (latent_stats.pth)")

    # =========================================================
    # 8. 训练 LDM / DDPM
    # =========================================================
    print("Starting 1D Latent Diffusion training...")
    ldm_history = train_1D_DDPM(
        vae=vae,
        condenc=condenc,
        epsnet=epsnet,
        sched=sched,
        train_loader=train_loader,
        test_loader=test_loader,
        z_mean=z_mean,
        z_std=z_std,
        device=device,
        epochs=500,
        lr_cond=1e-3,
        lr_eps=1e-4,
        cond_drop_prob=0.1,
        save_dir=model_dir
    )

    # =========================================================
    # 9. 条件推理测试
    # =========================================================
    pc_gt, cond = next(iter(test_loader))
    pc_gt = pc_gt.to(device)[:1]
    cond = cond.to(device)[:1]

    pc_hat = sample_conditional_1D(
        vae=vae,
        condenc=condenc,
        epsnet=epsnet,
        sched=sched,
        cond=cond,
        z_mean=z_mean,
        z_std=z_std,
        device=device,
        cfg_scale=2.0
    )

    print("[Infer] PC_gt shape:", pc_gt.shape)
    print("[Infer] PC_hat shape:", pc_hat.shape)

    pc_gt_physical = pc_gt[0].detach().cpu().numpy().astype(np.float32) * 0.8 + 0.8
    pc_hat_physical = pc_hat[0].detach().cpu().numpy().astype(np.float32) * 0.8 + 0.8

    np.save(os.path.join(out_dir, "PC_gt_0001.npy"), pc_gt_physical)
    np.save(os.path.join(out_dir, "PC_hat_0001.npy"), pc_hat_physical)
    print(f">>> 条件推理点云已保存到 {out_dir}")

    return vae_history, ldm_history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    main(
        device=device,
        train_data=6400,
        test_data=1600,
        batch_size=64,
        num_points=2048
    )