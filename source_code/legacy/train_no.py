import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

import setup
from setup import precompute_channels
from data_no_irs import ROILDMDataset, simulate_received_signal
from models import (
    PointVAE,
    AdvancedCondEncoder,
    LatentDiT_Token_CrossAttn,
)


# =========================================================
# 工具函数
# =========================================================
def save_train_config(save_dir, config_dict):
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "train_config.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in config_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"[Save] training config saved to {txt_path}")


def save_loss_curve(train_losses, val_losses, save_path, title="LDM Loss Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Save] loss curve saved to {save_path}")


def save_checkpoint(save_dir, epoch, cond_encoder, eps_model, optimizer, z_mean, z_std, history):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "cond_encoder": cond_encoder.state_dict(),
        "eps_model": eps_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "z_mean": z_mean,
        "z_std": z_std,
        "history": history,
    }
    save_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(ckpt, save_path)
    print(f"[Save] checkpoint saved to {save_path}")


def chamfer_distance_loss(p1, p2):
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(1)
    dist = torch.sum((p1 - p2) ** 2, dim=-1)
    min_dist_1 = torch.min(dist, dim=2)[0]
    min_dist_2 = torch.min(dist, dim=1)[0]
    return torch.mean(min_dist_1) + torch.mean(min_dist_2)


def point_vae_loss(x, reconstructed_x, mu, logvar, kl_weight=1e-6):
    cd_loss = chamfer_distance_loss(x, reconstructed_x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = cd_loss + kl_weight * kl_loss
    return total, cd_loss.detach(), kl_loss.detach()


def evaluate_vae(vae, loader, device="cuda", kl_weight=1e-6):
    vae.eval()
    total_loss = 0.0
    total_cd = 0.0
    total_kl = 0.0
    with torch.no_grad():
        for batch in loader:
            pc = batch[0].to(device)
            rec_pc, mu, logvar, _ = vae(pc)
            loss, cd_loss, kl_loss = point_vae_loss(pc, rec_pc, mu, logvar, kl_weight=kl_weight)
            total_loss += loss.item()
            total_cd += cd_loss.item()
            total_kl += kl_loss.item()
    n = len(loader)
    return total_loss / n, total_cd / n, total_kl / n


def train_PointVAE(vae, train_loader, test_loader, device="cuda", epochs=100, lr=1e-3, kl_weight=1e-6, save_dir="./model"):
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    history = {"train_total": [], "test_total": [], "train_cd": [], "test_cd": [], "train_kl": [], "test_kl": []}
    best_test_loss = float('inf')
    for ep in range(1, epochs + 1):
        vae.train()
        train_total = 0.0
        train_cd = 0.0
        train_kl = 0.0
        for batch in train_loader:
            pc = batch[0].to(device)
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
        test_total, test_cd, test_kl = evaluate_vae(vae, test_loader, device=device, kl_weight=kl_weight)
        history["train_total"].append(train_total)
        history["test_total"].append(test_total)
        history["train_cd"].append(train_cd)
        history["test_cd"].append(test_cd)
        history["train_kl"].append(train_kl)
        history["test_kl"].append(test_kl)
        print(f"[Point-VAE-NoIRS] epoch {ep}/{epochs} train_total={train_total:.6f} test_total={test_total:.6f} train_cd={train_cd:.6f} test_cd={test_cd:.6f} train_kl={train_kl:.6f} test_kl={test_kl:.6f}")
        if test_total < best_test_loss:
            best_test_loss = test_total
            torch.save(vae.state_dict(), os.path.join(save_dir, "vae_best.pth"))
            print(f">>> [Best] 已更新 VAE 最佳模型参数 (Epoch {ep}, Loss: {best_test_loss:.6f})")
        if ep % 50 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch_{ep}.pth"))
            print(f">>> 已保存 VAE 模型参数 (Epoch {ep})")
    torch.save(vae.state_dict(), os.path.join(save_dir, "vae_latest.pth"))
    np.save(os.path.join(save_dir, "vae_history.npy"), history, allow_pickle=True)
    print(">>> 已保存 VAE 最终模型参数 (vae_latest.pth)")
    print(">>> 已保存 VAE 训练历史 (vae_history.npy)")
    save_loss_curve(history["train_total"], history["test_total"], title="PointVAE No-IRS Loss Curve", save_path=os.path.join(save_dir, "vae_loss_curve.png"))
    print(">>> 已保存 VAE loss 曲线图 (vae_loss_curve.png)")
    return history


def normalize_vec(v, eps=1e-6):
    return v / (v.abs().mean() + eps)


# =========================================================
# DDPM Scheduler
# =========================================================
class DDPMScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_abar = torch.sqrt(self.abar)
        self.sqrt_1m_abar = torch.sqrt(1.0 - self.abar)

    def q_sample(self, z0, t, noise):
        a = self.sqrt_abar[t].view(-1, 1, 1)
        b = self.sqrt_1m_abar[t].view(-1, 1, 1)
        return a * z0 + b * noise


# =========================================================
# 缓存 latent 的数据集
# =========================================================
class CachedLatentDataset(Dataset):
    def __init__(self, base_dataset, vae, device="cuda", print_prefix="CachedLatentDataset"):
        self.samples = []
        vae.eval()

        total_num = len(base_dataset)

        with torch.no_grad():
            for i in range(total_num):
                if i % 1000 == 0:
                    print(f"[{print_prefix}] building sample {i}/{total_num}")

                point_cloud, roi_raw_t, roi_occ_t, phase_init, X_fixed = base_dataset[i]

                point_cloud_dev = point_cloud.unsqueeze(0).to(device)
                mu, _ = vae.encode(point_cloud_dev)
                mu = mu.squeeze(0).cpu()

                self.samples.append((
                    mu,
                    roi_raw_t,
                    roi_occ_t,
                    phase_init,
                    X_fixed
                ))

        print(f"[{print_prefix}] finish caching {total_num} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================================================
# latent 统计
# =========================================================
@torch.no_grad()
def estimate_latent_stats_from_cached(loader, device="cuda", max_batches=200):
    zs = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        mu = batch[0].to(device)
        zs.append(mu.detach())

    z = torch.cat(zs, dim=0)
    z_mean = z.mean(dim=0, keepdim=True)
    z_std = z.std(dim=0, keepdim=True) + 1e-8

    print(f"[Latent Stats] z_mean shape={tuple(z_mean.shape)}, z_std shape={tuple(z_std.shape)}")
    return z_mean, z_std


# =========================================================
# 单帧特征构造
# =========================================================
def build_x_feature(X):
    x_real = normalize_vec(torch.real(X).reshape(-1))
    x_imag = normalize_vec(torch.imag(X).reshape(-1))
    x_feat = torch.cat([x_real, x_imag], dim=0).float()
    return x_feat


def build_y_phase_feature(Y, phase):
    y_real = normalize_vec(torch.real(Y).reshape(-1))
    y_imag = normalize_vec(torch.imag(Y).reshape(-1))
    y_feat = torch.cat([y_real, y_imag], dim=0)

    phase_feat = torch.cat([torch.sin(phase), torch.cos(phase)], dim=0).float()
    return torch.cat([y_feat, phase_feat], dim=0).float()


# =========================================================
# 无反射面 rollout：不更新相位，相位固定为全零占位
# 输出: [B, Tau, 128]
# =========================================================
def rollout_physical_sequence(
    roi_occ,
    phase_init,
    pilot_matrix,
    H_dict,
    device
):
    B = roi_occ.size(0)
    Tau = setup.Tau

    roi_occ = roi_occ.to(device)
    phase_t = torch.zeros(B, setup.IRS_total, device=device)
    pilot_matrix = pilot_matrix.to(device)

    x_feat_batch = []
    for b in range(B):
        x_feat_batch.append(build_x_feature(pilot_matrix[b]).unsqueeze(0))
    x_feat_batch = torch.cat(x_feat_batch, dim=0).to(device)

    seq_feats = []

    for _ in range(Tau):
        feat_list = []

        for b in range(B):
            Y_t = simulate_received_signal(
                ROI_voxel=roi_occ[b],
                phase=phase_t[b],  # 占位，不参与无反射面信号生成
                X=pilot_matrix[b],
                H_dict=H_dict,
                device=device
            )

            yp_feat = build_y_phase_feature(
                Y=Y_t,
                phase=phase_t[b]
            )

            feat_t_b = torch.cat([x_feat_batch[b], yp_feat], dim=0)
            feat_list.append(feat_t_b.unsqueeze(0))

        feat_t = torch.cat(feat_list, dim=0)
        seq_feats.append(feat_t.unsqueeze(1))

    physical_signals = torch.cat(seq_feats, dim=1)
    return physical_signals


# =========================================================
# 单轮训练
# =========================================================
def train_one_epoch_ldm(
    cond_encoder,
    eps_model,
    scheduler,
    loader,
    optimizer,
    z_mean,
    z_std,
    H_dict,
    device="cuda",
    cond_drop_prob=0.1,
):
    cond_encoder.train()
    eps_model.train()

    total_loss = 0.0

    for batch in loader:
        mu = batch[0].to(device)
        roi_occ_t = batch[2].to(device)
        phase_init = batch[3].to(device)
        X_fixed = batch[4].to(device)
        B = mu.size(0)

        z0 = (mu - z_mean) / z_std

        physical_signals = rollout_physical_sequence(
            roi_occ=roi_occ_t,
            phase_init=phase_init,
            pilot_matrix=X_fixed,
            H_dict=H_dict,
            device=device
        )

        keep_mask = (torch.rand(B, 1, 1, device=device) > cond_drop_prob).float()
        physical_signals_drop = physical_signals * keep_mask

        cond_seq = cond_encoder(physical_signals_drop)

        t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = scheduler.q_sample(z0, t, noise)

        eps_hat = eps_model(zt, t, cond_seq=cond_seq)
        loss = F.mse_loss(eps_hat, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# 单轮验证
# =========================================================
@torch.no_grad()
def eval_one_epoch_ldm(
    cond_encoder,
    eps_model,
    scheduler,
    loader,
    z_mean,
    z_std,
    H_dict,
    device="cuda",
):
    cond_encoder.eval()
    eps_model.eval()

    total_loss = 0.0

    for batch in loader:
        mu = batch[0].to(device)
        roi_occ_t = batch[2].to(device)
        phase_init = batch[3].to(device)
        X_fixed = batch[4].to(device)
        B = mu.size(0)

        z0 = (mu - z_mean) / z_std

        physical_signals = rollout_physical_sequence(
            roi_occ=roi_occ_t,
            phase_init=phase_init,
            pilot_matrix=X_fixed,
            H_dict=H_dict,
            device=device
        )

        cond_seq = cond_encoder(physical_signals)

        t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = scheduler.q_sample(z0, t, noise)

        eps_hat = eps_model(zt, t, cond_seq=cond_seq)
        loss = F.mse_loss(eps_hat, noise)

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# 单样本采样
# =========================================================
@torch.no_grad()
def sample_one_ldm(
    cond_encoder,
    eps_model,
    scheduler,
    roi_occ_t,
    phase_init,
    X_fixed,
    H_dict,
    z_mean,
    z_std,
    device="cuda",
    cfg_scale=2.0,
):
    cond_encoder.eval()
    eps_model.eval()

    if roi_occ_t.dim() == 3:
        roi_occ_t = roi_occ_t.unsqueeze(0)
    if phase_init.dim() == 1:
        phase_init = phase_init.unsqueeze(0)
    if X_fixed.dim() == 2:
        X_fixed = X_fixed.unsqueeze(0)

    roi_occ_t = roi_occ_t.to(device)
    phase_init = phase_init.to(device)
    X_fixed = X_fixed.to(device)

    physical_signals = rollout_physical_sequence(
        roi_occ=roi_occ_t,
        phase_init=phase_init,
        pilot_matrix=X_fixed,
        H_dict=H_dict,
        device=device
    )

    cond_seq = cond_encoder(physical_signals)

    zt = torch.randn(
        1,
        z_mean.shape[1],
        z_mean.shape[2],
        device=device
    )

    for tt in reversed(range(scheduler.T)):
        t = torch.full((1,), tt, device=device, dtype=torch.long)
        eps_hat = eps_model.forward_with_cfg(
            zt_tokens=zt,
            t=t,
            cond_seq=cond_seq,
            cfg_scale=cfg_scale
        )

        beta_t = scheduler.betas[tt]
        alpha_t = scheduler.alphas[tt]
        abar_t = scheduler.abar[tt]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - abar_t + 1e-8)
        mean = coef1 * (zt - coef2 * eps_hat)

        if tt > 0:
            noise = torch.randn_like(zt)
            zt = mean + torch.sqrt(beta_t) * noise
        else:
            zt = mean

    z0 = zt * z_std + z_mean
    return z0.squeeze(0)


@torch.no_grad()
def save_ldm_result(
    save_dir,
    vae,
    cond_encoder,
    eps_model,
    scheduler,
    dataset_raw,
    H_dict,
    z_mean,
    z_std,
    device="cuda",
    sample_idx=0,
    cfg_scale=2.0,
):
    os.makedirs(save_dir, exist_ok=True)

    point_cloud, roi_raw_t, roi_occ_t, phase_init, X_fixed = dataset_raw[sample_idx]

    z_pred = sample_one_ldm(
        cond_encoder=cond_encoder,
        eps_model=eps_model,
        scheduler=scheduler,
        roi_occ_t=roi_occ_t,
        phase_init=phase_init,
        X_fixed=X_fixed,
        H_dict=H_dict,
        z_mean=z_mean,
        z_std=z_std,
        device=device,
        cfg_scale=cfg_scale,
    )

    vae.eval()
    recon_pred, _ = vae.decode(z_pred.unsqueeze(0))
    pred_pc = recon_pred.squeeze(0).detach().cpu().numpy()
    gt_pc = point_cloud.detach().cpu().numpy()

    roi_center = 1.6 / 2.0
    gt_pc_denorm = gt_pc * roi_center + roi_center
    pred_pc_denorm = pred_pc * roi_center + roi_center

    np.save(os.path.join(save_dir, "sample_0001_gt.npy"), gt_pc_denorm.astype(np.float32))
    np.save(os.path.join(save_dir, "sample_0001_pred.npy"), pred_pc_denorm.astype(np.float32))
    np.save(os.path.join(save_dir, "sample_0001_roi_raw.npy"), roi_raw_t.numpy().astype(np.float32))
    np.save(os.path.join(save_dir, "sample_0001_roi_occ.npy"), roi_occ_t.numpy().astype(np.float32))

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(gt_pc_denorm[:, 0], gt_pc_denorm[:, 1], gt_pc_denorm[:, 2], s=2)
    ax1.set_title("GT Point Cloud")
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(0, 1.6)
    ax1.set_zlim(0, 1.6)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pred_pc_denorm[:, 0], pred_pc_denorm[:, 1], pred_pc_denorm[:, 2], s=2)
    ax2.set_title("LDM Prediction")
    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(0, 1.6)
    ax2.set_zlim(0, 1.6)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "sample_0001_compare.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[Save] result figure saved to {fig_path}")


# =========================================================
# 总训练函数
# =========================================================
def train_ldm(
    cond_encoder,
    eps_model,
    train_loader,
    val_loader,
    H_dict,
    device="cuda",
    epochs=200,
    lr_cond=1e-4,
    lr_eps=1e-4,
    cond_drop_prob=0.1,
    diffusion_steps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
    save_dir="./outputs_ldm_no_irs/check",
):
    os.makedirs(save_dir, exist_ok=True)

    scheduler = DDPMScheduler(
        T=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device
    )

    optimizer = torch.optim.Adam([
        {"params": cond_encoder.parameters(), "lr": lr_cond},
        {"params": eps_model.parameters(), "lr": lr_eps},
    ])

    history = {"train_loss": [], "val_loss": []}
    start_ep = 1
    z_mean = None
    z_std = None

    resume_ckpt_path = os.path.join(save_dir, "resume.pth")
    if os.path.exists(resume_ckpt_path):
        print(f"[Resume] 发现断点续训文件: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        cond_encoder.load_state_dict(ckpt["cond_encoder"])
        eps_model.load_state_dict(ckpt["eps_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt["epoch"] + 1
        z_mean = ckpt["z_mean"]
        z_std = ckpt["z_std"]
        if "history" in ckpt and ckpt["history"] is not None:
            history = ckpt["history"]
        print(f"[Resume] 从 epoch {start_ep} 继续训练 (已训练 {ckpt['epoch']} epochs)")

    if z_mean is None or z_std is None:
        z_mean, z_std = estimate_latent_stats_from_cached(train_loader, device=device)
        torch.save({"z_mean": z_mean, "z_std": z_std}, os.path.join(save_dir, "latent_stats.pth"))

    best_val_loss = float('inf')
    if len(history["val_loss"]) > 0:
        best_val_loss = min(history["val_loss"])

    for ep in range(start_ep, epochs + 1):
        train_loss = train_one_epoch_ldm(
            cond_encoder=cond_encoder,
            eps_model=eps_model,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            z_mean=z_mean,
            z_std=z_std,
            H_dict=H_dict,
            device=device,
            cond_drop_prob=cond_drop_prob,
        )

        val_loss = eval_one_epoch_ldm(
            cond_encoder=cond_encoder,
            eps_model=eps_model,
            scheduler=scheduler,
            loader=val_loader,
            z_mean=z_mean,
            z_std=z_std,
            H_dict=H_dict,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[LDM-NoIRS] epoch {ep}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_dir=save_dir,
                epoch=ep,
                cond_encoder=cond_encoder,
                eps_model=eps_model,
                optimizer=optimizer,
                z_mean=z_mean,
                z_std=z_std,
                history=history,
            )
            os.rename(
                os.path.join(save_dir, f"epoch_{ep}.pth"),
                os.path.join(save_dir, "best.pth")
            )
            print(f">>> [Best] 已更新 LDM 最佳模型 (Epoch {ep}, Val Loss: {best_val_loss:.6f})")

        if ep % 10 == 0:
            save_checkpoint(
                save_dir=save_dir,
                epoch=ep,
                cond_encoder=cond_encoder,
                eps_model=eps_model,
                optimizer=optimizer,
                z_mean=z_mean,
                z_std=z_std,
                history=history,
            )
            resume_path = os.path.join(save_dir, "resume.pth")
            os.rename(
                os.path.join(save_dir, f"epoch_{ep}.pth"),
                resume_path
            )
            print(f">>> [Checkpoint] 已保存断点续训文件 (Epoch {ep})")

    np.save(os.path.join(save_dir, "ldm_history.npy"), history, allow_pickle=True)

    save_loss_curve(
        history["train_loss"],
        history["val_loss"],
        save_path=os.path.join(save_dir, "ldm_loss_curve.png"),
        title="LDM No-IRS Loss Curve"
    )

    return history, scheduler, z_mean, z_std


# =========================================================
# main
# =========================================================
def main(
    device="cuda",
    vae_ckpt_path="./outputs_vae/best.pth",
    total_samples=704,
    train_ratio=0.9,
    batch_size=8,
    num_points=2048,
    epochs=100,
    save_dir="./outputs_ldm_no_irs",
):
    cfg = {
        "device": device,
        "vae_ckpt_path": vae_ckpt_path,
        "total_samples": total_samples,
        "train_ratio": train_ratio,
        "batch_size": batch_size,
        "num_points": num_points,
        "epochs": epochs,
        "save_dir": save_dir,
        "Tau": setup.Tau,
        "fixed_X": True,
        "X_type": "16QAM_vector",
        "signal_model": "y_t = H_bs_roi_bs x_t",
        "x_shape": "[16,1]",
        "y_shape": "[16,1]",
        "cond_dim_per_frame": 128,
        "cache_latent": True,
        "scheme": "no_irs",
        "phase_update": False,
        "phase_used_as_placeholder": True,
        "cfg_enabled": True,
    }
    save_train_config(os.path.join(save_dir, "config"), cfg)

    print("[Step] start precompute_channels")
    H_dict = precompute_channels(device=device)
    print("[Step] finish precompute_channels")

    print("[Step] start building ROILDMDataset-NoIRS")
    full_dataset = ROILDMDataset(
        n_samples=total_samples,
        H_dict=H_dict,
        num_points=num_points,
        device="cpu",
    )
    print("[Step] finish building ROILDMDataset-NoIRS")

    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size])

    print(f"[Data] total={len(full_dataset)}, train={len(train_dataset_raw)}, val={len(val_dataset_raw)}")

    vae = PointVAE(
        num_points=num_points,
        hidden_dim=512,
        token_dim=32,
        num_latent_tokens=32,
    ).to(device)

    vae_ckpt_path = "./outputs_ldm_no_irs/vae_model/vae_best.pth"
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(ckpt)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print("[Step] finish loading VAE")
    print(f"[Load] pretrained VAE loaded from {vae_ckpt_path}")

    print("[Cache] encoding train dataset latents...")
    train_dataset = CachedLatentDataset(
        train_dataset_raw,
        vae,
        device=device,
        print_prefix="TrainLatentCache-NoIRS"
    )

    print("[Cache] encoding val dataset latents...")
    val_dataset = CachedLatentDataset(
        val_dataset_raw,
        vae,
        device=device,
        print_prefix="ValLatentCache-NoIRS"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    cond_encoder = AdvancedCondEncoder(
        seq_len=setup.Tau,
        input_size=80,
        hidden_size=128,
        out_emb=256,
    ).to(device)

    eps_model = LatentDiT_Token_CrossAttn(
        token_dim=32,
        num_latent_tokens=32,
        cond_emb=256,
        hidden_size=256,
        depth=6,
        num_heads=8,
        max_cond_len=setup.Tau,
    ).to(device)

    history, scheduler, z_mean, z_std = train_ldm(
        cond_encoder=cond_encoder,
        eps_model=eps_model,
        train_loader=train_loader,
        val_loader=val_loader,
        H_dict=H_dict,
        device=device,
        epochs=epochs,
        lr_cond=1e-4,
        lr_eps=1e-4,
        cond_drop_prob=0.1,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        save_dir=os.path.join(save_dir, "check"),
    )

    save_ldm_result(
        save_dir=os.path.join(save_dir, "result"),
        vae=vae,
        cond_encoder=cond_encoder,
        eps_model=eps_model,
        scheduler=scheduler,
        dataset_raw=val_dataset_raw,
        H_dict=H_dict,
        z_mean=z_mean,
        z_std=z_std,
        device=device,
        sample_idx=0,
        cfg_scale=2.0,
    )

    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    main(
        device=device,
        vae_ckpt_path="./outputs_vae/best.pth",
        total_samples=7000,
        train_ratio=0.9,
        batch_size=16,
        num_points=2048,
        epochs=500,
        save_dir="./outputs_ldm_no_irs",
    )