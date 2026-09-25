"""
fm_utils.py — Flow Matching（OT-CFM）训练与采样工具

与 legacy/train.py 中的 DDPM 工具同构，复用相同的模型（PointVAE /
AdvancedCondEncoder / LatentDiT1D_CrossAttn）与数据管道，仅替换生成范式：

  - 训练目标：条件流匹配（Optimal-Transport Conditional Flow Matching）
      x0 ~ N(0, I)（噪声端, t=0），x1 为归一化潜变量（数据端, t=1）
      x_t = (1-t)·x0 + t·x1，  回归目标 v* = x1 - x0
      loss = MSE(v_theta(x_t, t, c), v*)
  - 采样：从 N(0,I) 出发用 Euler / midpoint 法积分概率流 ODE，
      步数（NFE）可配，支持 classifier-free guidance（与 DDPM 相同的
      cond_drop 训练约定与 cfg_scale 推理约定）

时间嵌入约定：DiT 的 TimestepEmbedder 接受浮点标量 t，
这里传入 t * T_EMB_SCALE（与 DDPM 整数时间步 0..T-1 的量级兼容）。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

T_EMB_SCALE = 1000.0


class CFMScheduler:
    """OT 条件流匹配插值器：x0~N(0,I)（t=0）→ x1（t=1）的直线路径。"""

    def __init__(self, sigma_min=0.0):
        self.sigma_min = sigma_min

    def interpolate(self, x0, x1, t):
        """t: [B] ∈ [0,1]，返回 (x_t, 目标速度 v* = x1 - x0)。"""
        tb = t.view(-1, 1)
        xt = (1 - tb) * x0 + tb * x1
        if self.sigma_min > 0:
            xt = xt + self.sigma_min * torch.randn_like(xt)
        return xt, x1 - x0


def evaluate_1D_FM(vae, condenc, vnet, loader, z_mean, z_std, device="cuda"):
    """与 evaluate_1D_DDPM 同口径的验证损失（CFM 目标）。"""
    vae.eval()
    condenc.eval()
    vnet.eval()

    total_loss = 0.0
    sched = CFMScheduler()

    with torch.no_grad():
        for pc, cond in loader:
            pc = pc.to(device)
            cond = cond.to(device)
            B = cond.size(0)

            mu, _ = vae.encode(pc)
            x1 = (mu - z_mean) / z_std
            x0 = torch.randn_like(x1)

            t = torch.rand(B, device=device)
            xt, v_target = sched.interpolate(x0, x1, t)

            c = condenc(cond)
            v_hat = vnet(xt, t * T_EMB_SCALE, c)
            loss = F.mse_loss(v_hat, v_target)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_1D_FM(
    vae,
    condenc,
    vnet,
    train_loader,
    test_loader,
    z_mean,
    z_std,
    device="cuda",
    epochs=300,
    lr_cond=1e-3,
    lr_v=1e-4,
    cond_drop_prob=0.1,
    posterior_sample=False,
    save_dir="./model"
):
    """条件流匹配训练，超参数与 train_1D_DDPM 对齐（等算力公平对比）。"""
    from train import save_loss_curve

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    condenc.train()
    vnet.train()

    opt = torch.optim.Adam([
        {"params": condenc.parameters(), "lr": lr_cond},
        {"params": vnet.parameters(), "lr": lr_v}
    ])

    sched = CFMScheduler()
    history = {"train_total": [], "test_total": []}
    best_test_loss = float('inf')

    for ep in range(1, epochs + 1):
        condenc.train()
        vnet.train()

        train_total = 0.0

        for pc, cond in train_loader:
            pc = pc.to(device)
            cond = cond.to(device)
            B = cond.size(0)

            # classifier-free guidance 训练（与 DDPM 同约定）
            drop_mask = (torch.rand(B, 1, 1, device=device) > cond_drop_prob).float()
            cond_drop = cond * drop_mask

            with torch.no_grad():
                mu, logvar = vae.encode(pc)
                # ELBO 一致性：posterior_sample=True 时传输目标为后验样本 z~q
                # （聚合后验匹配），否则为后验均值 μ（旧行为）
                z_src = vae.reparam(mu, logvar) if posterior_sample else mu
                x1 = (z_src - z_mean) / z_std

            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            xt, v_target = sched.interpolate(x0, x1, t)

            c = condenc(cond_drop)
            v_hat = vnet(xt, t * T_EMB_SCALE, c)

            loss = F.mse_loss(v_hat, v_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += loss.item()

        train_total /= len(train_loader)
        test_total = evaluate_1D_FM(
            vae, condenc, vnet, test_loader, z_mean, z_std, device=device
        )

        history["train_total"].append(train_total)
        history["test_total"].append(test_total)

        print(
            f"[1D-CFM] epoch {ep}/{epochs} "
            f"train_total={train_total:.6f} test_total={test_total:.6f}"
        )

        if test_total < best_test_loss:
            best_test_loss = test_total
            torch.save(condenc.state_dict(), os.path.join(save_dir, "condenc_fm_best.pth"))
            torch.save(vnet.state_dict(), os.path.join(save_dir, "vnet_fm_best.pth"))
            print(f">>> [Best] 已更新 FM 最佳模型参数 (Epoch {ep}, Loss: {best_test_loss:.6f})")

        if ep % 50 == 0:
            torch.save(condenc.state_dict(), os.path.join(save_dir, f"condenc_fm_epoch_{ep}.pth"))
            torch.save(vnet.state_dict(), os.path.join(save_dir, f"vnet_fm_epoch_{ep}.pth"))
            print(f">>> 已保存 FM 模型参数 (Epoch {ep})")

    torch.save(condenc.state_dict(), os.path.join(save_dir, "condenc_fm_latest.pth"))
    torch.save(vnet.state_dict(), os.path.join(save_dir, "vnet_fm_latest.pth"))
    np.save(os.path.join(save_dir, "fm_history.npy"), history, allow_pickle=True)
    print(">>> 已保存 FM 最终模型参数 (condenc_fm_latest.pth, vnet_fm_latest.pth)")
    print(">>> 已保存 FM 训练历史 (fm_history.npy)")

    save_loss_curve(
        history["train_total"],
        history["test_total"],
        title="Latent Flow Matching Loss Curve",
        save_path=os.path.join(save_dir, "fm_loss_curve.png")
    )
    print(">>> 已保存 FM loss 曲线图 (fm_loss_curve.png)")

    return history


@torch.no_grad()
def sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                          device="cuda", cfg_scale=2.0, nfe=20, solver="euler"):
    """从 N(0,I) 积分概率流 ODE 生成潜变量并解码为点云。

    nfe: 函数评估次数（网络前向次数/步）；solver: euler | midpoint。
    与 sample_conditional_1D 同接口（除 sched → nfe/solver）。
    """
    vae.eval()
    condenc.eval()
    vnet.eval()

    B = cond.size(0)
    x = torch.randn((B, 256), device=device)

    c = condenc(cond.to(device))
    c_null = condenc(torch.zeros_like(cond).to(device))

    def v_cfg(xt, t_scalar):
        t = torch.full((B,), t_scalar * T_EMB_SCALE, device=device)
        v_c = vnet(xt, t, c)
        v_u = vnet(xt, t, c_null)
        return v_u + cfg_scale * (v_c - v_u)

    dt = 1.0 / nfe
    for i in range(nfe):
        t0 = i / nfe
        if solver == "midpoint":
            k1 = v_cfg(x, t0)
            k2 = v_cfg(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            x = x + dt * k2
        else:  # euler
            x = x + dt * v_cfg(x, t0)

    z0 = x * z_std + z_mean
    pc_hat, _ = vae.decode(z0)
    return pc_hat
