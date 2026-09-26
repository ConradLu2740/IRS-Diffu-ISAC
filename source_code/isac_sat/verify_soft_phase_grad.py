"""
verify_soft_phase_grad.py — 软体素相位设计：闭环可微化与 Danskin 各向异性（G-κ 门的光滑版）

背景（§7.1 G-κ 门）：硬体素 ROI 使闭环值函数 V(p) 成为阶梯函数，
Hessian 无定义、Danskin 分析失效。本脚本用**软占据** σ(p)（体素中心处的
高斯 bump）替代硬 0/1 占据，使相位设计与接收功率对目标位置 p 可微：

  1. G1 梯度正确性：autograd ∂logP/∂p vs 有限差分（相对误差 <5%）
  2. G2 Danskin 各向异性：软松弛下 κ(H_V) 与主方向对 LOS 的夹角
     （G-κ 门在光滑极限下的预测：κ≥10 且对齐 LOS）
  3. G3 端到端一步：从 MLP 位置估计出发，沿 ∂logP/∂p 做梯度上升，
     η_sense 提升 ≥3pp（优化理论方向 D2 的最小闭环）

物理约定与 verify_optimality_decomposition.py 完全一致：
相位用（软/硬）估计 ROI 设计，功率在**真实硬 ROI** 上评估。
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup_sat as ss
from data_sat import SatScenarioChannels, generate_ground_target_sample, _SIGNAL1
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing import SensingMLP


def soft_occupancy(p, centers, s=0.12):
    """体素中心 [R,3]（归一化坐标）处的软占据：σ_k = exp(-||c_k − p||²/2s²)。"""
    d2 = ((centers[:, :2] - p[:2]) ** 2).sum(dim=-1)
    return torch.exp(-d2 / (2 * s * s))


def build_linear_torch(Ht, X):
    """返回信道张量（complex64 torch）。"""
    H_BS_ROI = torch.as_tensor(Ht["H_BS_ROI"]).to(torch.complex64)
    H_ROI_UE = torch.as_tensor(Ht["H_ROI_UE"]).to(torch.complex64)
    H_BS_IRS = torch.as_tensor(Ht["H_BS_IRS"]).to(torch.complex64)
    H_ROI_IRS = torch.as_tensor(Ht["H_ROI_IRS"]).to(torch.complex64)
    H_IRS_ROI = torch.as_tensor(Ht["H_IRS_ROI"]).to(torch.complex64)
    H_IRS_UE = torch.as_tensor(Ht["H_IRS_UE"]).to(torch.complex64)
    Xc = torch.as_tensor(X).to(torch.complex64)
    return (H_BS_ROI, H_ROI_UE, H_BS_IRS, H_ROI_IRS, H_IRS_ROI, H_IRS_UE, Xc)


def soft_power_eval(Ht, X, S_soft, S_true):
    """软占据设计相位 → 硬真值评估功率（全 torch，对 S_soft 可微）。

    S_soft: [R] float（可微）；S_true: [R] float（硬 0/1）。
    """
    H_BS_ROI, H_ROI_UE, H_BS_IRS, H_ROI_IRS, H_IRS_ROI, H_IRS_UE, Xc = build_linear_torch(Ht, X)
    S_c = torch.complex(S_soft, torch.zeros_like(S_soft))
    S_true_c = torch.complex(S_true, torch.zeros_like(S_true))

    # 设计模型（软）
    d = (H_BS_ROI * S_c[None, :]).matmul(H_ROI_UE).matmul(Xc).flatten()
    C1 = (H_BS_ROI * S_c[None, :]).matmul(H_ROI_IRS)          # [BS, N]
    B1 = (H_IRS_ROI * S_c[None, :]).matmul(H_ROI_UE)          # [N, UE]
    HuX = H_IRS_UE.matmul(Xc).flatten()
    B1X = B1.matmul(Xc).flatten()
    M = C1 * HuX[None, :] + H_BS_IRS * B1X[None, :]          # [BS, N]

    c = torch.conj(d)[None, :].matmul(M).flatten()           # [N]
    phase = -torch.angle(c)
    v = torch.exp(1j * phase).to(torch.complex64)

    # 评估模型（硬真值）
    d_t = (H_BS_ROI * S_true_c[None, :]).matmul(H_ROI_UE).matmul(Xc).flatten()
    C1_t = (H_BS_ROI * S_true_c[None, :]).matmul(H_ROI_IRS)
    B1_t = (H_IRS_ROI * S_true_c[None, :]).matmul(H_ROI_UE)
    M_t = C1_t * HuX[None, :] + H_BS_IRS * B1X[None, :]
    y = d_t + M_t.matmul(v)
    return torch.sum(torch.abs(y) ** 2).real


def main(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device)
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    mid = frames[len(frames) // 2]
    los = np.array(mid["sat_pos"][:2]) - np.array(mid["target_pos"][:2])
    los = los / (np.linalg.norm(los) + 1e-12)

    # 体素中心（归一化坐标 [R,3]）
    res = 16
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, res)] * 3), axis=-1).reshape(-1, 3)
    centers = torch.tensor(grid, dtype=torch.float32)

    from data_sat import compute_range_profile

    g_ok, kappas, cosangs, eta_before, eta_after, n_used = 0, [], [], [], [], 0
    for sd in range(args.n_seeds):
        torch.manual_seed(sd); random.seed(sd); np.random.seed(sd)
        roi_true, _, _ = generate_ground_target_sample()
        pc = np.argwhere(roi_true > 0.5).astype(np.float32)
        p_true = (pc.mean(axis=0) / res * 2.0 - 1.0)[:2]
        S_true = torch.tensor(roi_true.astype(np.float32)).reshape(-1)

        rp = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                                   channels.wavelength_m, snr_db=args.snr_db, seed=0,
                                   align=True, sat_ecef=mid["sat_pos"])
        with torch.no_grad():
            _, pred_pos = model(torch.from_numpy(rp).float().unsqueeze(0))
        p0 = torch.tensor(pred_pos[0].cpu().numpy()[:2], dtype=torch.float32).requires_grad_(True)

        Ht = channels.channels_per_frame[len(channels.channels_per_frame) // 2]

        def power_at(p):
            S_soft = soft_occupancy(p, centers, s=args.sigma)
            return soft_power_eval(Ht, X, S_soft, S_true)

        # ---- G1 autograd vs 有限差分 ----
        P = power_at(p0)
        g_auto = torch.autograd.grad(torch.log(P), p0)[0].numpy()
        eps = 1e-3
        g_fd = np.array([
            (float(torch.log(power_at(p0 + torch.tensor([eps, 0])))) -
             float(torch.log(power_at(p0 - torch.tensor([eps, 0]))))) / (2 * eps),
            (float(torch.log(power_at(p0 + torch.tensor([0, eps])))) -
             float(torch.log(power_at(p0 - torch.tensor([0, eps]))))) / (2 * eps),
        ])
        rel = np.linalg.norm(g_auto - g_fd) / (np.linalg.norm(g_fd) + 1e-12)
        g_ok += int(rel < 0.05)

        # ---- G2 Hessian 各向异性（软松弛，autograd）----
        p = p0.detach().clone().requires_grad_(True)
        S_soft = soft_occupancy(p, centers, s=args.sigma)
        P2 = soft_power_eval(Ht, X, S_soft, S_true)
        logP = torch.log(P2)
        grad = torch.autograd.grad(logP, p, create_graph=True)[0]
        H = torch.zeros(2, 2)
        for i in range(2):
            gi = torch.autograd.grad(grad[i], p, retain_graph=True)[0]
            H[i] = gi.detach()
        H = 0.5 * (H + H.T)
        ev, evec = torch.linalg.eigh(H)
        if torch.abs(ev[0]) > 1e-12:
            kappas.append(float(torch.abs(ev[1] / ev[0])))
            cosangs.append(float(abs(torch.dot(evec[:, 1], torch.tensor(los, dtype=torch.float32)))))

        # ---- G3 端到端一步：沿梯度上升微调位置 ----
        p_opt = p0.detach().clone().requires_grad_(True)
        for _ in range(args.n_steps):
            S_soft = soft_occupancy(p_opt, centers, s=args.sigma)
            P3 = soft_power_eval(Ht, X, S_soft, S_true)
            g = torch.autograd.grad(torch.log(P3), p_opt)[0]
            p_opt = (p_opt.detach() + args.lr * g).requires_grad_(True)
        with torch.no_grad():
            p_ref = p_true
            eta_b = float(power_at(p0.detach()) / power_at(torch.tensor(p_ref, dtype=torch.float32)))
            eta_a = float(power_at(p_opt.detach()) / power_at(torch.tensor(p_ref, dtype=torch.float32)))
        eta_before.append(eta_b); eta_after.append(eta_a)
        n_used += 1

    verdicts = {
        "G1_grad_correct": bool(g_ok == n_used),
        "G2_kappa_ge_10": bool(np.median(kappas) >= 10 if kappas else False),
        "G2_cos_los_ge_085": bool(np.median(cosangs) >= 0.85 if cosangs else False),
        "G3_eta_gain_ge_3pp": bool((np.mean(eta_after) - np.mean(eta_before)) * 100 >= 3),
    }
    print(f"\n{'=' * 70}")
    print(f"软体素相位设计（{n_used} 种子，σ={args.sigma}）")
    print(f"  G1 梯度正确性: {g_ok}/{n_used} 种子 autograd≈FD")
    if kappas:
        print(f"  G2 κ(H_V) 中位 = {np.median(kappas):.1f}, "
              f"|cos⟨主方向,LOS⟩| 中位 = {np.median(cosangs):.3f}")
    print(f"  G3 η_sense: {np.mean(eta_before):.3f} → {np.mean(eta_after):.3f} "
          f"(+{(np.mean(eta_after) - np.mean(eta_before)) * 100:.1f}pp)")
    print(f"  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"n_seeds": n_used, "sigma": args.sigma,
           "g1_correct": f"{g_ok}/{n_used}",
           "kappa_median": float(np.median(kappas)) if kappas else None,
           "cos_los_median": float(np.median(cosangs)) if cosangs else None,
           "eta_before": float(np.mean(eta_before)), "eta_after": float(np.mean(eta_after)),
           "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "soft_phase_grad.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="软体素相位设计与 Danskin 各向异性")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=0.12, help="软占据高斯宽度（归一化坐标）")
    parser.add_argument("--n_steps", type=int, default=20, help="端到端梯度上升步数")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
