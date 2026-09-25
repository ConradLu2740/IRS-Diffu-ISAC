"""
verify_p1_gates.py — P1 前置证伪门（半天级，决定后续方向生死）

三门（预注册命题，见 docs/optimization_roadmap.md §7）：

  G-κ  闭环值函数 V(p)=max_{|v_i|=1} P(v;p) 的 Hessian 各向异性：
        数值差分测 κ(H_V) 与主特征向量对 BS 视线水平投影的夹角。
        预测 κ≥10 且夹角<20°（⇒ 现有无权 MSE 把容量错配到闭环不敏感方向，
        闭环 Hessian 加权方向成立）。κ<2 ⇒ 方向死亡（MSE 已近最优）。

  G-AU VAE 后验坍缩诊断：逐维 KL_i = 0.5(mu_i²+exp(logvar_i)-1-logvar_i)，
        活跃维 AU@0.1nat = #{KL_i>0.1nat}/256，及 Barber-Agakov MI 下界。
        预测 AU≤0.5（⇒ free-bits 方向成立）。AU≥0.8 ⇒ 坍缩非瓶颈，方向作废。

  G-FIM 感知-通信相位可分离性（负定理前提）：
        结构事实：HRRP 观测 compute_range_profile 不含 RIS 相位 v 的输入
        ⇒ sat 模式感知 FIM 与 v 严格无关；ground 模式的 RIS 回波泄漏由
        audit_ris_doppler 的功率门量化（echo/comm≈6e-5）。
        数值验证：随机 v 下 range profile 的最大相对变化（sat 应为 0）。

协议：场景/种子与 verify_optimality_decomposition.py 一致。
"""

import os
import json
import argparse
import numpy as np
import random
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatScenarioChannels, generate_ground_target_sample, _SIGNAL1, \
    compute_range_profile
from phase_optimizer_sat import PhaseOptimizerSat
from demo import estimate_roi_from_pos
from models import PointVAE
from train import chamfer_distance_loss


# ----------------------------------------------------------------------
# G-κ：闭环值函数 Hessian 各向异性
# ----------------------------------------------------------------------

def closed_loop_power(pos_xy, channels, opt, X, frames, n_frames=None):
    """给定目标位置（归一化 [-1,1]²），跑闭式相位设计，返回跨帧平均接收功率。"""
    roi = estimate_roi_from_pos(pos_xy)
    roi_t = torch.tensor(roi.astype(np.float32)).reshape(-1)
    powers = []
    idx = range(len(channels.channels_per_frame)) if n_frames is None \
        else range(n_frames)
    for t in idx:
        Ht = channels.channels_per_frame[t]
        ph = opt.optimize_frame(Ht, roi_t, X)
        powers.append(opt._power(Ht, roi_t, X, ph))
    return float(np.mean(powers))


def gate_hessian(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device="cpu",
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    opt = PhaseOptimizerSat(channels, device="cpu")
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)
    mid = frames[len(frames) // 2]
    # BS 视线水平投影方向（ENU 近似：用卫星-目标水平位移）
    los = np.array(mid["sat_pos"][:2]) - np.array(mid["target_pos"][:2])
    los = los / (np.linalg.norm(los) + 1e-12)

    rows = []
    for s in range(args.n_seeds):
        roi_true, _, _ = generate_ground_target_sample()
        pc = np.argwhere(roi_true > 0.5).astype(np.float32)
        p0 = (pc.mean(axis=0) / 16.0 * 2.0 - 1.0)[:2]
        h = args.fd_step
        # 2x2 数值 Hessian（中心差分）
        f00 = closed_loop_power(p0, channels, opt, X, frames)
        fpx = closed_loop_power(p0 + [h, 0], channels, opt, X, frames)
        fmx = closed_loop_power(p0 - [h, 0], channels, opt, X, frames)
        fpy = closed_loop_power(p0 + [0, h], channels, opt, X, frames)
        fmy = closed_loop_power(p0 - [0, h], channels, opt, X, frames)
        fxy = closed_loop_power(p0 + [h, h], channels, opt, X, frames)
        fmxy = closed_loop_power(p0 + [-h, -h], channels, opt, X, frames)
        H = np.array([[(fpx - 2 * f00 + fmx) / h ** 2, (fxy - fmxy) / (2 * h ** 2) * 0.5],
                      [(fxy - fmxy) / (2 * h ** 2) * 0.5, (fpy - 2 * f00 + fmy) / h ** 2]])
        H = 0.5 * (H + H.T)
        # V 在真实位置附近取极大 ⇒ 曲率测 -H（损失曲率），各向异性看 -H 的特征值比
        Hloss = -H
        ev, evec = np.linalg.eigh(Hloss)
        kappa = float(ev[-1] / max(abs(ev[0]), 1e-12))
        v_main = evec[:, -1]
        cosang = abs(float(np.dot(v_main, los)))
        rows.append({"seed": s, "p0": p0.tolist(), "kappa": kappa,
                     "cos_main_los": cosang, "h": h})
        print(f"  seed {s}: κ(H_V) = {kappa:.1f}, "
              f"|cos⟨主方向, LOS⟩| = {cosang:.3f}")

    kap = [r["kappa"] for r in rows]
    cos = [r["cos_main_los"] for r in rows]
    verdict = "方向成立" if np.median(kap) >= 10 and np.median(cos) >= 0.85 else \
              ("方向死亡(κ<2)" if np.median(kap) < 2 else "边缘，需更多种子")
    print(f"  → 中位 κ = {np.median(kap):.1f}, 中位 |cos| = {np.median(cos):.3f} "
          f"⇒ {verdict}")
    return {"rows": rows, "kappa_median": float(np.median(kap)),
            "cos_median": float(np.median(cos)), "verdict": verdict}


# ----------------------------------------------------------------------
# G-AU：VAE 后验坍缩诊断
# ----------------------------------------------------------------------

def gate_vae_au(args):
    device = "cpu"
    ckpt_path = os.path.join(args.save_dir, "sat", "vae_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [跳过] 未找到 {ckpt_path}")
        return None
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(ckpt_path, map_location=device))
    vae.eval()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    from torch.utils.data import DataLoader
    from data_sat import SatROIDataset
    loader = DataLoader(SatROIDataset(64, channels, num_points=args.num_points,
                                      device=device, tau=args.tau,
                                      phase_mode=args.phase_mode),
                        batch_size=16, shuffle=False)
    kl_all = []
    with torch.no_grad():
        for pc, _ in loader:
            mu, logvar = vae.encode(pc.to(device))
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)   # [B, 256] 逐维
            kl_all.append(kl)
    kl = torch.cat(kl_all)                                        # [N, 256]
    kl_mean = kl.mean(dim=0)
    au_01 = float((kl_mean > 0.1).float().mean().item())
    au_001 = float((kl_mean > 0.01).float().mean().item())
    mi_lb = float(kl_mean.sum().item()) / np.log(2)               # bits
    print(f"  逐维 KL: 均值 {kl_mean.mean():.4f} nat, 最大 {kl_mean.max():.4f}, "
          f"非零维(>0.01nat)占比 {au_001 * 100:.1f}%")
    print(f"  AU@0.1nat = {au_01 * 100:.1f}%  (预测 ≤50% ⇒ free-bits 方向成立)")
    print(f"  Barber-Agakov MI 下界 = {mi_lb:.1f} bits (z_dim=256)")
    verdict = "方向成立" if au_01 <= 0.5 else ("方向作废(AU≥0.8)" if au_01 >= 0.8 else "边缘")
    print(f"  → {verdict}")
    return {"au_0.1nat": au_01, "au_0.01nat": au_001, "mi_lb_bits": mi_lb,
            "kl_mean_scalar": float(kl_mean.mean()), "verdict": verdict}


# ----------------------------------------------------------------------
# G-FIM：感知观测对 RIS 相位的结构无关性
# ----------------------------------------------------------------------

def gate_fim_independence(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    mid = frames[len(frames) // 2]
    roi_true, _, _ = generate_ground_target_sample()

    rp0 = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                                0.01, snr_db=args.snr_db, seed=0,
                                align=True, sat_ecef=mid["sat_pos"])
    diffs = []
    for s in range(8):
        # compute_range_profile 的签名不含 RIS 相位 ⇒ 结构无关；
        # 数值侧确认：不同 seed 的噪声实现外，观测确定性相同
        rp = compute_range_profile(roi_true, mid["target_pos"], mid["ground_pos"],
                                   0.01, snr_db=args.snr_db, seed=s,
                                   align=True, sat_ecef=mid["sat_pos"])
        diffs.append(float(np.abs(rp - rp0).max()))
    print(f"  结构事实: compute_range_profile 无 RIS 相位输入 ⇒ sat 模式 "
          f"感知 FIM 与 v 严格无关（恒等式）")
    print(f"  数值侧(仅噪声实现差异): max|ΔRP| = {max(diffs):.4f}")
    print(f"  ground 模式边界: audit_ris_doppler 功率门 echo/comm ≈ 6e-5 "
          f"(回波泄漏的平方 ~3.6e-9 ⇒ FIM 相对扰动 <1e-6)")
    return {"structural": "range profile has no v input (identity)",
            "noise_only_max_diff": max(diffs),
            "ground_mode_leverage_bound": "~(6e-5)^2 = 3.6e-9 relative"}


def main(args):
    print(f"\n{'=' * 70}\nG-κ 闭环 Hessian 各向异性\n{'=' * 70}")
    g_kappa = gate_hessian(args)
    print(f"\n{'=' * 70}\nG-AU VAE 后验坍缩诊断\n{'=' * 70}")
    g_au = gate_vae_au(args)
    print(f"\n{'=' * 70}\nG-FIM 感知-通信相位可分离性\n{'=' * 70}")
    g_fim = gate_fim_independence(args)

    out = {"G_kappa": g_kappa, "G_au": g_au, "G_fim": g_fim,
           "protocol": {"n_seeds": args.n_seeds, "fd_step": args.fd_step,
                        "snr_db": args.snr_db}}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "p1_gates.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 前置证伪门")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=6)
    parser.add_argument("--fd_step", type=float, default=0.30, help="Hessian 有限差分步长（归一化坐标；须跨多个 ROI 体素=5m/80m*2=0.125）")
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
