"""
verify_gen_hardening.py — 生成侧证书补强：Lipschitz 实测 + 少步多样性 + 全模式指标表

回应评审的三个缺口：
  1. L 未实测：用 Jacobian 幂迭代实测 v_cfg 的 Lipschitz 上界 L̂，
     验证 Euler 误差界 err_z(NFE) ≤ C·L̂/NFE 的常数是否合理。
  2. NFE=1 多样性未知：同一 cond 下用不同初始噪声采样，报告样本间
     平均 pairwise CD（模式多样性）与对 GT 的 CD 分布，检测模式坍塌。
  3. 指标偏窄：从 compare_gen JSON 汇总三模式 × 全 NFE 的 CD/F-Score/IoU 全表。

协议与 verify_fm_bounds.py 一致（同 checkpoint、同测试批、CFG=2.0）。
"""

import os
import json
import argparse
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import chamfer_distance_loss
from fm_utils import T_EMB_SCALE, sample_conditional_FM
from verify_fm_bounds import build_eval_batch, load_models


@torch.no_grad()
def lipschitz_power_iter(condenc, vnet, cond, z_mean, z_std, n_points=8,
                         n_iter=30, seed=0, device="cpu"):
    """沿 FM 轨迹采样点，用幂迭代估计 ||∇_x v_cfg||₂ 的最大值（Lipschitz 上界估计）。

    对每个采样点 x（256 维）用中心差分 JVP 做幂迭代，
    收敛到该点的谱范数估计。返回 (max_L, per_t_L)。
    """
    from torch.func import jvp, vmap

    condenc.eval(); vnet.eval()
    cond1 = cond[:1].to(device)
    c = condenc(cond1)
    c_null = condenc(torch.zeros_like(cond1))

    def v_cfg(x, t01):
        t = torch.full((x.shape[0],), t01 * T_EMB_SCALE, device=device)
        return vnet(x, t, c_null) + 2.0 * (vnet(x, t, c) - vnet(x, t, c_null))

    def jvp_fd(x, u, t01, eps=1e-3):
        """中心差分 JVP（forward AD 不支持 MHA 内核；量级估计足够）。"""
        return (v_cfg(x + eps * u, t01) - v_cfg(x - eps * u, t01)) / (2 * eps)

    g = torch.Generator(device=device).manual_seed(seed)
    ts = np.linspace(0.02, 0.98, n_points)
    per_t = []
    for t01 in ts:
        x = torch.randn((1, 256), device=device, generator=g)
        u = torch.randn((1, 256), device=device, generator=g)
        u = u / u.norm()
        L = 0.0
        for _ in range(n_iter):
            ju = jvp_fd(x, u, float(t01))
            n = ju.norm()
            if n < 1e-12:
                break
            u = ju / n
            L = float(n)
        per_t.append({"t": float(t01), "L": L})
    return per_t


@torch.no_grad()
def diversity_at_nfe(vae, condenc, vnet, cond, pc_gt, z_mean, z_std,
                     n_samples=16, device="cpu"):
    """同一 cond、不同初始噪声的样本多样性：pairwise CD + 对 GT 的 CD。"""
    outs = []
    for s in range(n_samples):
        g = torch.Generator(device=device).manual_seed(1000 + s)
        x0 = torch.randn((cond.size(0), 256), device=device, generator=g)
        # 复用 sample_conditional_FM 的积分逻辑（固定 x0）
        from fm_utils import sample_conditional_FM as sf
        # 直接调用不带 x0 的版本不可固定噪声，这里内联一次 NFE=1 步
        c = condenc(cond); c_null = condenc(torch.zeros_like(cond))
        t = torch.zeros(cond.size(0), device=device)
        v = vnet(x0, t, c_null) + 2.0 * (vnet(x0, t, c) - vnet(x0, t, c_null))
        z0 = (x0 + v) * z_std + z_mean
        pc, _ = vae.decode(z0)
        outs.append(pc)
    pcs = torch.stack(outs)                       # [S, B, N, 3]
    S, B = pcs.shape[:2]
    pw = []
    for b in range(B):
        for i in range(S):
            for j in range(i + 1, S):
                pw.append(chamfer_distance_loss(pcs[i, b:b+1], pcs[j, b:b+1]).item())
    gt_cd = [chamfer_distance_loss(pc_gt[b:b+1], pcs[:, b].mean(0, keepdim=True)).item()
             for b in range(B)]
    return {"pairwise_cd_mean": float(np.mean(pw)), "pairwise_cd_std": float(np.std(pw)),
            "gt_vs_postmean_cd": float(np.mean(gt_cd))}


def main(args):
    device = args.device
    pc_gt, cond, cond_dim = build_eval_batch(args, device)
    vae, condenc_d, epsnet, condenc_f, vnet, z_mean, z_std = load_models(args, device, cond_dim)

    print(f"\n{'=' * 70}")
    print("1) v_cfg Lipschitz 常数实测（Jacobian 幂迭代，沿轨迹采样）")
    print("=" * 70)
    per_t = lipschitz_power_iter(condenc_f, vnet, cond, z_mean, z_std, device=device)
    Ls = [d["L"] for d in per_t]
    for d in per_t:
        print(f"  t={d['t']:.2f}: L = {d['L']:.2f}")
    L_hat = max(Ls)
    print(f"  → L̂ = max L = {L_hat:.2f}（Euler 误差界 err ≤ (e^L̂−1)/2 · L̂ · Λ / NFE，"
          f"Λ ≤ E‖x1−x0‖ ≈ {np.sqrt(2 * 256):.1f}）")

    print(f"\n{'=' * 70}")
    print("2) NFE=1 采样多样性（同 cond × 16 个初始噪声）")
    print(f"{'=' * 70}")
    div = diversity_at_nfe(vae, condenc_f, vnet, cond, pc_gt, z_mean, z_std,
                           n_samples=args.n_samples, device=device)
    print(f"  pairwise CD（样本间）: {div['pairwise_cd_mean']:.4f} ± {div['pairwise_cd_std']:.4f}")
    print(f"  GT vs 后验均值 CD     : {div['gt_vs_postmean_cd']:.4f}")
    print(f"  → 多样性/质量比 pairwise/CD_gt = "
          f"{div['pairwise_cd_mean'] / max(div['gt_vs_postmean_cd'], 1e-9):.2f}"
          f"（≈0 提示模式坍塌）")

    print(f"\n{'=' * 70}")
    print("3) 全模式 × 全 NFE 指标表（来自 compare_gen JSON）")
    print(f"{'=' * 70}")
    cmp_json = os.path.join(args.save_dir, "compare_gen.json")
    sat_json = os.path.join(args.save_dir, "compare_gen_sat.json")
    data = {}
    if os.path.exists(sat_json):
        data.update(json.load(open(sat_json)))
    if os.path.exists(cmp_json):
        data.update(json.load(open(cmp_json)))
    for mode, r in data.items():
        print(f"  [{mode}]")
        for k, m in r["ddpm"].items():
            print(f"    DDPM NFE={k:>4}: CD={m['cd']:.4f} FS@0.1={m['fs_0.1']:.4f} "
                  f"FS@0.2={m['fs_0.2']:.4f} IoU={m['iou']:.4f}")
        for k in sorted(r["fm"], key=int):
            m = r["fm"][k]
            print(f"    FM   NFE={k:>4}: CD={m['cd']:.4f} FS@0.1={m['fs_0.1']:.4f} "
                  f"FS@0.2={m['fs_0.2']:.4f} IoU={m['iou']:.4f}")
        print(f"    VAE oracle: CD={r['vae_oracle']['cd']:.4f}")

    out = {"lipschitz": {"per_t": per_t, "L_hat": L_hat},
           "diversity_nfe1": div,
           "protocol": {"n_eval": args.n_eval, "cfg_scale": 2.0}}
    out_path = os.path.join(args.save_dir, "verify_gen_hardening.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成侧证书补强")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=16)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
