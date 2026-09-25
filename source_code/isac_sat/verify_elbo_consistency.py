"""
verify_elbo_consistency.py — ELBO 一致性修复 + 逐维白化的 A/B 证书

两项变分推断角度的不一致修复（默认已启用，旧行为可用 flag 复现）：
  1. posterior_sample: 传输模型（DDPM/FM）的训练目标从**后验均值 μ** 改为
     **后验样本 z~q_φ**（reparameterization）。ELBO 一致性：生成先验匹配的应是
     聚合后验 ∫q_φ(z|x)p(x|c)dx，而非确定性均值映射。
  2. whiten=perdim: 潜空间归一化从**标量** (z-μ)/σ 改为**逐维白化**
     (z-μ_i)/σ_i。标量归一化留下各向异性聚合后验，破坏"x1 ~ N(0,I) 逐维成立"
     这一 CFM/DDPM 全部理论恒等式的前提。

证书（本脚本实测）：
  - 白化证书：归一化后逐维 (mean, var) 与高斯 KL 0.5·Σ(μ_i²+σ_i²−1−ln σ_i²)
    标量 vs 逐维两种口径的直接对比（聚合后验与标准正态的距离）。
  - 质量 A/B：同一协议重训（compare_gen，仅这两个 flag 不同），CD(NFE) 对比。

预注册命题：
  W1 逐维白化使聚合后验高斯 KL 显著下降（≥30%）
  E1 质量中性或更优：|ΔCD(NFE)| ≤ 10% 对所有 NFE（一致性修复理论中性）
  E2 若先验失配偏差 b 显著，NFE=1 的改善大于 NFE=100（少步更受益）
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

from torch.utils.data import DataLoader
import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE


def whiten_certificate(ckpt_dir, args):
    """加载 VAE，在训练集潜变量上计算两种归一化口径的聚合后验高斯 KL。"""
    device = "cpu"
    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(ckpt_dir, "vae_best.pth"), map_location=device))
    vae.eval()

    # 留出集：与训练统计不同的新鲜样本（避免 in-sample 平凡化）
    torch.manual_seed(args.seed + 999); np.random.seed(args.seed + 999)
    sc = ss.SatISACScenario(tau=args.tau)
    frames = sc.build_frames()
    ch = SatScenarioChannels(frames, irs_mode="sat", device=device)
    loader = DataLoader(SatROIDataset(128, ch, num_points=args.num_points,
                                      device=device, tau=args.tau,
                                      phase_mode=args.phase_mode),
                        batch_size=32, shuffle=False)
    zs = []
    with torch.no_grad():
        for pc, _ in loader:
            mu, _ = vae.encode(pc)
            zs.append(mu)
    z = torch.cat(zs, dim=0)                        # [N, 256]
    # 统计/评估分离：前半估计归一化统计，后半留出评估（避免 in-sample 平凡化）
    n_stat = z.shape[0] // 2
    z_stat, z_eval = z[:n_stat], z[n_stat:]

    out = {}
    for name, mean, std in [
        ("scalar", z_stat.mean(), z_stat.std() + 1e-8),
        ("perdim", z_stat.mean(dim=0), z_stat.std(dim=0) + 1e-8),
    ]:
        zn = (z_eval - mean) / std
        m = zn.mean(dim=0)
        v = zn.var(dim=0, unbiased=False)
        kl = 0.5 * (m.pow(2) + v - 1 - v.clamp_min(1e-12).log()).sum().item()
        out[name] = {"gaussian_kl_nats": float(kl),
                     "mean_abs": float(m.abs().mean()), "var_mean": float(v.mean()),
                     "var_std": float(v.std())}
    out["kl_reduction"] = 1.0 - out["perdim"]["gaussian_kl_nats"] / max(
        out["scalar"]["gaussian_kl_nats"], 1e-12)
    return out


def load_cd(save_dir):
    path = os.path.join(save_dir, "compare_gen.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sat = d.get("sat", d)
    return {"ddpm": {int(k): v["cd"] for k, v in sat["ddpm"].items()},
            "fm": {int(k): v["cd"] for k, v in sat["fm"].items()},
            "vae": sat["vae_oracle"]["cd"]}


def main(args):
    print(f"\n{'=' * 70}\n1) 白化证书（聚合后验 → 标准正态的高斯 KL）\n{'=' * 70}")
    cert_new = whiten_certificate(os.path.join(args.save_dir, "sat"), args)
    cert_old = None
    for cand in [os.path.join("./sat_model_g15", "sat"), os.path.join("./sat_model_cmp", "sat")]:
        if os.path.exists(os.path.join(cand, "vae_best.pth")):
            cert_old = whiten_certificate(cand, args)
            print(f"  基线 VAE: {cand}")
            break
    print(f"  标量归一化 KL = {cert_new['scalar']['gaussian_kl_nats']:.1f} nat"
          f"  (逐维 var std={cert_new['scalar']['var_std']:.3f})")
    print(f"  逐维白化   KL = {cert_new['perdim']['gaussian_kl_nats']:.1f} nat"
          f"  (逐维 var std={cert_new['perdim']['var_std']:.3f})")
    print(f"  → KL 降幅 = {cert_new['kl_reduction'] * 100:.1f}%"
          f"  (W1 预测 ≥30%)")
    if cert_old is not None:
        print(f"  [基线 VAE] 标量 KL = {cert_old['scalar']['gaussian_kl_nats']:.1f}, "
              f"逐维 KL = {cert_old['perdim']['gaussian_kl_nats']:.1f}, "
              f"降幅 {cert_old['kl_reduction'] * 100:.1f}%")

    print(f"\n{'=' * 70}\n2) 质量 A/B（CD，同协议仅两个 flag 不同）\n{'=' * 70}")
    new = load_cd(args.save_dir)
    old = None
    for cand in ["./sat_model_g15", "./sat_model_cmp"]:
        old = load_cd(cand)
        if old:
            print(f"  基线: {cand}")
            break
    if new and old:
        print(f"  {'NFE':>5} {'FM旧':>9} {'FM新':>9} {'Δ%':>8}")
        deltas = []
        for nfe in sorted(new["fm"]):
            f_old, f_new = old["fm"].get(nfe), new["fm"].get(nfe)
            if f_old is None or f_new is None:
                continue
            fd = (f_new / f_old - 1) * 100
            deltas.append((nfe, None, fd))
            print(f"  {nfe:>5} {f_old:>9.4f} {f_new:>9.4f} {fd:>+8.1f}")
        # DDPM 只有 NFE=T 一个点
        d_old = old["ddpm"].get(100); d_new = new["ddpm"].get(100)
        if d_old and d_new:
            print(f"  DDPM NFE=100: 旧 {d_old:.4f} → 新 {d_new:.4f} "
                  f"({(d_new / d_old - 1) * 100:+.1f}%)")
        print(f"  VAE oracle: 旧 {old['vae']:.4f} → 新 {new['vae']:.4f}")
        max_abs = max(abs(x[2]) for x in deltas if x[2] == x[2])
        e1 = max_abs <= 10.0
        n1 = [x for x in deltas if x[0] == 1][0]
        n100 = [x for x in deltas if x[0] == 100][0]
        print(f"\n  E1 质量中性（max|ΔCD| ≤ 10%）: {'PASS' if e1 else 'FAIL'} "
              f"(max|Δ|={max_abs:.1f}%)")
        imp1, imp100 = -n1[2], -n100[2]
        print(f"  E2 NFE=1 改善 {imp1:+.1f}% vs NFE=100 改善 {imp100:+.1f}%: "
              f"{'少步改善更大（符合先验失配预测）' if imp1 > imp100 else '未见少步偏好（先验失配非主因）'}")
        verdicts = {"W1_kl_reduction_ge_30pct": bool(cert_new["kl_reduction"] >= 0.30),
                    "E1_quality_neutral": bool(e1),
                    "E2_lownfe_preferred": bool(-n1[2] > -n100[2]),
                    "max_abs_delta_pct": float(max_abs)}
    else:
        verdicts = {"W1_kl_reduction_ge_30pct": bool(cert_new["kl_reduction"] >= 0.30)}

    out = {"whiten_certificate_new": cert_new,
           "whiten_certificate_old": cert_old,
           "cd_ab": {"new": new, "old": old},
           "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "elbo_consistency.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELBO 一致性 + 白化 A/B 证书")
    parser.add_argument("--save_dir", type=str, default="./sat_model_elbo")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
