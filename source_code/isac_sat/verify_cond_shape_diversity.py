"""
verify_cond_shape_diversity.py — N1/N2/N3：数据本身有没有"同条件不同形状"？

背景：§7.31 发现 C1 teacher NFE=1 近模式坍塌（16 个 x0 的输出 pairwise CD
仅 0.0053）。两种互斥解释：
  (a) 建模失败——数据里存在真实的"同 HRRP 条件、不同点云"多样性，teacher 把它
      平均掉了（训练侧可优化：多样性保持的正则/目标）；
  (b) 贝叶斯正确——给定 HRRP 条件，后验本来就接近点质量（条件均值估计器就是
      最优解），学生的"多样性"是噪声不是信息。
本脚本用非参数最近邻分析裁定：在训练数据分布上，条件距离（HRRP 剖面 L2）
与形状距离（点云 CD）的关系。注意 HRRP 条件质心对齐（不含绝对位置），
而点云保留 ROI 内绝对位置——因此分别报告原始 CD（位置+形状）与质心对齐
CD（纯形状），把平移多样性与形状多样性分开。

预注册命题（docs/optimization_roadmap.md §7.33）：
  N1 位置维条件多样性存在：最近 10% 条件对的 CD_raw ≥ 3 × 0.0053
     （0.0053 = §7.31 teacher 坍塌 spread 的实测参考）
  N2 纯形状条件多样性存在：最近 10% 条件对的 CD_aligned ≥ 2 × 0.0053
  N3 条件携带形状信息：Spearman ρ(条件距离, CD_aligned) ≥ 0.5
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import torch
import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from train import chamfer_distance_loss

TEACHER_COLLAPSED_PAIRWISE = 0.0053   # §7.31 实测：teacher 16×x0 的 pairwise CD


def pairwise_cd(pcs):
    """[N, P, 3] -> [N, N] chamfer 距离矩阵。"""
    N = pcs.shape[0]
    D = torch.zeros((N, N), device=pcs.device)
    for i in range(N):
        a = pcs[i:i + 1].expand(N, -1, -1)
        D[i] = chamfer_distance_loss(a, pcs)
    return D


def spearman(x, y):
    """numpy 实现的 Spearman 秩相关（无 scipy 依赖）。"""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main(args):
    device = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random; random.seed(args.seed)

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    ds = SatROIDataset(args.n_samples, channels, num_points=args.num_points,
                       device=device, tau=args.tau, phase_mode=args.phase_mode,
                       cond_feat=args.cond_feat)

    pcs, feats = [], []
    for i in range(args.n_samples):
        pc, cond, feat = ds[i]
        pcs.append(pc.to(device))
        feats.append(feat.to(device))
    pcs = torch.stack(pcs)                 # [N, P, 3]
    feats = torch.stack(feats)             # [N, K]
    print(f"样本: pcs {tuple(pcs.shape)}, HRRP 特征 {tuple(feats.shape)}")

    # 条件距离（HRRP 剖面 L2）与形状距离（原始 / 质心对齐）
    cd_cond = torch.cdist(feats, feats)                       # [N, N]
    cd_raw = pairwise_cd(pcs)
    pcs_al = pcs - pcs.mean(dim=1, keepdim=True)
    cd_al = pairwise_cd(pcs_al)

    N = args.n_samples
    iu = torch.triu_indices(N, N, offset=1)
    d_cond = cd_cond[iu[0], iu[1]].cpu().numpy()
    d_raw = cd_raw[iu[0], iu[1]].cpu().numpy()
    d_al = cd_al[iu[0], iu[1]].cpu().numpy()

    # 最近 10% 条件对
    k = max(1, int(0.1 * len(d_cond)))
    idx_near = np.argsort(d_cond)[:k]
    near_raw, near_al = d_raw[idx_near], d_al[idx_near]

    rho = spearman(d_cond, d_al)

    verdicts = {
        "N1_near_cond_pairs_cd_raw_ge_3x_teacher": bool(
            near_raw.mean() >= 3.0 * TEACHER_COLLAPSED_PAIRWISE),
        "N2_near_cond_pairs_cd_aligned_ge_2x_teacher": bool(
            near_al.mean() >= 2.0 * TEACHER_COLLAPSED_PAIRWISE),
        "N3_spearman_cond_vs_shape_ge_0.5": bool(rho >= 0.5),
    }

    print(f"\n{'=' * 70}")
    print(f"全部 {len(d_cond)} 个条件对：CD_raw {d_raw.mean():.4f} ± {d_raw.std():.4f}"
          f"  CD_aligned {d_al.mean():.4f} ± {d_al.std():.4f}")
    print(f"最近 10% 条件对（n={k}）：CD_raw {near_raw.mean():.4f} ± {near_raw.std():.4f}"
          f"  CD_aligned {near_al.mean():.4f} ± {near_al.std():.4f}")
    print(f"Spearman ρ(条件距离, CD_aligned) = {rho:.3f}")
    print(f"\n参考：teacher 坍塌 spread = {TEACHER_COLLAPSED_PAIRWISE:.4f}（§7.31）")
    print(f"  N1 阈值 3× = {3 * TEACHER_COLLAPSED_PAIRWISE:.4f}；"
          f"N2 阈值 2× = {2 * TEACHER_COLLAPSED_PAIRWISE:.4f}")
    print(f"裁决: {verdicts}")
    print("\n解读：N1 过/N2 不过 → 多样性主要是平移（条件不含位置）；两者皆过 → "
          "存在纯形状多样性，teacher 坍塌是建模损失；N3 不过 → 条件对形状信息量弱。")

    out = {"n_samples": N, "n_pairs": int(len(d_cond)),
           "cd_raw_mean": float(d_raw.mean()), "cd_raw_std": float(d_raw.std()),
           "cd_aligned_mean": float(d_al.mean()), "cd_aligned_std": float(d_al.std()),
           "near_decile": {"k": int(k),
                           "cd_raw_mean": float(near_raw.mean()),
                           "cd_raw_std": float(near_raw.std()),
                           "cd_aligned_mean": float(near_al.mean()),
                           "cd_aligned_std": float(near_al.std())},
           "spearman_cond_vs_shape_aligned": rho,
           "teacher_collapsed_pairwise_ref": TEACHER_COLLAPSED_PAIRWISE,
           "verdicts": verdicts}
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "cond_shape_diversity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N1/N2/N3：条件-形状多样性分析")
    parser.add_argument("--n_samples", type=int, default=96)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                        default="hrrp")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./sat_model_c1")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
