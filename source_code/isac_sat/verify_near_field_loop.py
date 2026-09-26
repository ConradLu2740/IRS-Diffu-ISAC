"""
verify_near_field_loop.py — NF-2：近场 vs 远场的 ML 定位效率对比（几何方向的神经网络验证）

问题：近场证书（F2：σ_R=65mm@100m）说近场几何可学；远场墙说交叉距离不可学。
本脚本用同一个 MLP 定位网络在两种 regime 的合成数据上估计**交叉距离 y**
（R 固定——墙的直接形式），检验：
  - 近场网络是否达到 CRB 效率（信息可被网络提取）
  - 远场网络的 y 误差是否被先验钉住（信息不存在，网络只能退回先验）

预注册命题：
  L1 近场网络效率 η = CRB/RMSE² ≥ 0.3（信息可学、网络有效）
  L2 远场 y-RMSE ≥ 10× 近场 y-RMSE（墙的 ML 形式）
  L3 远场 y-RMSE ≈ 先验 std（±20% 内）——后验=先验的直接证据
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from isac_sim.channels.near_field import NearFieldUla, make_dataset


class LocMLP(nn.Module):
    """单参数定位网络：阵列响应 → 交叉距离 y（R 固定）。"""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def run_regime(ula, R, farfield, args, phase_noise_deg=0.0):
    Xtr, Ytr = make_dataset(ula, args.n_train, (R, R), (-1.0, 1.0),
                            gamma_db=args.gamma_db, farfield=farfield, seed=1,
                            phase_noise_deg=phase_noise_deg)
    Xte, Yte = make_dataset(ula, args.n_test, (R, R), (-1.0, 1.0),
                            gamma_db=args.gamma_db, farfield=farfield, seed=2,
                            phase_noise_deg=phase_noise_deg)

    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysd = Ytr[:, 1].mean(), Ytr[:, 1].std() + 1e-8
    Xtr_n = torch.tensor((Xtr - mu) / sd)
    Ytr_n = torch.tensor(((Ytr[:, 1] - ymu) / ysd).astype(np.float32)).unsqueeze(1)
    model = LocMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xtr_n))
        for i in range(0, len(Xtr_n), 256):
            idx = perm[i:i + 256]
            loss = F.mse_loss(model(Xtr_n[idx]), Ytr_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor((Xte - mu) / sd)).numpy().ravel() * ysd + ymu
    rmse_y = float(np.sqrt(np.mean((pred - Yte[:, 1]) ** 2)))
    prior_std = float(Yte[:, 1].std())

    # y 的 CRB（解析，全局相位为未知干扰参数）
    crb_y = (ula.crb_y_farfield(R, args.gamma_db) if farfield
             else ula.crb_y_nuisance(R, args.gamma_db))
    return {
        "R_m": float(R), "rmse_y": rmse_y, "crb_y_std": float(np.sqrt(crb_y)),
        "prior_std_y": prior_std,
        "efficiency_vs_crb": float(crb_y / max(rmse_y ** 2, 1e-30)),
    }


def main(args):
    print("=" * 74)
    print("NF-2 近场 vs 远场 ML 定位（同网络、同数据规模、同训练，单参数 y）")
    print("=" * 74)

    ula = NearFieldUla(args.n_elements, args.aperture_m, args.fc_hz)
    R_F = ula.rayleigh_m()
    print(f"ULA: N={args.n_elements}, 孔径={args.aperture_m}m, "
          f"λ={ula.lam * 100:.1f}cm, Rayleigh 距离={R_F:.0f}m")

    # NF-3：相位校准噪声扫描（两种 regime）
    sweep = {}
    print("\n相位校准噪声扫描（MLP y-RMSE, mm）:")
    print(f"  {'σ_ψ(°)':>8} {'近场(100m)':>12} {'远场(1000m)':>12}")
    for pn in args.phase_noise_list:
        rn = run_regime(ula, args.r_near, False, args, phase_noise_deg=pn)
        rf = run_regime(ula, args.r_far, True, args, phase_noise_deg=pn)
        sweep[pn] = {"near": rn, "far": rf}
        print(f"  {pn:>8} {rn['rmse_y'] * 1000:>12.1f} {rf['rmse_y'] * 1000:>12.1f}")

    regimes = {}
    for name, farfield, R in [("near", False, args.r_near), ("far", True, args.r_far)]:
        r = run_regime(ula, R, farfield, args)
        r["R_over_RF"] = float(R / R_F)
        regimes[name] = r
        print(f"\n[{name}] R={R:.0f}m (R/R_F={R / R_F:.2f}), y∈[-1,1]m:")
        print(f"  MLP y-RMSE : {r['rmse_y'] * 1000:>9.2f}mm   （先验 std {r['prior_std_y'] * 1000:.0f}mm）")
        print(f"  CRB  y-std : {r['crb_y_std'] * 1000:>9.2f}mm   "
              f"(网络效率 η={r['efficiency_vs_crb']:.3f})")

    n, f = regimes["near"], regimes["far"]
    verdicts = {
        "L1_near_efficiency_ge_0.3": bool(n["efficiency_vs_crb"] >= 0.3),
        "L2_far_y_ge_10x_near": bool(f["rmse_y"] >= 10 * n["rmse_y"]),
        "L3_far_at_prior": bool(abs(f["rmse_y"] - f["prior_std_y"]) < 0.2 * f["prior_std_y"]),
    }
    better = [pn for pn, r in sweep.items() if r["near"]["rmse_y"] < r["far"]["rmse_y"]]
    verdicts["N1_near_beats_far_below_threshold"] = bool(len(better) > 0)
    if better:
        print(f"\n  NF-3: 近场优于远场的校准噪声范围 σ_ψ ≤ {max(better)}°")
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"regimes": regimes, "rayleigh_m": R_F, "verdicts": verdicts,
           "phase_noise_sweep": {str(k): {"near_rmse_y": v["near"]["rmse_y"],
                                          "far_rmse_y": v["far"]["rmse_y"]}
                                 for k, v in sweep.items()},
           "protocol": {"n_elements": args.n_elements, "aperture_m": args.aperture_m,
                        "gamma_db": args.gamma_db, "epochs": args.epochs}}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "near_field_loop.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NF-2 近场/远场 ML 定位对比")
    parser.add_argument("--n_elements", type=int, default=200)
    parser.add_argument("--aperture_m", type=float, default=1.0)
    parser.add_argument("--fc_hz", type=float, default=30e9)
    parser.add_argument("--r_near", type=float, default=100.0)
    parser.add_argument("--r_far", type=float, default=1000.0)
    parser.add_argument("--gamma_db", type=float, default=20.0)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--phase_noise_list", nargs="+", type=float,
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
                        help="逐单元相位校准残差幅度（度）扫描（NF-3）")
    args = parser.parse_args()
    main(args)
