"""
verify_gen_scale.py — M3：生成模型训练预算实验（VAE 固定为缩放版）

背景：VAE 缩放实验（§7.8）发现瓶颈从 VE 转移到生成模型自身的训练预算——
VAE 天花板减半（0.008）后，生成 CD（0.15–0.27）距天花板 20–30×。
本实验固定 VAE（复用 sat_model_scale_<seed> 的 200ep/1024 VAE），
把生成训练 100 → 400 epoch，检验生成质量是否向新天花板收敛。

对比基线：sat_model_scale_<seed>（同 VAE，生成 100 epoch）

预注册命题：
  G-a FM NFE=10 CD 下降 ≥20%
  G-b 间隙比（生成 CD / VAE 天花板）从 ~30× 降到 ≤20×
  G-c FM NFE=1 ≤ FM NFE=100（少步性质在更长训练下保持）
  G-d FM NFE=1 vs DDPM NFE=100 的差跨种子符号一致（M2 的 3 种子版前置）
"""

import os
import json
import argparse
import numpy as np


def _load(base, seed, tag):
    path = os.path.join(base, f"sat_model_{tag}_{seed}", "compare_gen.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sat = d.get("sat", d)
    return {"ddpm100": float(sat["ddpm"]["100"]["cd"]),
            "fm": {int(k): float(v["cd"]) for k, v in sat["fm"].items()},
            "vae": float(sat["vae_oracle"]["cd"])}


def main(args):
    b = {sd: _load("./", sd, "scale") for sd in args.seeds}
    m = {sd: _load("./", sd, "m3") for sd in args.seeds}
    b = {k: v for k, v in b.items() if v}
    m = {k: v for k, v in m.items() if v}
    if not b or not m:
        print("缺少基线（scale）或 M3 运行结果")
        return

    print(f"\n{'=' * 80}")
    print("M3 生成训练预算实验（VAE 固定 200ep/1024；生成 100ep → 400ep）")
    print(f"{'=' * 80}")
    print(f"{'seed':>6} {'VAE':>7} {'FM1':>17} {'FM10':>17} {'FM100':>17} {'DDPM100':>17} "
          f"{'gap10':>7}")
    rows = []
    for sd in sorted(set(b) & set(m)):
        gap_b = b[sd]["fm"][10] / b[sd]["vae"]
        gap_m = m[sd]["fm"][10] / m[sd]["vae"]
        rows.append({"seed": sd, "vae": m[sd]["vae"],
                     "fm1_b": b[sd]["fm"][1], "fm1_m": m[sd]["fm"][1],
                     "fm10_b": b[sd]["fm"][10], "fm10_m": m[sd]["fm"][10],
                     "fm100_m": m[sd]["fm"][100],
                     "ddpm_b": b[sd]["ddpm100"], "ddpm_m": m[sd]["ddpm100"],
                     "gap10_b": gap_b, "gap10_m": gap_m})
        print(f"{sd:>6} {m[sd]['vae']:>7.4f} "
              f"{b[sd]['fm'][1]:>7.4f}→{m[sd]['fm'][1]:>7.4f} "
              f"{b[sd]['fm'][10]:>7.4f}→{m[sd]['fm'][10]:>7.4f} "
              f"{'':>7}→{m[sd]['fm'][100]:>7.4f} "
              f"{b[sd]['ddpm100']:>7.4f}→{m[sd]['ddpm100']:>7.4f} "
              f"{gap_b:>6.0f}→{gap_m:>4.0f}×")

    g10 = [100 * (r["fm10_b"] - r["fm10_m"]) / r["fm10_b"] for r in rows]
    gap_red = [r["gap10_m"] / r["gap10_b"] for r in rows]
    lownfe_ok = [r["fm1_m"] <= r["fm100_m"] for r in rows]
    sign_ok = [r["fm1_m"] < r["ddpm_m"] for r in rows]
    print("-" * 80)
    print(f"  FM10 改善: {[f'{x:+.1f}%' for x in g10]}")
    print(f"  间隙比缩放: {[f'{x:.2f}' for x in gap_red]}")
    print(f"  G-c FM1 ≤ FM100: {lownfe_ok}")
    print(f"  G-d FM1 < DDPM100: {sign_ok}")

    verdicts = {
        "Ga_fm10_gain_ge_20pct": bool(all(g >= 20 for g in g10)),
        "Gb_gap_ratio_le_20x": bool(all(r["gap10_m"] <= 20 for r in rows)),
        "Gc_fm1_le_fm100": bool(all(lownfe_ok)),
        "Gd_fm1_lt_ddpm100_all_seeds": bool(all(sign_ok)),
    }
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"rows": rows, "gains_fm10_pct": g10, "gap_ratios": gap_red,
           "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "gen_scale.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 生成训练预算实验")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()
    main(args)
