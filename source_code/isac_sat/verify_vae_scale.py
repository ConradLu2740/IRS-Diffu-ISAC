"""
verify_vae_scale.py — VAE 训练规模实验（M1 定位的瓶颈：VAE/训练规模方差）

M1 发现：生成质量距 VAE 天花板 15–20×，跨种子 CV 18%，且 VAE 在 50 epoch
时仍在快速改善（40→50 epoch 仍有 +24~29% test_cd 改善）⇒ VAE **欠训练**而非
容量受限。本脚本对比：
  - 基线：vae_epochs=50,  train_data=256 （sat_model_m1_<seed>）
  - 缩放：vae_epochs=200, train_data=1024（sat_model_scale_<seed>）
同种子配对，指标：VAE 天花板、FM/DDPM 各 NFE 的 CD、跨种子 CV。

预注册命题：
  V1 VAE test_cd(ep200) < 0.010（基线 ep50 为 0.0148–0.0173）
  V2 同种子 FM(NFE=10) CD 改善 ≥15%
  V3 生成 CD 跨种子 CV 从 18% 降到 <12%
  V4 VAE 天花板（同评估批）同步下降 ≥20%
"""

import os
import json
import argparse
import numpy as np


def _load(save_dir, seed, tag):
    path = os.path.join(save_dir, f"sat_model_{tag}_{seed}", "compare_gen.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sat = d.get("sat", d)
    return {"ddpm100": float(sat["ddpm"]["100"]["cd"]),
            "fm": {int(k): float(v["cd"]) for k, v in sat["fm"].items()},
            "vae": float(sat["vae_oracle"]["cd"])}


def _vae_curve(save_dir, seed, tag):
    path = os.path.join(save_dir, f"sat_model_{tag}_{seed}", "sat", "vae_history.npy")
    if not os.path.exists(path):
        return None
    h = np.load(path, allow_pickle=True).item()
    tc = np.array(h["test_cd"])
    return {"ep50": float(tc[49]) if len(tc) >= 50 else None,
            "epfinal": float(tc[-1]), "n_epochs": len(tc),
            "last10_gain_pct": float(100 * (tc[-11] - tc[-1]) / tc[-11]) if len(tc) > 10 else None}


def cv(x):
    x = np.asarray(x)
    return float(x.std() / x.mean() * 100)


def main(args):
    base, scaled = "./", "./"
    b = {sd: _load(base, sd, "m1") for sd in args.seeds}
    s = {sd: _load(scaled, sd, "scale") for sd in args.seeds}
    b = {k: v for k, v in b.items() if v}
    s = {k: v for k, v in s.items() if v}
    if not b or not s:
        print("缺少基线或缩放运行结果")
        return

    print(f"\n{'=' * 78}")
    print("VAE 训练规模实验（同种子配对：50ep/256样本 vs 200ep/1024样本）")
    print(f"{'=' * 78}")
    print(f"{'seed':>6} {'VAE天花板':>18} {'FM1':>16} {'FM10':>16} {'DDPM100':>16}")
    rows = []
    for sd in sorted(set(b) & set(s)):
        vb, vs = b[sd]["vae"], s[sd]["vae"]
        f1b, f1s = b[sd]["fm"][1], s[sd]["fm"][1]
        f10b, f10s = b[sd]["fm"][10], s[sd]["fm"][10]
        db, ds = b[sd]["ddpm100"], s[sd]["ddpm100"]
        rows.append({"seed": sd, "vae_base": vb, "vae_scaled": vs,
                     "fm1_base": f1b, "fm1_scaled": f1s,
                     "fm10_base": f10b, "fm10_scaled": f10s,
                     "ddpm_base": db, "ddpm_scaled": ds})
        print(f"{sd:>6} {vb:>8.4f}→{vs:>8.4f} {f1b:>7.4f}→{f1s:>7.4f} "
              f"{f10b:>7.4f}→{f10s:>7.4f} {db:>7.4f}→{ds:>7.4f}")

    gains = {
        "vae": [100 * (r["vae_base"] - r["vae_scaled"]) / r["vae_base"] for r in rows],
        "fm1": [100 * (r["fm1_base"] - r["fm1_scaled"]) / r["fm1_base"] for r in rows],
        "fm10": [100 * (r["fm10_base"] - r["fm10_scaled"]) / r["fm10_base"] for r in rows],
        "ddpm": [100 * (r["ddpm_base"] - r["ddpm_scaled"]) / r["ddpm_base"] for r in rows],
    }
    print("-" * 78)
    for k, v in gains.items():
        print(f"  {k:>5} 改善: {[f'{x:+.1f}%' for x in v]}  均值 {np.mean(v):+.1f}%")

    cv_b = cv([r["fm10_base"] for r in rows])
    cv_s = cv([r["fm10_scaled"] for r in rows])
    print(f"\n  FM10 跨种子 CV: 基线 {cv_b:.1f}% → 缩放 {cv_s:.1f}%"
          f"（M1 全量基线 18.0%）")

    # VAE 曲线诊断
    print(f"\n  VAE 训练曲线诊断:")
    for sd in sorted(set(b) & set(s)):
        cb = _vae_curve("./", sd, "m1")
        cs = _vae_curve("./", sd, "scale")
        if cb and cs:
            print(f"    seed {sd}: 基线末10ep改善 {cb['last10_gain_pct']:+.1f}% "
                  f"(ep50={cb['ep50']:.4f}) → 缩放末10ep改善 {cs['last10_gain_pct']:+.1f}% "
                  f"(ep{cs['n_epochs']}={cs['epfinal']:.4f})")

    verdicts = {
        "V1_vae_cd_below_0.010": bool(all(r["vae_scaled"] < 0.010 for r in rows)),
        "V2_fm10_gain_ge_15pct": bool(all(g >= 15 for g in gains["fm10"])),
        "V3_cv_below_12pct": bool(cv_s < 12),
        "V4_vae_gain_ge_20pct": bool(all(g >= 20 for g in gains["vae"])),
    }
    print(f"\n  裁决: {json.dumps(verdicts, indent=2)}")

    out = {"rows": rows, "gains_pct": gains,
           "cv_fm10_base": cv_b, "cv_fm10_scaled": cv_s,
           "m1_baseline_cv": 18.0, "verdicts": verdicts}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "vae_scale.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE 训练规模实验")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()
    main(args)
