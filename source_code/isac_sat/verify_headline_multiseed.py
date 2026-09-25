"""
verify_headline_multiseed.py — M1：头条数字的多种子配对验证

背景：四次同协议 DDPM 运行（不同 flag/种子）CD 在 0.36–0.56（±20%），
单跑 A/B 不足以支撑头条。本脚本聚合 3 个种子（固定配置：后验样本目标 +
逐维白化 + lr_cond=1e-4）的 compare_gen 结果，检验：

  M1a FM(NFE=1) − DDPM(NFE=100) 的差跨种子符号一致（FM 更小）
  M1b FM(NFE=10) − DDPM(NFE=100) 跨种子符号一致
  M1c 逐种子报告 mean±std 与 Bootstrap 95% CI

输出 isac_demo/headline_multiseed.json。
"""

import os
import json
import argparse
import numpy as np


def load_seed(save_dir, seed):
    path = os.path.join(save_dir, f"sat_model_m1_{seed}", "compare_gen.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sat = d.get("sat", d)
    return {"ddpm100": float(sat["ddpm"]["100"]["cd"]),
            "fm": {int(k): float(v["cd"]) for k, v in sat["fm"].items()},
            "vae": float(sat["vae_oracle"]["cd"]),
            "fs_fm1": float(sat["fm"]["1"]["fs_0.1"]),
            "iou_fm1": float(sat["fm"]["1"]["iou"])}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main(args):
    rows = []
    for sd in args.seeds:
        r = load_seed("./", sd)
        if r is None:
            print(f"  [跳过] seed {sd} 无结果")
            continue
        r["seed"] = sd
        rows.append(r)
    if not rows:
        print("没有可聚合的结果；先运行 compare_gen --save_dir ./sat_model_m1_<seed>")
        return

    print(f"\n{'=' * 74}")
    print(f"M1 多种子头条验证（{len(rows)} 种子，固定配置：后验样本 + 逐维白化 + lr_cond=1e-4）")
    print(f"{'=' * 74}")
    print(f"{'seed':>6} {'DDPM100':>9} {'FM1':>9} {'FM10':>9} {'FM100':>9} "
          f"{'FM1-DDPM':>10} {'FM10-DDPM':>11} {'VAE':>8}")
    for r in rows:
        d1 = r["fm"][1] - r["ddpm100"]
        d10 = r["fm"][10] - r["ddpm100"]
        print(f"{r['seed']:>6} {r['ddpm100']:>9.4f} {r['fm'][1]:>9.4f} {r['fm'][10]:>9.4f} "
              f"{r['fm'][100]:>9.4f} {d1:>+10.4f} {d10:>+11.4f} {r['vae']:>8.4f}")

    d1s = [r["fm"][1] - r["ddpm100"] for r in rows]
    d10s = [r["fm"][10] - r["ddpm100"] for r in rows]
    rel1 = [100 * x / r["ddpm100"] for x, r in zip(d1s, rows)]
    rel10 = [100 * x / r["ddpm100"] for x, r in zip(d10s, rows)]

    m1a = all(x < 0 for x in d1s)
    m1b = all(x < 0 for x in d10s)
    print("-" * 74)
    print(f"M1a FM(NFE=1) < DDPM(NFE=100) 跨种子符号一致: "
          f"{'PASS' if m1a else 'FAIL'}  (相对差 {[f'{x:+.1f}%' for x in rel1]})")
    print(f"M1b FM(NFE=10) < DDPM(NFE=100) 跨种子符号一致: "
          f"{'PASS' if m1b else 'FAIL'}  (相对差 {[f'{x:+.1f}%' for x in rel10]})")
    ci1 = boot_ci(rel1)
    ci10 = boot_ci(rel10)
    print(f"Bootstrap 95% CI: FM1-DDPM {ci1[0]:+.1f}% ~ {ci1[1]:+.1f}%; "
          f"FM10-DDPM {ci10[0]:+.1f}% ~ {ci10[1]:+.1f}%")
    ddpm_vals = [r["ddpm100"] for r in rows]
    print(f"DDPM100 跨种子: {np.mean(ddpm_vals):.4f} ± {np.std(ddpm_vals):.4f} "
          f"(CV {np.std(ddpm_vals) / np.mean(ddpm_vals) * 100:.1f}%)")

    out = {"n_seeds": len(rows),
           "rows": rows,
           "rel_diff_fm1_pct": rel1, "rel_diff_fm10_pct": rel10,
           "ci95_fm1": ci1, "ci95_fm10": ci10,
           "ddpm100_cv_pct": float(np.std(ddpm_vals) / np.mean(ddpm_vals) * 100),
           "verdicts": {"M1a_sign_consistent_fm1": bool(m1a),
                        "M1b_sign_consistent_fm10": bool(m1b)}}
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "isac_demo", "headline_multiseed.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 多种子头条验证")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    main(args)
