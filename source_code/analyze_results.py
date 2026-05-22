import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def load_results(results_path):
    with open(results_path, "r") as f:
        return json.load(f)


def group_by_exp_key(results):
    groups = {}
    for label, entry in results.items():
        if entry.get("status") != "ok":
            continue
        exp_key = entry["exp_key"]
        cd_eval = entry.get("cd_eval")
        if cd_eval is None or cd_eval.get("cd_list") is None:
            continue
        groups.setdefault(exp_key, []).extend(cd_eval["cd_list"])
    return groups


def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
    rng = np.random.RandomState(42)
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, n, replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi, np.mean(means)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s_pooled = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / s_pooled


def run_statistical_tests(groups, baseline_key="none"):
    try:
        from scipy import stats as scipy_stats
        from scipy.stats import ttest_ind, f_oneway
    except ImportError:
        print("[WARN] scipy not available, skipping statistical tests")
        return {}, {}

    baseline = groups.get(baseline_key)
    if baseline is None:
        print(f"[WARN] baseline '{baseline_key}' not found")
        return {}, {}

    tests = {}
    effect_sizes = {}

    for exp_key, cds in groups.items():
        if exp_key == baseline_key:
            continue
        t_stat, p_val = ttest_ind(baseline, cds, equal_var=False)
        d = cohens_d(cds, baseline)
        tests[exp_key] = {"t_statistic": float(t_stat), "p_value": float(p_val)}
        effect_sizes[exp_key] = {"cohens_d": float(d), "vs": baseline_key}

    group_list = [groups[k] for k in groups if k in groups]
    if len(group_list) >= 2:
        f_stat, p_anova = f_oneway(*group_list)
        tests["anova"] = {"F_statistic": float(f_stat), "p_value": float(p_anova)}

    return tests, effect_sizes


def generate_report(groups, tests, effect_sizes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("# IRS_Diffu_ISAC 公平实验对比报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("- 统一架构: PointVAE(Token) + LatentDiT_Token_CrossAttn + AdvancedCondEncoder(80dim)")
    lines.append("- 每组实验: 5 次独立训练 (seeds: 42, 123, 456, 789, 1024)")
    lines.append("")
    lines.append("## Chamfer Distance 汇总")
    lines.append("")
    lines.append("| 实验组 | 平均CD | 标准差 | 95% CI (低) | 95% CI (高) | 样本数 |")
    lines.append("|--------|--------|--------|-------------|-------------|--------|")
    for exp_key in sorted(groups.keys()):
        cds = groups[exp_key]
        mean_cd = np.mean(cds)
        std_cd = np.std(cds)
        lo, hi, _ = bootstrap_ci(cds)
        lines.append(f"| {exp_key:12s} | {mean_cd:.6f} | {std_cd:.6f} | {lo:.6f} | {hi:.6f} | {len(cds)} |")
    lines.append("")

    baseline_key = "none"
    if baseline_key in groups:
        lines.append("## 效应量 (Cohen's d vs No IRS)")
        lines.append("")
        lines.append("| 实验组 | Cohen's d | 解释 |")
        lines.append("|--------|-----------|------|")
        for exp_key, es in sorted(effect_sizes.items()):
            d = abs(es.get("cohens_d", 0))
            if d < 0.2:
                interp = "可忽略"
            elif d < 0.5:
                interp = "小效应"
            elif d < 0.8:
                interp = "中等效应"
            else:
                interp = "大效应"
            lines.append(f"| {exp_key:12s} | {d:+.4f} | {interp} |")
        lines.append("")

    if tests:
        lines.append("## 统计检验")
        lines.append("")
        for name, t in sorted(tests.items()):
            if name == "anova":
                lines.append(f"- **One-way ANOVA**: F = {t['F_statistic']:.4f}, p = {t['p_value']:.4f}")
            else:
                sig = "***" if t["p_value"] < 0.001 else "**" if t["p_value"] < 0.01 else "*" if t["p_value"] < 0.05 else "n.s."
                lines.append(f"- **{name} vs {baseline_key}**: t = {t['t_statistic']:.4f}, p = {t['p_value']:.4f} {sig}")
        lines.append("")
        lines.append("显著性标记: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001, n.s. = 不显著")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Save] report saved to {report_path}")


def generate_figures(groups, tests, effect_sizes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    colors = {"none": "#4ECDC4", "zero": "#45B7D1", "random": "#FF6B6B", "optimized": "#96CEB4"}
    exp_order = ["none", "zero", "random", "optimized"]
    present_keys = [k for k in exp_order if k in groups]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    data = [groups[k] for k in present_keys]
    labels = present_keys
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, key in zip(bp['boxes'], present_keys):
        patch.set_facecolor(colors.get(key, "#CCCCCC"))
    ax1.set_ylabel("Chamfer Distance")
    ax1.set_title("CD Distribution by Experiment")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    means = [np.mean(groups[k]) for k in present_keys]
    stds = [np.std(groups[k]) for k in present_keys]
    cis_lo = []
    cis_hi = []
    for k in present_keys:
        lo, hi, _ = bootstrap_ci(groups[k])
        cis_lo.append(np.mean(groups[k]) - lo)
        cis_hi.append(hi - np.mean(groups[k]))
    bars = ax2.bar(range(len(present_keys)), means, yerr=[cis_lo, cis_hi],
                   color=[colors.get(k, "#CCCCCC") for k in present_keys], capsize=5)
    ax2.set_xticks(range(len(present_keys)))
    ax2.set_xticklabels(present_keys)
    ax2.set_ylabel("Mean Chamfer Distance")
    ax2.set_title("Mean CD with 95% Bootstrap CI")
    ax2.grid(axis="y", alpha=0.3)
    for i, (m, std) in enumerate(zip(means, stds)):
        ax2.annotate(f"{m:.4f}", (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax3 = axes[2]
    baseline_key = "none"
    if baseline_key in groups and len(present_keys) > 1:
        d_vals = []
        d_labels = []
        for k in present_keys:
            if k == baseline_key:
                continue
            d_vals.append(cohens_d(groups[k], groups[baseline_key]))
            d_labels.append(k)
        bars_d = ax3.barh(d_labels, d_vals, color=[colors.get(k, "#CCCCCC") for k in d_labels])
        ax3.axvline(0, color="black", linewidth=0.8)
        ax3.axvline(0.2, color="gray", linestyle="--", alpha=0.5)
        ax3.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax3.axvline(0.8, color="gray", linestyle="--", alpha=0.5)
        for bar, d in zip(bars_d, d_vals):
            ax3.text(bar.get_width() + (0.02 if d >= 0 else -0.02),
                     bar.get_y() + bar.get_height() / 2,
                     f"{d:+.3f}", va="center", fontsize=9)
        ax3.set_xlabel("Cohen's d (vs No IRS)")
        ax3.set_title("Effect Size")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "statistical_analysis.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Save] figure saved to {fig_path}")


def main(args):
    results = load_results(args.input)
    groups = group_by_exp_key(results)
    print(f"Loaded {len(results)} experiment entries, {sum(len(v) for v in groups.values())} total CD samples")
    for k, v in sorted(groups.items()):
        print(f"  {k}: {len(v)} samples, mean_CD={np.mean(v):.6f}")

    tests, effect_sizes = run_statistical_tests(groups, baseline_key="none")
    generate_report(groups, tests, effect_sizes, args.output_dir)
    generate_figures(groups, tests, effect_sizes, args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis for experiment results")
    parser.add_argument("--input", type=str, required=True, help="Path to experiment_results.json")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Output directory")
    args = parser.parse_args()
    main(args)