"""run_stats_suite.py — 统计套件编排器（spec 2026-09-08 第二优先级）

原则：只编排、不重实现。各 suite 复用现有入口（import 或 subprocess），
按 seed 配对收集结果到 JSON，再汇总成 mean±std markdown。
全量运行（论文表格底稿）：
    python run_stats_suite.py --suite all --n_seeds 10
冒烟验证（本次已跑）：
    python run_stats_suite.py --suite all --n_seeds 2

Seeding 审计结论（Task 0，2026-09-08）：
  - train_sensing.py / baseline_classic.py / demo*.py 均有 --seed 且入口处
    manual_seed（torch/np/random 三流）；
  - SatROIDataset.__getitem__ 依赖全局 random/np 流（逐样本生成受 seed 控制）；
  - verify_tracking_rician.py: 莱斯扰动种子 = seed*100003 + t（帧独立），
    随机相位基线固定 torch 42（跨 seed 共用基线，属"相对固定基线"协议）；
  - make_roi_voxel(legacy) 在 seed 设置后调用 → ROI 物体随 seed 变化
    （同 seed 跨方法配对一致；跨 seed 物体不同，符合 Finding 4 的物体依赖前提）；
  - demo_mot / mot_data 未纳入本套件（非目标）。
"""
import argparse
import json
import os
import re
import random
import shutil
import subprocess
import sys

import numpy as np
import torch

OUT_DIR = "./sat_verify/stats"
CKPT_DIR = "./stats_ckpt"


def _seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _build_base(seed):
    import setup_sat as ss
    from data_sat import SatScenarioChannels
    _seed_all(seed)
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    return ss, SatScenarioChannels(frames, irs_mode="sat", device="cpu")


def _roi_x(channels, seed):
    from data_sat import _SIGNAL1
    from verify_tracking import make_roi_voxel
    _seed_all(seed)
    ROI = make_roi_voxel()
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:4], dtype=torch.complex64).view(4, 1)
    return ROI, X


def _dump(name, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"[stats] {path} ({len(rows)} rows)")
    return rows


def _run(cmd):
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r.stdout


# ---------------------------------------------------------------- suites
def suite_tracking(seeds):
    """free-space 无随机噪声源；协议与 TECH_REPORT 一致：几何/ROI/随机基线固定，
    只跑一次（seeds 参数保留仅为接口统一）。跨种子方差由 suite_rician 提供。"""
    from phase_optimizer_sat import compare_tracking
    _, ch = _build_base(seeds[0])
    ROI, X = _roi_x(ch, seeds[0])
    torch.manual_seed(42)  # 随机相位基线与 verify_tracking_rician 同协议
    res = compare_tracking(ch, ROI, X, device="cpu", n_iter=5)
    base = res["random"]["power"]
    rows = [{"seed": seeds[0], **{k: {"power": v["power"],
                                      "boost": 100 * (v["power"] / base - 1)}
                                 for k, v in res.items()}}]
    return _dump("tracking", rows)


def suite_rician(seeds, k_dbs):
    from phase_optimizer_sat import compare_tracking
    from verify_tracking_rician import perturb_channels
    rows = []
    _, base_ch = _build_base(seeds[0])
    ROI, X = _roi_x(base_ch, seeds[0])  # 几何与 ROI 固定；只有莱斯扰动按 seed 变化
    for s in seeds:
        for k_db in k_dbs:
            ch = perturb_channels(base_ch, k_db, seed=s)
            torch.manual_seed(42)  # 与 verify_tracking_rician 相同：固定随机基线
            res = compare_tracking(ch, ROI, X, device="cpu", n_iter=5)
            rand_p = res["random"]["power"]
            rows.append({"seed": s, "k_db": k_db,
                         **{f"K={k}": 100 * (res[f"track_K={k}"]["power"] / rand_p - 1)
                            for k in (1, 2, 4, 8)}})
    return _dump("rician", rows)


def suite_sensing(seeds):
    rows = []
    for s in seeds:
        ckpt = os.path.join(CKPT_DIR, f"s{s}")
        out = os.path.join(OUT_DIR, f"sensing_s{s}.json")
        _run([sys.executable, "train_sensing.py", "--wideband", "--seed", str(s),
              "--save_dir", ckpt, "--out_json", out])
        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))
        shutil.rmtree(ckpt, ignore_errors=True)
    return _dump("sensing", rows)


def suite_loop(seeds):
    rows = []
    for s in seeds:
        ckpt = os.path.join(CKPT_DIR, f"loop_s{s}")
        ck = os.path.join(ckpt, "sensing_best.pth")
        _run([sys.executable, "train_sensing.py", "--wideband", "--seed", str(s),
              "--save_dir", ckpt])
        out = os.path.join(OUT_DIR, f"loop_s{s}.json")
        _run([sys.executable, "demo.py", "--checkpoint", ck, "--seed", str(s),
              "--out_json", out])
        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))
        shutil.rmtree(ckpt, ignore_errors=True)
    return _dump("loop", rows)


def suite_multi(seeds):
    rows = []
    for s in seeds:
        ckpt = os.path.join(CKPT_DIR, f"multi_s{s}")
        ck = os.path.join(ckpt, "sensing_multi_best.pth")
        _run([sys.executable, "train_sensing_multi.py", "--wideband", "--seed", str(s),
              "--save_dir", ckpt, "--out_json",
              os.path.join(OUT_DIR, f"multi_train_s{s}.json")])
        out = os.path.join(OUT_DIR, f"multi_s{s}.json")
        _run([sys.executable, "demo_multi.py", "--checkpoint", ck, "--seed", str(s),
              "--out_json", out])
        with open(out, encoding="utf-8") as f:
            rows.append(json.load(f))
        shutil.rmtree(ckpt, ignore_errors=True)
    return _dump("multi", rows)


_UNSEEN = ["_template_building", "_template_tank", "_template_tower", "_template_cubesat"]


def suite_ood(seeds, n_per=3):
    """未见模板 OOD：4 个可插拔模板（训练时不可见）上的定位误差 + 5 个已见类别参照。"""
    import math
    from data_sat import (compute_range_profile, GROUND_TARGET_TEMPLATES,
                          _template_building, _template_tank, _template_tower,
                          _template_cubesat)
    from train_sensing import SensingMLP
    from scipy.ndimage import rotate as ndi_rotate

    ck = "./isac_demo/sensing_best.pth"
    if not os.path.exists(ck):
        raise RuntimeError("缺 ./isac_demo/sensing_best.pth（先跑 sensing suite 或 make demo）")
    state = torch.load(ck, map_location="cpu")
    model = SensingMLP(in_dim=state["feat_dim"])
    model.load_state_dict(state["model"])
    model.eval()

    _, ch = _build_base(seeds[0])
    mid = ch.frames[len(ch.frames) // 2]
    wl = ch.wavelength_m

    def make_sample(maker, rng):
        space = np.zeros((16, 16, 16), dtype=np.float32)
        obj = maker().astype(np.float32)
        x, y, z = rng.randint(0, 8), rng.randint(0, 8), rng.randint(0, 8)
        space[x:x + 8, y:y + 8, z:z + 8] = obj
        ang = rng.uniform(0.0, 360.0)
        if abs(ang) > 1e-6:
            space = ndi_rotate(space, ang, axes=(0, 1), reshape=False,
                               order=1, mode="constant", cval=0.0)
            space = (space > 0.5).astype(np.float32)
        return space

    rows = []
    unseen = {"building": _template_building, "tank": _template_tank,
              "tower": _template_tower, "cubesat": _template_cubesat}
    makers = ([("seen:" + n, m) for n, m in GROUND_TARGET_TEMPLATES]
              + [("unseen:" + n, m) for n, m in unseen.items()])
    for s in seeds:
        rng = np.random.RandomState(s)
        for name, maker in makers:
            errs = []
            for _ in range(n_per):
                roi = make_sample(maker, rng)
                occ = np.argwhere(roi > 0.5)
                pos_true = (occ.mean(axis=0) / 16.0 * 2.0 - 1.0)[:2]
                feat = compute_range_profile(roi, mid["target_pos"], mid["ground_pos"],
                                             wl, snr_db=20.0, seed=int(rng.randint(1e6)),
                                             align=False, center="roi",
                                             sat_ecef=mid["sat_pos"])
                with torch.no_grad():
                    _, pred = model(torch.from_numpy(feat).float().unsqueeze(0))
                errs.append(float(np.linalg.norm(pred.numpy().reshape(-1)[:2] - pos_true)))
            rows.append({"seed": s, "template": name,
                         "pos_err_mean": float(np.mean(errs)),
                         "pos_err_std": float(np.std(errs)), "n": len(errs)})
    return _dump("ood", rows)


_BASE_PAT = [
    ("cfar_det", r"2D-CFAR \] 检测率\s*:\s*([0-9.]+)"),
    ("cfar_los", r"沿视线定位 RMSE\s*:\s*([0-9.]+) m"),
    ("music_mae", r"DOA MAE\s*:\s*([0-9.]+) deg"),
    ("ml_abs_acc", r"\[ML\(绝对\)\] 分类准确率\s*:\s*([0-9.]+)"),
    ("ml_abs_2d", r"\[ML\(绝对\)\] 2D 定位 RMSE\s*:\s*([0-9.]+) m"),
    ("ml_abs_los", r"\[ML\(绝对\)\] 沿视线 RMSE\s*:\s*([0-9.]+) m"),
    ("ml_abs_cross", r"横向 RMSE ([0-9.]+) m"),
    ("ml_old_acc", r"\[ML\(旧特\)\] 分类准确率\s*:\s*([0-9.]+)"),
    ("ml_old_2d", r"\[ML\(旧特\)\] 2D 定位 RMSE\s*:\s*([0-9.]+) m"),
    ("ml_shape_acc", r"\[ML\(形状\)\] 分类准确率\s*:\s*([0-9.]+)"),
    ("ml_shape_2d", r"\[ML\(形状\)\] 2D 定位 RMSE\s*:\s*([0-9.]+) m"),
]


def suite_baseline(seeds):
    rows = []
    for s in seeds:
        out = _run([sys.executable, "baseline_classic.py", "--seed", str(s)])
        row = {"seed": s}
        for key, pat in _BASE_PAT:
            m = re.search(pat, out)
            row[key] = float(m.group(1)) if m else None
        rows.append(row)
    return _dump("baseline", rows)


# ---------------------------------------------------------------- summary
def summarize():
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".json")
                   and not f.startswith(("sensing_s", "loop_s", "multi_", "tracking_")))
    lines = ["# Stats suite summary\n"]

    def fmt(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})"

    def load(name):
        p = os.path.join(OUT_DIR, name)
        return json.load(open(p, encoding="utf-8"))["rows"] if os.path.exists(p) else []

    tr = load("tracking.json")
    if tr:
        lines.append("## RIS tracking (boost % vs random)\n")
        for k in ("track_K=1", "track_K=2", "track_K=4", "track_K=8"):
            lines.append(f"- {k}: {fmt([r[k]['boost'] for r in tr])}")
        lines.append("")
    ri = load("rician.json")
    if ri:
        lines.append("## Rician K-sweep (boost %)\n")
        for k in sorted({r["k_db"] for r in ri}):
            sub = [r for r in ri if r["k_db"] == k]
            lines.append(f"- K={k:.0f} dB: " + " | ".join(
                f"{c}: {fmt([r[c] for r in sub])}" for c in ("K=1", "K=2", "K=4", "K=8")))
        lines.append("")
    se = load("sensing.json")
    if se:
        lines.append("## Sensing (5-class wideband HRRP)\n")
        lines.append(f"- best cls acc: {fmt([r['best_cls_acc'] for r in se])}")
        lines.append(f"- pos err (final): {fmt([r['pos_err_final'] for r in se])}")
        lines.append("")
    lo = load("loop.json")
    if lo:
        lines.append("## Closed loop (single)\n")
        for c in ("p_rand", "p_sensed", "p_oracle"):
            lines.append(f"- {c}: {fmt([r[c] for r in lo])}")
        lines.append(f"- efficiency: {fmt([100 * r['p_sensed'] / r['p_oracle'] for r in lo])}")
        lines.append(f"- cls ok rate: {fmt([1.0 if r['cls_ok'] else 0.0 for r in lo])}")
        lines.append("")
    mu = load("multi.json")
    if mu:
        lines.append("## Closed loop (multi)\n")
        for c in ("p_rand", "p_sensed", "p_oracle"):
            lines.append(f"- {c}: {fmt([r[c] for r in mu])}")
        lines.append(f"- efficiency: {fmt([100 * r['p_sensed'] / r['p_oracle'] for r in mu])}")
        lines.append(f"- detection (x/{mu[0]['n_targets']}): {fmt([r['det'] for r in mu])}")
        lines.append("")
    od = load("ood.json")
    if od:
        lines.append("## OOD localization pos-err (normalized)\n")
        tmpl = sorted({r["template"] for r in od})
        for t in tmpl:
            sub = [r for r in od if r["template"] == t]
            lines.append(f"- {t}: {fmt([r['pos_err_mean'] for r in sub])}")
        lines.append("")
    ba = load("baseline.json")
    if ba:
        lines.append("## Baselines (CFAR/ML)\n")
        for c in ("cfar_det", "cfar_los", "ml_abs_acc", "ml_abs_2d", "ml_abs_los",
                  "ml_abs_cross", "ml_old_acc", "ml_old_2d", "ml_shape_acc", "ml_shape_2d"):
            lines.append(f"- {c}: {fmt([r[c] for r in ba])}")
        lines.append("")
    out = os.path.join(OUT_DIR, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[stats] {out}")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="all",
                   choices=["all", "tracking", "rician", "sensing", "loop",
                            "multi", "ood", "baseline", "summary"])
    p.add_argument("--n_seeds", type=int, default=2, help="冒烟默认 2；全量用 10")
    p.add_argument("--k_dbs", nargs="+", type=float, default=[10.0, 5.0, 0.0])
    args = p.parse_args()

    seeds = list(range(42, 42 + args.n_seeds))
    suites = {"tracking": suite_tracking, "rician": suite_rician,
              "sensing": suite_sensing, "loop": suite_loop,
              "multi": suite_multi, "ood": suite_ood, "baseline": suite_baseline}
    if args.suite == "summary":
        summarize()
        return
    todo = list(suites) if args.suite == "all" else [args.suite]
    for name in todo:
        print(f"\n===== suite: {name} (seeds {seeds}) =====")
        if name == "rician":
            suites[name](seeds, args.k_dbs)
        else:
            suites[name](seeds)
    summarize()


if __name__ == "__main__":
    main()
