import os
import sys
import json
import argparse
import subprocess
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup
from setup import precompute_channels
from data_no_irs import ROILDMDataset, simulate_received_signal
from models import PointVAE, AdvancedCondEncoder, LatentDiT_Token_CrossAttn
from phase_optimizer import compute_received_signal_irs, PhaseOptimizer
from train_unified import (
    set_all_seeds, DDPMScheduler, CachedLatentDataset,
    rollout_physical_sequence, sample_one_ldm,
)


EXPERIMENT_MATRIX = {
    "none":      {"desc": "No IRS (Baseline)",         "irs_mode": "none"},
    "zero":      {"desc": "IRS + Zero Phase",          "irs_mode": "zero"},
    "random":    {"desc": "IRS + Random Phase",         "irs_mode": "random"},
    "optimized": {"desc": "IRS + Optimized Phase",      "irs_mode": "optimized"},
}

SEEDS = [42, 123, 456, 789, 1024]


def chamfer_distance(pc1, pc2):
    pc1 = torch.from_numpy(pc1).float()
    pc2 = torch.from_numpy(pc2).float()
    d1 = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0).min(dim=1)[0].mean()
    d2 = torch.cdist(pc2.unsqueeze(0), pc1.unsqueeze(0)).squeeze(0).min(dim=1)[0].mean()
    return (d1 + d2).item()


def load_best_ckpt(check_dir, device):
    best_path = os.path.join(check_dir, "best.pth")
    if not os.path.exists(best_path):
        return None
    ckpt = torch.load(best_path, map_location=device)
    return ckpt


def infer_cd_for_experiment(run_dir, device="cuda"):
    check_dir = os.path.join(run_dir, "check")
    ckpt = load_best_ckpt(check_dir, device)
    if ckpt is None:
        print(f"  [WARN] no best.pth in {check_dir}")
        return None

    config_path = os.path.join(run_dir, "config", "train_config.txt")
    irs_mode = "none"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if line.startswith("irs_mode:"):
                    irs_mode = line.split(":", 1)[1].strip()

    H_dict = precompute_channels(device=device)

    vae = PointVAE(num_points=2048, hidden_dim=512, token_dim=32, num_latent_tokens=32).to(device)
    vae_ckpt_paths = [
        os.path.join(run_dir, "vae_model", "vae_best.pth"),
        os.path.join(os.path.dirname(run_dir), "..", "outputs_test_none", "vae_model", "vae_best.pth"),
    ]
    vae_loaded = False
    for path in vae_ckpt_paths:
        if os.path.exists(path):
            ckpt_vae = torch.load(path, map_location=device)
            vae.load_state_dict(ckpt_vae)
            vae_loaded = True
            break
    if not vae_loaded:
        ckpt_vae_path = os.path.join(check_dir, "..", "vae_model", "vae_best.pth")
        if os.path.exists(ckpt_vae_path):
            ckpt_vae = torch.load(ckpt_vae_path, map_location=device)
            vae.load_state_dict(ckpt_vae)
            vae_loaded = True
    if not vae_loaded:
        print("  [WARN] VAE checkpoint not found, skipping CD eval")
        return None

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    cond_encoder = AdvancedCondEncoder(seq_len=setup.Tau, input_size=80, hidden_size=128, out_emb=256).to(device)
    eps_model = LatentDiT_Token_CrossAttn(token_dim=32, num_latent_tokens=32, cond_emb=256,
                                          hidden_size=256, depth=6, num_heads=8, max_cond_len=setup.Tau).to(device)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    eps_model.load_state_dict(ckpt["eps_model"])
    cond_encoder.eval()
    eps_model.eval()

    scheduler = DDPMScheduler(T=1000, device=device)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    phase_optimizer = None
    if irs_mode == "optimized":
        phase_optimizer = PhaseOptimizer(H_dict, device=device)

    total_samples = min(100, 7000)
    dataset_raw = ROILDMDataset(n_samples=total_samples, H_dict=H_dict, num_points=2048, device="cpu")

    cd_list = []
    n_eval = min(10, len(dataset_raw))
    print(f"  [Eval] computing CD for {n_eval} samples (irs_mode={irs_mode})...")
    for sample_idx in range(n_eval):
        point_cloud, roi_raw_t, roi_occ_t, phase_init, X_fixed = dataset_raw[sample_idx]
        z_pred = sample_one_ldm(
            cond_encoder=cond_encoder, eps_model=eps_model, scheduler=scheduler,
            roi_occ_t=roi_occ_t, phase_init=phase_init, X_fixed=X_fixed,
            H_dict=H_dict, irs_mode=irs_mode, z_mean=z_mean, z_std=z_std,
            device=device, cfg_scale=2.0, phase_optimizer=phase_optimizer,
        )
        recon_pred, _ = vae.decode(z_pred.unsqueeze(0))
        pred_pc = recon_pred.squeeze(0).detach().cpu().numpy()
        gt_pc = point_cloud.detach().cpu().numpy()
        cd = chamfer_distance(gt_pc, pred_pc)
        cd_list.append(cd)
        print(f"    sample {sample_idx}: CD={cd:.6f}")

    mean_cd = float(np.mean(cd_list))
    std_cd = float(np.std(cd_list))
    print(f"  [Eval] mean_CD={mean_cd:.6f} ± {std_cd:.6f}")
    return {"cd_list": cd_list, "mean_cd": mean_cd, "std_cd": std_cd, "n_eval": n_eval}


def run_single_experiment(exp_key, seed, args):
    label = f"{exp_key}_seed{seed}"
    info = EXPERIMENT_MATRIX[exp_key]
    run_dir = os.path.join(args.output_dir, label)

    cmd = [
        sys.executable, "train_unified.py",
        "--irs_mode", info["irs_mode"],
        "--seed", str(seed),
        "--total_samples", str(args.total_samples),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--save_dir", run_dir,
    ]
    if args.vae_ckpt:
        cmd.extend(["--vae_ckpt", args.vae_ckpt])
    if args.occlusion:
        cmd.append("--occlusion")

    print(f"\n{'='*60}")
    print(f"[RUN] {label} -- {info['desc']}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] {label} returned code {result.returncode}")
        return {"exp_key": exp_key, "seed": seed, "status": "failed", "elapsed_sec": elapsed}

    cd_result = infer_cd_for_experiment(run_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    entry = {
        "exp_key": exp_key,
        "seed": seed,
        "status": "ok",
        "elapsed_sec": elapsed,
        "cd_eval": cd_result,
    }
    return entry


def main(args):
    print(f"Experiment Matrix: {len(EXPERIMENT_MATRIX)} modes × {len(SEEDS)} seeds = {len(EXPERIMENT_MATRIX) * len(SEEDS)} runs")
    print(f"Output dir: {args.output_dir}")

    results_path = os.path.join(args.output_dir, "experiment_results.json")
    results = {}
    if os.path.exists(results_path) and not args.force:
        with open(results_path, "r") as f:
            results = json.load(f)
        print(f"[Resume] loaded {len(results)} existing results from {results_path}")

    for seed in SEEDS:
        set_all_seeds(seed)
        for exp_key, info in EXPERIMENT_MATRIX.items():
            label = f"{exp_key}_seed{seed}"
            if label in results and results[label]["status"] == "ok":
                print(f"[SKIP] {label} already completed")
                continue
            entry = run_single_experiment(exp_key, seed, args)
            results[label] = entry
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] results updated ({len(results)} entries)")

    print(f"\n{'='*60}")
    print("Summary:")
    for label, entry in sorted(results.items()):
        status = entry["status"]
        cd_str = ""
        if entry.get("cd_eval") and entry["cd_eval"].get("mean_cd") is not None:
            cd_str = f" mean_CD={entry['cd_eval']['mean_cd']:.6f}"
        elapsed = entry.get("elapsed_sec", 0)
        print(f"  {label:30s} [{status}] {elapsed:.0f}s{cd_str}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch experiment runner for unified IRS training")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Root output directory")
    parser.add_argument("--total_samples", type=int, default=7000, help="Total samples per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs per run")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to pretrained VAE checkpoint")
    parser.add_argument("--occlusion", action="store_true", default=False, help="Apply BS occlusion")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-run all experiments")
    args = parser.parse_args()
    main(args)