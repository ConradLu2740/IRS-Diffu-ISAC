import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup
from setup import precompute_channels
from data_no_irs import ROILDMDataset
from models import PointVAE, AdvancedCondEncoder, LatentDiT_Token_CrossAttn
from phase_optimizer import PhaseOptimizer
from train_unified import (
    set_all_seeds, DDPMScheduler, sample_one_ldm,
)


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


def infer_cd_for_run(run_dir, vae_ckpt_path, device="cuda"):
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
    if os.path.exists(vae_ckpt_path):
        ckpt_vae = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(ckpt_vae)
    else:
        print(f"  [WARN] VAE checkpoint not found: {vae_ckpt_path}")
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

    total_samples = 100
    dataset_raw = ROILDMDataset(
        n_samples=total_samples,
        H_dict=H_dict,
        num_points=2048,
        device="cpu",
        irs_mode=irs_mode,
        phase_optimizer=phase_optimizer,
        save_dir=run_dir,
    )

    cd_list = []
    n_eval = 10
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    runs = {
        "none_seed42": {
            "run_dir": os.path.join(base_dir, "outputs_test_none3"),
            "vae_ckpt": os.path.join(base_dir, "outputs_test_none3", "vae_model", "vae_best.pth"),
            "exp_key": "none",
            "seed": 42,
        },
        "zero_seed42": {
            "run_dir": os.path.join(base_dir, "outputs_test_zero2"),
            "vae_ckpt": os.path.join(base_dir, "outputs_test_none3", "vae_model", "vae_best.pth"),
            "exp_key": "zero",
            "seed": 42,
        },
        "random_seed42": {
            "run_dir": os.path.join(base_dir, "outputs_test_random"),
            "vae_ckpt": os.path.join(base_dir, "outputs_test_none3", "vae_model", "vae_best.pth"),
            "exp_key": "random",
            "seed": 42,
        },
        "optimized_seed42": {
            "run_dir": os.path.join(base_dir, "outputs_test_optimized_precompute"),
            "vae_ckpt": os.path.join(base_dir, "outputs_test_none3", "vae_model", "vae_best.pth"),
            "exp_key": "optimized",
            "seed": 42,
        },
    }

    results = {}
    for label, info in runs.items():
        print(f"\n{'='*60}")
        print(f"[RUN] {label}")
        print(f"{'='*60}")

        if not os.path.exists(info["run_dir"]):
            print(f"  [SKIP] run_dir not found: {info['run_dir']}")
            continue

        t0 = time.time()
        cd_result = infer_cd_for_run(info["run_dir"], info["vae_ckpt"], device=device)
        elapsed = time.time() - t0

        entry = {
            "exp_key": info["exp_key"],
            "seed": info["seed"],
            "status": "ok" if cd_result is not None else "failed",
            "elapsed_sec": elapsed,
            "cd_eval": cd_result,
        }
        results[label] = entry

    output_path = os.path.join(base_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] results saved to {output_path}")

    print(f"\n{'='*60}")
    print("Summary:")
    for label, entry in sorted(results.items()):
        status = entry["status"]
        cd_str = ""
        if entry.get("cd_eval") and entry["cd_eval"].get("mean_cd") is not None:
            cd_str = f" mean_CD={entry['cd_eval']['mean_cd']:.6f}"
        elapsed = entry.get("elapsed_sec", 0)
        print(f"  {label:30s} [{status}] {elapsed:.0f}s{cd_str}")


if __name__ == "__main__":
    import time
    main()
