# IRS-Diffu-ISAC

**IRS-Aided Integrated Sensing and Communication via Diffusion Models for 3D Point Cloud Reconstruction**

This project explores using Intelligent Reflecting Surfaces (IRS) and Conditional Latent Diffusion Models for 3D point cloud reconstruction in ISAC scenarios.

## Project Overview

- **Task**: Reconstruct 3D point clouds of objects from wireless signals
- **Key Technologies**: Diffusion Models, Point Cloud VAE, IRS (Intelligent Reflecting Surface)
- **Application**: Integrated Sensing and Communication (ISAC)

## Architecture

```
Stage 1: PointVAE Training
  PointCloud → [Encoder] → Latent z → [Decoder] → Reconstructed PointCloud

Stage 2: Latent Diffusion Model Training
  Physical Signals (Tx/Rx/IRS Phase) → [Condition Encoder] → Condition Embedding
  z₀ + Noise → [DDPM Forward] → zₜ → [DiT/UNet] → Predicted Noise

Inference
  Random Noise → [DDPM Reverse T steps] → z₀ → [VAE Decoder] → Point Cloud
```

## Project Structure

```
IRS_Diffu_ISAC/
├── source_code/
│   ├── setup.py              # Simulation parameters & channel precomputation
│   ├── data.py               # IRS scenario data generation
│   ├── data_no_irs.py        # No-IRS scenario data generation
│   ├── models.py              # Neural network models (PointVAE, DiT)
│   ├── models_unet.py        # U-Net architecture model
│   ├── train.py              # Main training script (IRS + random phase)
│   ├── train_no.py           # No-IRS training script
│   ├── train_r.py            # Random IRS training script
│   ├── train_unified.py      # Unified training framework
│   ├── phase_optimizer.py    # IRS phase optimization
│   ├── run_experiments.py    # Batch experiment scheduler
│   └── analyze_results.py    # Statistical analysis & visualization
├── architecture.md           # Architecture documentation
└── Code_Wiki.md             # Detailed code documentation
```

## Quick Start

### 1. Environment Setup

```bash
pip install torch numpy scipy matplotlib
```

### 2. Run Training

**IRS + Random Phase (Full two-stage training)**:
```bash
cd source_code
python train.py
```

**No-IRS Scenario**:
```bash
python train_no.py
```

**Random IRS Scenario**:
```bash
python train_r.py
```

### 3. Run Experiments

```bash
python run_experiments.py --total_samples 7000 --epochs 500
python analyze_results.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_points` | 2048 | Point cloud sampling points |
| `z_dim` | 256 | Latent space dimension |
| `Tau` | 8 | Time slots |
| `IRS_elements` | 32 | Total IRS elements (2 panels × 16) |
| `ROI_Length` | 16 | ROI grid size |

## Signal Model

The system considers 5 propagation paths:
1. BS → ROI → UE (direct scatter)
2. BS → ROI → IRS1 → UE (via IRS1)
3. BS → ROI → IRS2 → UE (via IRS2)
4. BS → IRS1 → ROI → UE (IRS1 forward scatter)
5. BS → IRS2 → ROI → UE (IRS2 forward scatter)

## Space ISAC Extension (Phase 1)

This project also includes a **Space ISAC** extension: LEO-satellite-aided integrated sensing and communication (ISAC-NTN) with dynamic geometry.

- `setup_sat.py` — SGP4 orbit propagation (real ISS/Starlink TLE), dynamic geometry, far-field channel with Doppler/delay
- `data_sat.py` — dynamic data generation for 3 IRS modes: `none` / `sat` (spaceborne RIS) / `ground`; ground target templates (car/uav/building/tank/tower/cubesat)
- `train_sat.py` — comparative training across modes (reuses PointVAE + CondEncoder + DiT)
- `eval_sat.py` — metrics: Chamfer Distance / F-Score / Voxel IoU + visualization
- `verify_sat.py` — physics validation (orbit, overpass dynamics, channel formulas)
- `phase_optimizer_sat.py` — dynamic RIS phase tracking (analytic alignment, segmented tracking)
- `verify_tracking.py` / `verify_robustness.py` — phase tracking tradeoff & orbit/frequency robustness

See [space_isac_design.md](space_isac_design.md) for full design details.

## Citation

If you use this code, please cite:

```
IRS-Diffu-ISAC: IRS-Aided ISAC via Diffusion Models for 3D Point Cloud Reconstruction
```

## License

MIT License
