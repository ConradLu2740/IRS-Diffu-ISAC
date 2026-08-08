# RIS-Aided Integrated Sensing and Communication Toward Space ISAC: A Physics-Grounded Open-Source Engineering System

**Conrad Lu** (conrad.lu.2740@gmail.com)

*School of Information Science and Engineering, Northeastern University, Shenyang, China*

**Version**: v1.1 (2026-08-08) — companion to the open-source repository
[https://github.com/ConradLu2740/IRS-Diffu-ISAC](https://github.com/ConradLu2740/IRS-Diffu-ISAC)

---

## Abstract

This report describes an open-source, physics-grounded engineering system for RIS-aided Integrated Sensing and Communication (ISAC) extended to space ISAC (ISAC-NTN). It combines real LEO orbit propagation (SGP4), dynamic RIS phase tracking, learning-based sensing, 3D multi-object tracking, and a sensing–communication closed loop, all reproducible with one-command scripts. Key results: orbit physics matches real ISS values; RIS frame-by-frame tracking improves power by **+89%** (K=1), while reconfiguration-limited tracking (K=8) loses the gain; a sensing-aided closed loop achieves **+309%** communication gain (97.6% of the ideal oracle); wideband HRRP classification reaches **0.80** (9-class); 3D multi-object tracking of 10 targets achieves 0.60 detection recall. A systematic comparison with classical baselines (2D-CFAR, MUSIC) on identical test sets uncovers two transferable findings: (i) a feature-construction defect — centroid-relative range-profile delays silently discard absolute target position, collapsing ML localization to a class prior (2D RMSE 22.6 m vs 12.1 m with absolute-range features); (ii) a far-field angle-resolution wall — at ~695 km slant range, the 80 m ROI subtends only 0.0066°, far below an 8-element ULA resolution (~14°), so mono-static angle-based cross-range localization is physically unavailable. All numbers are reproducible from the repository with fixed seeds.

**Keywords**: ISAC, RIS, non-terrestrial networks, LEO satellite, diffusion models, CFAR, MUSIC

---

## 1. Introduction

### 1.1 Background and Motivation

Integrated Sensing and Communication (ISAC) unifies wireless communication and radar-like sensing in a single system and is a key enabler of 6G [1, 2]. Reconfigurable Intelligent Surfaces (RIS) extend coverage and enhance both communication and sensing at low hardware cost [3, 4]. Non-terrestrial networks (NTN), in particular LEO constellations, are being integrated into 5G-Advanced/6G, and combining ISAC with NTN ("space ISAC") is of growing interest for space situational awareness and terrestrial monitoring [5, 6, 7].

Most published work remains at the level of analytical studies or link-level simulations with simplified geometry. Mature open-source link/PHY platforms (e.g., Sionna [8], MATLAB 5G/NTN toolboxes) exist but do not combine real orbit dynamics, dynamic RIS optimization, learning-based sensing, and closed-loop demonstration in one system. This report accompanies an engineering system that does; it makes no claim of algorithmic novelty over any single component. Its contributions:

1. **A physics-grounded, reproducible space ISAC pipeline** from real TLE orbit data to closed-loop demo, with one-command verification and CI.
2. **Dynamic RIS phase tracking** for LEO overpasses with an explicit quantification of the "reconfiguration rate vs channel coherence time" trade-off.
3. **A learning-based sensing layer** (classification + localization on wideband HRRP features) and **3D multi-object tracking**.
4. **A systematic classical-baseline comparison** (2D-CFAR + MUSIC vs ML) on identical test sets, surfacing a feature-construction defect and a quantified far-field angle wall.
5. **An SDR data interface** (IQ format + ingest pipeline) easing the transition to hardware.

### 1.2 Report Organization

Section 2: system/signal models. Section 3: dynamic RIS tracking and the closed loop. Section 4: wideband sensing and 3D MOT. Section 5: classical baselines and the two findings. Section 6: experiments and reproducibility. Section 7: limitations.

---

## 2. System and Signal Model

### 2.1 Scenario Geometry

A LEO satellite acts as the base station (BS), performs mono-static sensing of a ground region of interest (ROI), and serves a ground user equipment (UE) station. An RIS panel can be **spaceborne** (~10 m) or **ground** (~1 m, near UE); modes `none`/`sat`/`ground` are compared. Real TLE ephemerides (ISS NORAD 25544, Starlink) are propagated with SGP4 [9]; ECI→ECEF conversion includes GMST and Earth-rotation velocity corrections. The overpass window is searched over 48 h; frames are sampled at 1 s near window center where elevation > 20°.

Defaults: 30 GHz carrier (λ = 1 cm); ISS altitude ~420 km; ROI 80 m × 80 m at (30°N, 120°E); UE at (30°N, 119.5°E); BS–ROI slant ≈ 695 km; Doppler −611…+611 kHz; SNR 20 dB.

**Physics verification** (`verify_sat.py`): ISS altitude 418.3 km, velocity 7.66 km/s, period 92.9 min, Doppler S-curve, channel equations — consistent with published values.

### 2.2 Signal Model

Five propagation paths: (1) BS→ROI direct illumination, (2) ROI→UE direct scatter, (3) BS→UE direct leakage, (4) BS→RIS→ROI reflected illumination, (5a) BS→RIS→UE and (5b) ROI→RIS→UE forwarded paths. For voxel scatterers with complex amplitudes a_i and round-trip delays τ_i, the baseband frequency response is

H(f) = Σ_i a_i exp(−j2π f τ_i),

and the wideband range profile (HRRP) is the inverse FFT of H(f) over K = 512 subcarriers at 1 GHz bandwidth (range resolution ≈ 0.15 m), plus AWGN. The ISAR sequence extends this over M = 32 frames while the target rotates, synthesizing cross-range resolution. Two delay conventions are used:

- **centroid-relative** (`center='centroid'`): delays relative to the voxel centroid — shape/pose features, position-independent (classic HRRP);
- **ROI-center-relative** (`center='roi'`): delays retain absolute target position within the ROI (needed for localization); a differential-delay formulation (τ = 1.1·d_proj/c, combining the 0.1 BS-side and 1.0 UE-side projection coefficients) avoids K-bin wrap-around.

As shown in Section 5, using the centroid-relative convention for localization silently discards position information.

### 2.3 RIS Model and Phase Optimization

The RIS has N unit-modulus phase elements. Phase-aligned configuration maximizes coherent combination of RIS-assisted and direct paths at the UE. Because the satellite moves at ~7.5 km/s, the optimal phase pattern changes over the channel coherence time. We compare:

- **Frame-by-frame tracking** (K=1): recompute phases every frame — power **+89.0%** vs random (seed-fixed, reproducible);
- **Segmented tracking** (K=2/4/8): reconfiguration limited to every K-th frame — K=2: +60.0%, K=4: +36.6%, K=8: **−41.5%** (stale phases can even hurt). This quantitatively illustrates the reconfiguration-rate vs coherence-time trade-off.

### 2.4 Target Models

Ground targets are 16³ voxel templates (5 m voxels; 80 m ROI) with 9 classes: car, UAV, building, tank, tower, cubesat, bicycle, pedestrian, train. Each sample is placed at a random position with a random pose (rotation about the vertical axis), modulating the scattering-center distribution. Isotropic scattering is assumed (no RCS angular dependence, polarization, or occlusion modeled).

---

## 3. Dynamic RIS Tracking and the Sensing–Communication Closed Loop

### 3.1 Sensing Layer

- **Diffusion-based 3D reconstruction** (`train_sat.py`): conditional latent diffusion (PointVAE encoder + DiT-style denoiser + CFG) reconstructs the ROI point cloud; evaluated with Chamfer Distance (CD), F-Score, voxel IoU. Training-scale results: CD 0.137 (sat) / 0.169 (ground) vs 0.233 (no RIS).
- **Classification + localization** (`train_sensing*.py`): MLP over wideband HRRP (K = 512) outputs 9-class labels and normalized 2D position within the ROI; CPU real-time.

### 3.2 Communication Layer and Closed Loop

Given sensing output, the IRS phase pattern is configured toward the sensed target. Received power comparison (reproducible, seed-fixed): random phase 1.00×, sensing-aided **+309.4%**, ideal oracle +319.3% (closed-loop efficiency **97.6%**). Multi-target closed loop: detection 1/2 in the current 5-class setting, IRS pointing gain **+443.8%** (93% of oracle). A robustness observation: even when classification is wrong, coarse localization captures most of the communication gain.

### 3.3 Closed-Loop Demo

The pipeline is packaged as `run_demo.sh` plus a single-file HTML player and GIF animation; a Colab notebook gives a 60-second zero-setup experience.

---

## 4. Wideband/ISAR Sensing and 3D Multi-Object Tracking

### 4.1 Narrowband → Wideband → ISAR

Classification accuracy progression (early experiments, 6-class templates; reported for reference): narrowband 0.383 → wideband HRRP 0.867 → ISAR sequence 0.933. With the current 9-class templates, wideband HRRP classification is **0.80** (reproducible, `train_sensing.py --wideband`, seed-fixed). The progression quantifies the information added by range resolution and by synthetic aperture over rotation.

### 4.2 3D Multi-Object Tracking (MOT)

- **Scene data** (`mot_data.py`): 10 moving targets of 5 classes (car, drone, bicycle, pedestrian, train), trajectories straight/accelerating/turning, rendered into per-frame 5-path channel data and range profiles with Doppler.
- **Detector** (`train_detect.py`): CNN over range profiles → (class, position), CPU-trainable.
- **Tracker** (`mot_tracker.py`): Hungarian association with constant-velocity prediction, ID maintenance, class majority voting, α-β smoothing.
- **3D output** (`demo_mot_html.py`): interactive Plotly HTML with full 3D trajectories.

Reproducible results (seed-fixed): detection recall **0.60** (10 targets), trajectory class accuracy **0.73**. A physical constraint is enforced: ground targets have z locked to 0 (mono-static range profiles carry weak height information); only aerial targets (drone) have free z.

### 4.3 SDR Data Interface

An IQ format (`sdr_io.py`) and ingest pipeline (`sdr_ingest.py`) convert time-domain IQ via FFT to range profiles (fidelity **0.998** vs simulation reference) and feed the sensing layer. Hardware-ready (RTL-SDR/USRP) without changing downstream processing.

---

## 5. Classical Baselines and Physical/Engineering Findings

Classical radar baselines (`baseline_classic.py`) evaluated on the same fixed test set (60 samples, SNR 20 dB, seed-fixed):

- **2D-CA-CFAR** (P_fa = 10⁻⁴, convolution-vectorized) on the range–Doppler map (slow-time FFT of the ISAR sequence);
- **1D-CFAR** on the absolute-range profile with regression-calibrated bin→meters mapping (20-sample calibration set, R² ≈ 0.83);
- **MUSIC** with an 8-element ULA (λ/2, 64 snapshots). *Honesty note: MUSIC is validated with synthetically generated point-source snapshots (a·s+n) — it does not share the same signal stream as CFAR/ML; the 0.017° figure verifies algorithmic self-consistency, not end-to-end sensing accuracy.*

### 5.1 Results

| Method | Detection | LOS RMSE | Cross-range RMSE | Class acc. |
|---|---|---|---|---|
| 2D-CFAR (detection) | **1.000** | — | — | — |
| 1D-CFAR (localization) | — | **8.14 m** | — (no angle info) | — |
| MUSIC (ULA-8, synthetic) | — | — | — | DOA MAE **0.017°** |
| ML (absolute-range) | — | **2.27 m** | 11.84 m | 0.733 |
| ML (centroid-relative, old) | — | — | 2D RMSE **22.63 m** | 0.817 |
| ML (shape, aligned) | — | — | 2D RMSE 21.86 m | 0.700 |

### 5.2 Finding 1: Feature-Construction Defect

The original range-profile function computed delays relative to the voxel centroid, silently discarding the target's absolute position. A controlled single-voxel experiment confirmed this: moving the target across the ROI left the range-profile centroid bin *constant*. Consequently, ML localization trained on centroid-relative features collapses to a class-prior estimate: 2D RMSE **22.63 m** (centroid-relative) and 21.86 m (aligned shape features) vs **12.06 m** with absolute-range (`center='roi'`) features — a ~2× gap.

**Fix and verification**: `compute_range_profile` now supports `center='roi'` (differential delay); `SatROIDataset` uses it when `rp_align=False`. After retraining: classification 0.80, 2D localization RMSE 12.1 m (LOS 2.3 m), and the closed-loop demo is unaffected (+309.4%, 97.6% of oracle). Physics verification and module imports remain green.

### 5.3 Finding 2: The Far-Field Angle-Resolution Wall

At ~695 km slant range, 1 m of cross-range offset subtends ≈ 8×10⁻⁵ degrees; the full 80 m ROI subtends ≈ 0.0066°. An 8-element ULA at λ/2 has a Rayleigh resolution of ≈ 0.886·λ/(Nd) ≈ **12.7°** (upper-bound estimate; even a finer reading λ/D ≈ 14.3° is orders of magnitude larger). Therefore mono-static angle information cannot localize targets within the ROI: ML cross-range RMSE ≈ 11.8 m reflects exactly this wall (its cross-range output is driven by class priors and training statistics, not observable angles). This is a physical geometry bound, not an implementation artifact; improvements require multi-static geometry, long-aperture interferometric/ISAR imaging, or temporal priors (tracking).

---

## 6. Experiments and Reproducibility

### 6.1 Summary of Quantitative Results (all reproducible, seed-fixed 42)

| # | Experiment | Result |
|---|---|---|
| 1 | Orbit physics (ISS) | Altitude 418.3 km / 7.66 km/s / 92.9 min — matches real values |
| 2 | Overpass Doppler (30 GHz) | −611…+611 kHz S-curve (real LEO order) |
| 3 | RIS frame-by-frame tracking (K=1) | Power **+89.0%** vs random phase |
| 4 | RIS segmented tracking | K=2: +60.0% · K=4: +36.6% · K=8: **−41.5%** (stale phases harmful) |
| 5 | Sensing–comm closed loop (single) | Class 80%, comm gain **+309.4%** (97.6% of oracle) |
| 6 | Sensing–comm closed loop (multi) | Detection 1/2, IRS gain **+443.8%** (93% of oracle) |
| 7 | Classification (9-class, wideband HRRP) | **0.80** (early 6-class: 0.383→0.867→ISAR 0.933) |
| 8 | 3D MOT | 10 targets / 5 classes, recall **0.60**, class acc. 0.73 |
| 9 | SDR pipeline | IQ→FFT→range profile fidelity **0.998** |
| 10 | Robustness | ISS / Starlink × 30 GHz / 28 GHz: all PASS |
| 11 | 2D-CFAR / 1D-CFAR | Detection 100% (P_fa=10⁻⁴), LOS RMSE **8.14 m** |
| 12 | MUSIC (synthetic) | ULA-8 DOA MAE **0.017°**; far-field angle wall quantified |
| 13 | Feature fix | Localization 2D RMSE 22.63 → **12.06 m**; LOS 2.3 m |

### 6.2 Reproducibility

All results are produced by one-command scripts with **fixed global seeds** (`torch.manual_seed` + `np.random.seed` + `random.seed(42)` at every entry point; verified: two consecutive runs produce identical outputs):

```bash
cd source_code/isac_sat
../../.venv/bin/python verify_sat.py            # physics verification (ALL PASS)
../../.venv/bin/python verify_tracking.py       # RIS tracking trade-off
../../.venv/bin/python train_sensing.py --wideband  # sensing (class + localization)
../../.venv/bin/python baseline_classic.py      # CFAR + MUSIC vs ML comparison
../../.venv/bin/python demo.py --checkpoint ./isac_demo/sensing_best.pth  # closed loop
../../.venv/bin/python demo_mot.py              # 3D multi-object tracking
```

A GitHub Actions CI pipeline runs import checks, physics smoke tests, and SDR fidelity on every push. A Colab notebook reproduces the core demo in ~60 s.

---

## 7. Limitations and Honest Discussion

1. **Absolute attitude estimation is not feasible** in the far-field star–ground setting with simple symmetric templates — a physical upper bound, not an implementation gap.
2. **Single-station multi-target classification is limited** by signal mixing in range profiles (detection/localization remain usable).
3. **Cross-range localization is angle-limited** (Section 5.3): mono-static angle-based cross-range localization is physically unavailable at practical array sizes for ~695 km links.
4. **Target templates are simple voxel models** (isotropic scattering; no RCS angular dependence/polarization); space-debris/satellite geometry models are planned.
5. **No over-the-air hardware validation yet**: the SDR interface is verified on simulated IQ; RTL-SDR/USRP capture is the natural next step.
6. **Atmosphere/ionosphere effects are not modeled** (free-space far-field approximation).
7. **Evaluation scale is modest** (60–150 test samples, smoke-level training, single seed per experiment; multiple seeds and larger runs are a matter of compute). MOT is evaluated on a single scene.
8. **Historical values**: some early results (6-class classification progression) were produced before the current 9-class templates; they are reported for reference and are reproducible only from earlier commits.

---

## 8. Conclusion

We presented an open-source, physics-grounded space ISAC engineering system — real LEO orbits, dynamic RIS tracking, learning-based sensing, 3D MOT, closed loop, SDR interface — with fixed-seed reproducibility and CI verification. Two transferable findings emerge: (i) a feature-construction defect (centroid-relative delays discard absolute position) that we fixed and quantified (~2× localization gap); (ii) a quantitative far-field angle-resolution wall bounding mono-static cross-range localization. The system serves as a practical testbed for space ISAC research; the honest limitation reporting aims to raise the reproducibility bar in this emerging area.

---

## Acknowledgments

School research project developed with AI tooling assistance (Proma agent). The repository is maintained at [https://github.com/ConradLu2740/IRS-Diffu-ISAC](https://github.com/ConradLu2740/IRS-Diffu-ISAC) under the MIT license. An independent adversarial review pass (AI reviewer) identified and helped fix reproducibility issues in v1.0.

---

## References

1. F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, "Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond," *IEEE J. Sel. Areas Commun.*, vol. 40, no. 6, pp. 1728–1767, 2022.
2. A. Zhang, M. L. Rahman, X. Huang, Y. J. Guo, S. Chen, and R. W. Heath, "Perceptive mobile networks: Cellular networks with radio vision via joint communication and radar sensing," *IEEE Veh. Technol. Mag.*, vol. 16, no. 2, pp. 20–30, 2021.
3. Q. Wu and R. Zhang, "Intelligent reflecting surface enhanced wireless network via joint active and passive beamforming," *IEEE Trans. Wireless Commun.*, vol. 18, no. 11, pp. 5394–5409, 2019.
4. C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, "Reconfigurable intelligent surfaces for energy efficiency in wireless communication," *IEEE Trans. Wireless Commun.*, vol. 18, no. 8, pp. 4157–4170, 2019.
5. 3GPP, "Solutions for NR to support non-terrestrial networks (NTN)," TR 38.821, Release 16, 2020.
6. H. Wymeersch et al., "Integration of communication and sensing in 6G: A joint industrial and academic perspective," in *Proc. IEEE PIMRC*, 2021.
7. 3GPP, "Study on integrated sensing and communication," TR 22.837, Release 19, 2023.
8. J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, "Sionna: An open-source, GPU-accelerated library for simulation of wireless systems," in *Proc. IEEE SPAWC*, 2022.
9. F. R. Hoots and R. L. Roehrich, "Spacetrack report no. 3: Models for propagation of NORAD element sets," U.S. Air Force Aerospace Defense Command, 1980.
10. Y. Luo et al., "LION: Latent point diffusion models for 3D shape generation," *NeurIPS*, 2022.
11. Z. Lyu et al., "PVD: Point-voxel diffusion for 3D generative modeling," *ICCV*, 2021.
12. A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, "Point-E: A system for generating 3D point clouds from complex prompts," *arXiv:2212.08751*, 2022.
13. H. Rohling, "Radar CFAR thresholding in clutter and multiple-target situations," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-19, no. 4, pp. 608–621, 1983.
14. R. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Trans. Antennas Propag.*, vol. 34, no. 3, pp. 276–280, 1986.
15. M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
16. C. Z. Lu, "IRS-Diffu-ISAC: RIS-aided ISAC via diffusion models for 3D point cloud reconstruction," GitHub repository, 2026. [Online]. Available: https://github.com/ConradLu2740/IRS-Diffu-ISAC

---

*Report v1.1. All numbers are produced by the scripts in the companion repository with fixed seeds and are reproducible at the commit accompanying this version (2026-08-08).*
