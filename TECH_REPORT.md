# RIS-Aided Integrated Sensing and Communication Toward Space ISAC: A Physics-Grounded Open-Source Engineering System

**Conrad Lu** (conrad.lu.2740@gmail.com)

*School of Information Science and Engineering, Northeastern University, Shenyang, China*

**Version**: v1.0 (2026-08-08) — companion to the open-source repository
[https://github.com/ConradLu2740/IRS-Diffu-ISAC](https://github.com/ConradLu2740/IRS-Diffu-ISAC)

---

## Abstract

Integrated Sensing and Communication (ISAC) is a cornerstone of 6G networks, and reconfigurable intelligent surfaces (RIS) offer a promising means to enhance both communication and sensing coverage. This report describes an open-source, physics-grounded engineering system that extends RIS-aided ISAC from indoor near-range simulation to *space ISAC* (ISAC-NTN): real LEO satellite orbits propagated with SGP4, dynamic RIS phase tracking, a sensing–communication closed loop, wideband HRRP/ISAR sensing, 3D multi-object tracking, and an SDR data interface. All experiments are reproducible with one-command scripts and CI-verified. Key quantitative results include: orbit physics verified against real ISS values (altitude 418 km, velocity 7.66 km/s); dynamic RIS tracking improving received power by **+283%**; a sensing-aided closed loop achieving **+233%** communication gain (98% of the ideal oracle); wideband HRRP classification of **0.867** (narrowband 0.383 → wideband 0.867 → ISAR sequence 0.933); 3D multi-object tracking of 10 targets across 5 classes with 81% detection recall. We further report a systematic comparison with classical radar baselines — 2D-CFAR detection/localization and MUSIC direction finding — on identical test sets. This comparison uncovers a *feature-construction defect* in the original range-profile pipeline (centroid-relative projection silently discards absolute target position) and quantifies the *far-field angle-resolution wall*: at ~700 km slant range, 1 m of cross-range offset corresponds to 8×10⁻⁵ degrees, far below the resolution of an 8-element ULA (~22.5°). We fix the defect (absolute-range features, `center='roi'`) and report improved localization (2D MAE 20.4 m → 12.2 m). The system, findings, and reproducible code are intended as a practical reference for space ISAC research and engineering.

**Keywords**: ISAC, RIS, non-terrestrial networks, LEO satellite, diffusion models, 3D reconstruction, CFAR, MUSIC

---

## 1. Introduction

### 1.1 Background and Motivation

Integrated Sensing and Communication (ISAC) aims to unify wireless communication and radar-like sensing in a single spectrum- and hardware-efficient system, and is widely recognized as a key enabler of 6G [1, 2]. Reconfigurable Intelligent Surfaces (RIS) further extend coverage and enhance both communication links and sensing channels at low hardware cost [3, 4].

Non-terrestrial networks (NTN) — in particular LEO satellite constellations — are being integrated into 5G-Advanced/6G architectures to provide global connectivity. Combining ISAC with NTN ("space ISAC" / ISAC-NTN) has recently attracted attention for applications such as space situational awareness, terrestrial target monitoring, and integrated satellite communication [5, 6]. However, most published work remains at the level of analytical studies or link-level simulations with simplified geometry; few open-source implementations exist that combine *real orbit dynamics*, *dynamic RIS optimization*, *learning-based sensing*, and *end-to-end closed-loop demonstration* in a single reproducible system.

This report accompanies an open-source engineering system that bridges that gap. It is not a claim of algorithmic novelty over any single component — the components (SGP4 orbit propagation, HRRP/ISAR processing, CFAR/MUSIC, conditional diffusion models) are all established techniques. Its contributions are:

1. **A physics-grounded, fully reproducible space ISAC pipeline** from real TLE orbit data to sensing-communication closed-loop demo, with one-command verification scripts and CI.
2. **Dynamic RIS phase tracking** for LEO overpasses, including an explicit quantification of the "RIS reconfiguration rate vs channel coherence time" trade-off.
3. **A learning-based sensing layer** (classification + localization on wideband HRRP/ISAR features) and a **3D multi-object tracking** capability.
4. **A systematic classical-baseline comparison** (2D-CFAR + MUSIC vs. ML) on identical test sets, which surfaces two non-obvious findings: a feature-construction defect that silently discards absolute target position, and a quantitative characterization of the far-field angle-resolution wall for cross-range localization.
5. **An SDR data interface** (IQ format + ingest pipeline) to ease the transition from simulation to hardware.

### 1.2 Contributions of This Report

- Section 2: system and signal models (orbit geometry, 5-path channel, RIS model, target templates, wideband features).
- Section 3: dynamic RIS tracking and the sensing–communication closed loop.
- Section 4: wideband/ISAR sensing and 3D multi-object tracking.
- Section 5: classical baselines (2D-CFAR, MUSIC) and the physical/engineering findings.
- Section 6: experiments, results, and reproducibility.
- Section 7: limitations and honest discussion.

---

## 2. System and Signal Model

### 2.1 Scenario Geometry

We consider a **satellite-to-ground ISAC link**: a LEO satellite acts as the base station (BS) and performs mono-static sensing of a ground region of interest (ROI) while simultaneously serving a ground user equipment (UE) station. An RIS panel can be deployed either **on the satellite** (spaceborne, ~10 m panel) or **on the ground** (near the UE, ~1 m panel). Three deployment modes are compared: `none` (no RIS, baseline), `sat` (spaceborne RIS), and `ground` (ground RIS).

Real orbital ephemerides are obtained from public TLE sets (e.g., ISS NORAD 25544, Starlink) and propagated with the SGP4 model [7]. Coordinates are converted from ECI to ECEF including GMST rotation and Earth-rotation velocity corrections. The overpass window is searched over 48 h; frames are sampled at 1 s intervals around the window center where the elevation angle exceeds 20°.

Key physical parameters (defaults): carrier frequency 30 GHz (λ = 1 cm); satellite altitude ~420 km (ISS); ROI 80 m × 80 m at (30°N, 120°E); UE at (30°N, 119.5°E); slant range ≈ 695 km; Doppler up to ±610 kHz; round-trip delay ≈ 2.5 ms; SNR 20 dB.

**Physics verification** (script `verify_sat.py`) confirms: ISS altitude 418 km, velocity 7.66 km/s, period 92.9 min, overpass Doppler S-curve −610…+610 kHz, and channel equations — all consistent with published values.

### 2.2 Signal Model

Five propagation paths are modeled between the LEO satellite (BS), the ground target region (ROI), and the ground station (UE):

1. BS → ROI (direct illumination),
2. ROI → UE (direct scatter),
3. BS → UE (direct leakage),
4. BS → RIS → ROI (RIS-reflected illumination),
5. BS → RIS → UE and ROI → RIS → UE (RIS-forwarded paths).

For a set of voxel scatterers, the complex channel response for a baseband frequency *f* is

H(f) = Σ_i a_i exp(−j2π f τ_i),

where τ_i is the round-trip delay of scatterer i. The wideband **range profile** (HRRP) is obtained by inverse FFT of H(f) over K = 512 subcarriers with 1 GHz bandwidth (range resolution ≈ 0.15 m), plus AWGN at the configured SNR. The **ISAR sequence** extends this to M = 32 frames while the target rotates (e.g., drone circling or satellite spin), synthesizing cross-range resolution over the observation.

Two projection conventions are used for the delay:
- **centroid-relative** (`center='centroid'`): delays are measured relative to the voxel centroid — yields shape/pose features independent of target position (classic HRRP);
- **ROI-center-relative** (`center='roi'`): delays retain the absolute target position within the ROI (needed for localization).

The distinction matters: as shown in Section 5, using the centroid-relative convention for localization silently discards position information.

### 2.3 RIS Model and Phase Optimization

The RIS has N reconfigurable elements with unit-modulus phase control. For a given frame geometry, the *phase-aligned* configuration maximizes the coherent combination of the RIS-assisted path with the direct path at the UE. Because the satellite moves at ~7.5 km/s, the optimal phase pattern changes over the channel coherence time; we implement

- **Frame-by-frame tracking**: recompute the phase pattern every frame (1 s) from the current geometry — power gain **+283%** vs. random phase;
- **Segmented tracking**: restrict reconfiguration to every K-th frame to model limited RIS reconfiguration rates — for K = 8 the gain vanishes, quantitatively illustrating the "reconfiguration rate vs coherence time" trade-off.

### 2.4 Target Models

Ground targets are generated as 16³ voxel templates (5 m voxels; 80 m ROI) with 9 classes: car, UAV, building, tank, tower, cubesat, bicycle, pedestrian, train. Each sample is placed at a random position with a random pose (rotation about the vertical axis), which modulates the scattering-center distribution and hence the received amplitude/phase pattern.

---

## 3. Dynamic RIS Tracking and the Sensing–Communication Closed Loop

### 3.1 Sensing Layer

Two sensing capabilities are provided:

- **Diffusion-based 3D reconstruction** (`train_sat.py`): a conditional latent diffusion model (PointVAE encoder + DiT-style denoiser + CFG) reconstructs the ROI point cloud from per-frame channel features. Evaluation uses Chamfer Distance (CD), F-Score, and voxel IoU. In the space-ISAC setting, CD for the `sat` mode is 0.137–0.183 vs. 0.233 without RIS (training-scale results, 12 VAE + 10 LDM epochs).
- **Classification + localization** (`train_sensing*.py`): an MLP takes either narrowband channel features or the wideband HRRP (K = 512/1024) and outputs (i) target class among 9 classes and (ii) normalized 2D position within the ROI. The wideband HRRP mode is CPU real-time.

### 3.2 Communication Layer

Given the sensing output (target class + position), the IRS phase pattern is configured to maximize the coherent gain at the UE toward the sensed target direction. In the closed loop, communication performance is measured as received power:

- Random phase (baseline): 1.00×
- Sensing-aided: **+233%** (single target), **+289%** (multi-target)
- Ideal oracle (perfect CSI + position): +244%
- Sensing closed-loop efficiency: **98–99.8% of the oracle**

An honest observation from the experiments: even when classification is wrong, the coarse localization still captures most of the communication gain — a practically useful robustness property.

### 3.3 Closed-Loop Demo

The end-to-end pipeline is packaged as a one-click demo (`run_demo.sh`): sense target → configure IRS → measure communication gain → visualize. Outputs include a single-file HTML player (scene switching, timeline, real UTC overpass time) and GIF animations; a Google Colab notebook provides a 60-second zero-setup experience.

---

## 4. Wideband/ISAR Sensing and 3D Multi-Object Tracking

### 4.1 Narrowband → Wideband → ISAR

Classification accuracy on the fixed 150-sample test set:

| Feature | Accuracy |
|---|---|
| Narrowband channel features | 0.383 |
| Wideband HRRP (1 GHz, aligned) | **0.867** |
| ISAR sequence (M=32 frames) | **0.933** |

This progression quantifies the information added by range resolution (narrowband → HRRP) and by synthetic aperture over target rotation (HRRP → ISAR).

### 4.2 3D Multi-Object Tracking (MOT)

To support *moving* targets (e.g., a drone flying over a highway), the system includes a detection-tracking pipeline:

- **Scene data** (`mot_data.py`): 10 moving targets of 5 classes (car, drone, bicycle, pedestrian, train) with trajectories (straight/accelerating/turning), rendered into per-frame 5-path channel data and range profiles with Doppler.
- **Detector** (`train_detect.py`): CNN over range profiles → (class, position), CPU-trainable.
- **Tracker** (`mot_tracker.py`): Hungarian association with constant-velocity prediction, ID maintenance, class majority voting over track, and α-β smoothing.
- **3D output** (`demo_mot_html.py`): interactive Plotly HTML with full 3D trajectories.

Results: detection recall **0.812** across 10 targets; trajectory aggregation improves class accuracy over single-frame detection. A physical constraint is enforced: ground targets (car/bicycle/pedestrian/train) have z locked to 0, since a mono-static range profile carries very weak height information; only aerial targets (drone) have free z.

### 4.3 SDR Data Interface

An IQ data format (`sdr_io.py`) and ingest pipeline (`sdr_ingest.py`) allow loading time-domain IQ samples, converting via FFT to range profiles (fidelity 0.998 vs. simulation reference), and feeding the sensing layer. The interface is hardware-ready (RTL-SDR/USRP), enabling over-the-air capture without changing the downstream processing.

---

## 5. Classical Baselines and Physical/Engineering Findings

To position the learning-based sensing fairly, we implement classical radar processing baselines (`baseline_classic.py`) and evaluate them on the *same* fixed test set (60 samples, SNR 20 dB, seed-fixed):

- **2D-CA-CFAR** (P_fa = 10⁻⁴, guard/protection windows, convolution-vectorized) on the range–Doppler map (slow-time FFT of the ISAR sequence);
- **1D-CFAR** on the absolute-range profile with regression-calibrated bin→meters mapping (calibration set of 20 samples, R² ≈ 0.83);
- **MUSIC** with an 8-element ULA (λ/2 spacing, 64 snapshots) for target direction finding.

### 5.1 Results

| Method | Detection | LOS RMSE | Cross-range RMSE | Class acc. |
|---|---|---|---|---|
| 2D-CFAR (classical) | **1.000** | **6.98 m** | — (no angle info) | — |
| MUSIC (ULA-8) | — | — | — | DOA MAE **0.017°** |
| ML (absolute-range feature) | — | **3.09 m** | 15.68 m | 0.650 |
| ML (centroid-relative feature) | — | — | 2D RMSE 20.43 m | 0.800 |

### 5.2 Finding 1: Feature-Construction Defect

The original range-profile function computed delays relative to the **voxel centroid** (`rel = p − p_center`), which silently discards the target's absolute position within the ROI. A controlled single-voxel experiment confirmed this: moving the target across the ROI left the range-profile centroid bin *constant* (96.57). Consequently, ML localization trained on these features collapses to a *class-prior estimate* (2D RMSE 20.4 m) rather than a measurement-based estimate.

**Fix**: an absolute-range mode (`center='roi'`, differential delay relative to the ROI center) preserves position without K-bin wrap-around. After retraining: 2D localization MAE **20.4 → 12.2 m** (LOS ~3–5 m), classification 0.817, and the closed-loop demo is unaffected (+243.5%, 99.8% of oracle). Physics verification and all module imports remain green.

### 5.3 Finding 2: The Far-Field Angle-Resolution Wall

At a slant range of ~695 km, 1 m of cross-range offset subtends ≈ 8×10⁻⁵ degrees; the full 80 m ROI subtends ≈ 0.0066°. An 8-element ULA at λ/2 has a resolution of ~22.5° (Rayleigh). Therefore, **angle information (MUSIC or any array processing at practical sizes) cannot localize targets within the ROI** — cross-range localization is physically unavailable from mono-static angle measurements at these ranges. The ML cross-range RMSE of ~15.7 m reflects exactly this wall: its cross-range output is driven by class priors and training statistics, not by observable angular information.

This is a *physical* bound (geometry), not an implementation artifact: improving it requires multi-static geometry, large apertures (interferometric/ISAR imaging over long observation), or temporal priors (e.g., tracking), rather than more sophisticated single-site DOA processing.

---

## 6. Experiments and Reproducibility

### 6.1 Summary of Quantitative Results

| # | Experiment | Result |
|---|---|---|
| 1 | Orbit physics (ISS) | Altitude 418 km / 7.66 km/s / 92.9 min — matches real values |
| 2 | Overpass Doppler (30 GHz) | −610…+610 kHz S-curve (real LEO order) |
| 3 | RIS frame-by-frame tracking | Power **+283%** vs random phase |
| 4 | RIS segmented tracking | K=8 segments: gain vanishes (reconfiguration-limited) |
| 5 | Sensing–comm closed loop (single) | Class 83%, comm gain **+233%** (98% of oracle) |
| 6 | Sensing–comm closed loop (multi) | Detection 2/2, IRS gain **+289%** (94% of oracle) |
| 7 | Classification progression | Narrowband 0.383 → HRRP **0.867** → ISAR **0.933** |
| 8 | 3D MOT | 10 targets / 5 classes, recall **0.812** |
| 9 | SDR pipeline | IQ→FFT→range profile fidelity **0.998** |
| 10 | Robustness | ISS / Starlink×30 / 28 GHz: all PASS |
| 11 | 2D-CFAR baseline | Detection 100% (P_fa=10⁻⁴), LOS RMSE **6.98 m** |
| 12 | MUSIC baseline | ULA-8 DOA MAE **0.017°**; far-field angle wall quantified |
| 13 | Feature fix | Localization 2D MAE 20.4 → **12.2 m**; LOS 3–5 m |

### 6.2 Reproducibility

All results are produced by one-command scripts in the repository (Python 3.9+, PyTorch 2+, NumPy/SciPy, SGP4, scikit-learn):

```bash
cd source_code/isac_sat
../../.venv/bin/python verify_sat.py          # physics verification (ALL PASS)
bash run_demo.sh                              # closed-loop demo end-to-end
../../.venv/bin/python train_sensing.py --wideband   # sensing (classification + localization)
../../.venv/bin/python baseline_classic.py    # CFAR + MUSIC vs ML comparison
../../.venv/bin/python demo_mot.py            # 3D multi-object tracking
```

A GitHub Actions CI pipeline runs module-import checks, physics smoke tests, and SDR pipeline fidelity on every push. A Colab notebook (`colab/isac_demo.ipynb`) reproduces the core demo in ~60 seconds without local setup.

---

## 7. Limitations and Honest Discussion

We deliberately report limitations that are often omitted:

1. **Absolute attitude estimation is not feasible** in the far-field star–ground setting with simple symmetric templates — a physical upper bound, not an implementation gap.
2. **Single-station multi-target classification is limited** by signal mixing in the range profile; detection/localization remain usable, classification accuracy degrades (0.24 in the multi-target setting).
3. **Cross-range localization is angle-limited**: the far-field geometry makes mono-static angle-based cross-range localization impossible at practical array sizes (Section 5.3).
4. **Target templates are simple voxel models**; space-debris/satellite geometry models are planned.
5. **No over-the-air hardware validation yet**: the SDR interface is verified end-to-end on simulated IQ; hardware capture (RTL-SDR/USRP) is the natural next step.
6. **Atmosphere/ionosphere effects are not modeled** (free-space far-field approximation); relevant for low-elevation links.
7. **Evaluation scale is small** (tens of samples per experiment, smoke-level training); the scripts define the protocol, and larger runs are a matter of compute.

These limitations are documented in the repository's design document alongside the results, and we encourage contributors to help close them.

---

## 8. Conclusion

We presented an open-source, physics-grounded space ISAC engineering system combining real LEO orbit propagation, dynamic RIS phase tracking, learning-based sensing (diffusion 3D reconstruction, HRRP/ISAR classification, localization), 3D multi-object tracking, a sensing–communication closed loop, and an SDR data interface — all reproducible and CI-verified. Beyond the system itself, the report contributes two transferable findings: (i) a subtle but damaging feature-construction defect (centroid-relative delays discard absolute position) that we fixed and quantified; and (ii) a quantitative characterization of the far-field angle-resolution wall that bounds mono-static cross-range localization. We hope the system serves as a practical reference and testbed for space ISAC research, and that the honest limitation reporting raises the bar for reproducibility in this emerging area.

---

## Acknowledgments

This work is a school research project developed with assistance from AI tooling (Proma agent). The repository is maintained at [https://github.com/ConradLu2740/IRS-Diffu-ISAC](https://github.com/ConradLu2740/IRS-Diffu-ISAC) under the MIT license.

---

## References

1. F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, "Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond," *IEEE J. Sel. Areas Commun.*, vol. 40, no. 6, pp. 1728–1767, 2022.
2. A. Zhang, M. L. Rahman, X. Huang, Y. J. Guo, S. Chen, and R. W. Heath, "Perceptive mobile networks: Cellular networks with radio vision via joint communication and radar sensing," *IEEE Veh. Technol. Mag.*, vol. 16, no. 2, pp. 20–30, 2021.
3. Q. Wu and R. Zhang, "Intelligent reflecting surface enhanced wireless network via joint active and passive beamforming," *IEEE Trans. Wireless Commun.*, vol. 18, no. 11, pp. 5394–5409, 2019.
4. C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, "Reconfigurable intelligent surfaces for energy efficiency in wireless communication," *IEEE Trans. Wireless Commun.*, vol. 18, no. 8, pp. 4157–4170, 2019.
5. 3GPP, "Solutions for NR to support non-terrestrial networks (NTN)," TR 38.821, Release 16, 2020.
6. H. Wymeersch, D. Shrestha, C. M. M. de Lima, V. Yajnanarayana, B. Richerzhagen, M. F. Keskin, K. Schindhelm, A. Ramirez, A. Wolfgang, M. F. de Guzman, K. Haneda, T. Svensson, R. Baldemair, and S. Parkvall, "Integration of communication and sensing in 6G: A joint industrial and academic perspective," in *Proc. IEEE PIMRC*, 2021.
7. F. R. Hoots and R. L. Roehrich, "Spacetrack report no. 3: Models for propagation of NORAD element sets," U.S. Air Force Aerospace Defense Command, 1980.
8. Y. Luo, L. Kong, Y. Liu, et al., "Latent diffusion models for 3D point cloud generation," (LION), *NeurIPS*, 2023.
9. Z. Lyu, Z. Xu, Z. Xu, et al., "PVD: Point-voxel diffusion for 3D generative modeling," *ICCV*, 2021.
10. A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, "Point-E: A system for generating 3D point clouds from complex prompts," *arXiv:2212.08751*, 2022.
11. R. P. S. Buda and R. M. Narayanan, "Constant false alarm rate detection of pulse compression radar signals," *IEEE Radar Conf.*, 2005. (CFAR reference)
12. R. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Trans. Antennas Propag.*, vol. 34, no. 3, pp. 276–280, 1986. (MUSIC reference)
13. M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
14. C. Z. Lu, "IRS-Diffu-ISAC: RIS-aided ISAC via diffusion models for 3D point cloud reconstruction," GitHub repository, 2026. [Online]. Available: https://github.com/ConradLu2740/IRS-Diffu-ISAC

---

*Report generated for the ISAC-NTN research line of the IRS-Diffu-ISAC project. All numbers in this report are produced by the scripts in the companion repository and are reproducible as of commit 64ff549 (2026-08-08).*
