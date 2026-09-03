# RIS-Aided Integrated Sensing and Communication Toward Space ISAC: A Physics-Grounded Open-Source Engineering System

**Conrad Lu** (conrad.lu.2740@gmail.com)

*School of Information Science and Engineering, Northeastern University, Shenyang, China*

**Version**: v1.6 (2026-09-03) — companion to the open-source repository
[https://github.com/ConradLu2740/IRS-Diffu-ISAC](https://github.com/ConradLu2740/IRS-Diffu-ISAC)

*v1.4 additions: Rician-fading robustness of the RIS tracking trade-off (Section 6.3); a metric-dependence finding (ROI-object and baseline dependence of headline boosts); the layered `isac_sim/` reference library (Section 6.2).*
*v1.5 additions: the escape route from the far-field angle wall — two-station trilateration with the existing ground UE (Section 6.4), including a rank-deficiency warning for degenerate UE geometries.*
*v1.6 additions: cross-stack channel validation against 3GPP TR 38.901 via Sionna 2.x (Section 6.5) — the flat-fading and per-frame-independent approximations of the in-house channel layers are quantified against a standard CDL-D profile, and the K-sweep tracking experiment is rerun at the standard profile's K-factor.*

---

## Abstract

This report describes an open-source, physics-grounded engineering system for RIS-aided Integrated Sensing and Communication (ISAC) extended to space ISAC (ISAC-NTN). It combines real LEO orbit propagation (SGP4), dynamic RIS phase tracking, learning-based sensing, 3D multi-object tracking, and a sensing–communication closed loop, reproducible with one-command scripts (fixed seeds). Results: orbit physics matches real ISS values; RIS frame-by-frame tracking improves power by +89% (K=1), while reconfiguration-limited tracking (K=8) loses the gain; a sensing-aided closed loop achieves +309% communication gain (97.6% of the ideal oracle); wideband HRRP classification reaches 0.80 (5-class); 3D multi-object tracking (10 targets) achieves 0.60 recall. A comparison with classical baselines (2D-CFAR, MUSIC) uncovers two findings: (i) a feature-construction defect — centroid-relative delays discard absolute target position, collapsing ML localization to a class prior (22.6 m vs 12.1 m 2D RMSE with absolute-range features); (ii) a far-field angle wall — at ~695 km, the 80 m ROI subtends 0.0066°, far below an 8-element ULA resolution (~14°), so mono-static cross-range localization is physically unavailable. All numbers are reproducible (torch 2.8.0 reference, fixed seeds). Section 1.2 positions this system against the 2025–2026 ISAC literature, where the combination of diffusion-based point-cloud reconstruction with space ISAC (ISAC-NTN) remains an open niche as of August 2026. v1.4 additionally verifies (Section 6.3) that the RIS reconfiguration-rate trade-off survives per-frame independent Rician fading at K-factors down to 0 dB, and documents a metric-dependence finding: relative boost magnitudes depend on the ROI scatterer object and on the random-phase baseline, so cross-setting comparisons must fix both. v1.5 further demonstrates (Section 6.4) that the angle wall has an escape route already present in the scenario: two-station trilateration with the existing ground UE achieves 0.31 m cross-range RMSE (vs 11.8 m mono-static ML) at default geometry, with a rank-deficiency warning for degenerate UE placements. v1.6 adds cross-stack channel validation (Section 6.5): against a 3GPP TR 38.901 CDL-D profile (K ≈ 9 dB) generated with Sionna 2.x, the in-house channel layers' flat-fading and per-frame-independent approximations are quantified — the 1 GHz sensing bandwidth sits at the |ρ| = 0.89 LOS floor (frequency-flat processing is adequate), NLOS scatter decorrelates in tens of µs (10⁴× shorter than the 1 s frame interval, justifying per-frame independence), and the K-sweep trade-off is confirmed at the standard profile's K-factor.

**Keywords**: ISAC, RIS, non-terrestrial networks, LEO satellite, diffusion models, CFAR, MUSIC

---

## 1. Introduction

### 1.1 Background and Motivation

Integrated Sensing and Communication (ISAC) unifies wireless communication and radar-like sensing in a single system and is a key enabler of 6G [1, 2]. Reconfigurable Intelligent Surfaces (RIS) extend coverage and enhance both communication and sensing at low hardware cost [3, 4]. Non-terrestrial networks (NTN), in particular LEO constellations, are being integrated into 5G-Advanced/6G, and combining ISAC with NTN ("space ISAC") is of growing interest for space situational awareness and terrestrial monitoring [5, 6, 7].

Standardization is moving fast: 3GPP completed the Release 19 ISAC channel-modeling study item in May 2025 [17, 21], the Release 20 NR ISAC work item (TR 38.765) is nearing completion with normative specifications expected around 2027 [17], IEEE 802.11bf-2025 (WLAN sensing) was published in 2025 [19], and ITU-R lists ISAC among the six key usage scenarios of IMT-2030 [20]. NTN sensing, however, still has no dedicated 3GPP study/work item as of August 2026 [18, 22], leaving an open window for reproducible space-ISAC implementations.

Most published work remains at the level of analytical studies or link-level simulations with simplified geometry. Mature open-source link/PHY platforms (e.g., Sionna [8], MATLAB 5G/NTN toolboxes) exist but do not combine real orbit dynamics, dynamic RIS optimization, learning-based sensing, and closed-loop demonstration in one system. This report accompanies an engineering system that does; it makes no claim of algorithmic novelty over any single component. Its contributions:

1. **A physics-grounded, reproducible space ISAC pipeline** from real TLE orbit data to closed-loop demo, with one-command verification and CI.
2. **Dynamic RIS phase tracking** for LEO overpasses with an explicit quantification of the "reconfiguration rate vs channel coherence time" trade-off.
3. **A learning-based sensing layer** (classification + localization on wideband HRRP features) and **3D multi-object tracking**.
4. **A systematic classical-baseline comparison** (2D-CFAR + MUSIC vs ML) on identical test sets, surfacing a feature-construction defect and a quantified far-field angle wall.
5. **An SDR data interface** (IQ format + ingest pipeline) easing the transition to hardware.

### 1.2 Related Work and Positioning

**RIS-aided ISAC.** RIS has been shown to improve both communication and sensing at low hardware cost [3, 4]. 2025–2026 work has expanded toward new surface architectures (STAR-RIS, beyond-diagonal RIS, transmissive RIS transceivers, movable/multi-functional RIS) and toward near-field extremely-large surfaces; dynamic-RIS topics such as low-overhead beam training [23], 1-bit discrete phase optimization [24], and finite-blocklength sensing–communication trade-offs [25] are being actively studied. Our contribution is complementary: we quantify the "reconfiguration rate vs channel coherence time" trade-off for a fast-moving LEO overpass with an analytical phase-tracking scheme and a segmented-reconfiguration (K) sweep, on top of a real orbit (SGP4) geometry.

**Generative models for sensing and reconstruction.** Diffusion models have become a mainstream tool in wireless: channel estimation [26], radio-environment-map construction [27], and radar/LiDAR point-cloud generation [28, 29] are active areas, and a comprehensive survey of diffusion models for future networks has recently been submitted to the Proceedings of the IEEE [30]. In ISAC sensing, diffusion-based environment/point-cloud reconstruction from communication signals has started to appear (e.g., noise-sparsity-aware diffusion for ISAC environment reconstruction [31]; conditional diffusion point-cloud imaging for UAV sensing [32, 33]). These works focus on terrestrial/low-altitude scenarios with simplified geometry and do not involve RIS or real orbit dynamics; we differ by combining conditional latent point diffusion with RIS-aided space ISAC and real LEO ephemerides.

**ISAC-NTN / space ISAC.** Surveys and design principles for ISAC-enabled non-terrestrial networks have recently appeared [22], and LEO-satellite ISAC is moving from conceptual studies to system-level designs [34]. Doppler-robust waveforms (OTFS/AFDM) are the leading candidates discussed for high-dynamics ISAC [35]. To the best of our knowledge — and confirmed by searches of arXiv and public code repositories as of August 2026 — no open implementation combines real-orbit geometry, dynamic RIS phase tracking, diffusion-based 3D reconstruction, and a sensing–communication closed loop in one reproducible system; this report documents such a system.

### 1.3 Report Organization

Section 2: system/signal models. Section 3: dynamic RIS tracking and the closed loop. Section 4: wideband sensing and 3D MOT. Section 5: classical baselines and the two findings. Section 6: experiments and reproducibility. Section 7: limitations. Section 8: conclusion.

---

## 2. System and Signal Model

### 2.1 Scenario Geometry

A LEO satellite acts as the base station (BS), performs mono-static sensing of a ground region of interest (ROI), and serves a ground user equipment (UE) station. An RIS panel can be **spaceborne** (~10 m) or **ground** (~1 m, near UE); modes `none`/`sat`/`ground` are compared. Real TLE ephemerides (ISS NORAD 25544, Starlink) are propagated with SGP4 [9]; ECI→ECEF conversion includes GMST and Earth-rotation velocity corrections. The overpass window is searched over 48 h; frames are sampled at 1 s near window center where elevation > 20°.

Defaults: 30 GHz carrier (λ = 1 cm); ISS altitude ~420 km; ROI 80 m × 80 m at (30°N, 120°E); UE at (30°N, 119.5°E); BS–ROI slant ≈ 695 km; Doppler −611…+611 kHz; SNR 20 dB.

**Physics verification** (`verify_sat.py`): ISS altitude 418.3 km, velocity 7.66 km/s, period 92.9 min, Doppler S-curve, channel equations — consistent with published values.

### 2.2 Signal Model

Five propagation paths: (1) BS→ROI direct illumination, (2) ROI→UE direct scatter, (3) BS→UE direct leakage, (4) BS→RIS→ROI reflected illumination, (5a) BS→RIS→UE and (5b) ROI→RIS→UE forwarded paths. For voxel scatterers with complex amplitudes a_i and two-way (bistatic) delays τ_i, the baseband frequency response is

H(f) = Σ_i a_i exp(−j2π f τ_i),

and the wideband range profile (HRRP) is the inverse FFT of H(f) over K = 512 subcarriers at 1 GHz bandwidth (range resolution ≈ 0.15 m), plus AWGN. The ISAR sequence extends this over M = 32 frames while the target rotates, synthesizing cross-range resolution. Two delay conventions are used:

- **centroid-relative** (`center='centroid'`): delays relative to the voxel centroid — shape/pose features, position-independent (classic HRRP);
- **ROI-center-relative** (`center='roi'`): delays retain absolute target position within the ROI (needed for localization); a differential-delay formulation (τ = 1.1·d_proj/c, combining the 0.1 BS-side and 1.0 UE-side projection coefficients) avoids K-bin wrap-around.

As shown in Section 5, using the centroid-relative convention for localization silently discards position information.

### 2.3 RIS Model and Phase Optimization

The RIS has N unit-modulus phase elements. Phase-aligned configuration maximizes coherent combination of RIS-assisted and direct paths at the UE. Because the satellite moves at ~7.5 km/s, the optimal phase pattern changes over the channel coherence time. We compare:

- **Frame-by-frame tracking** (K=1): recompute phases every frame — power **+89.0%** vs random (seed-fixed, reproducible);
- **Segmented tracking** (K=2/4/8): reconfiguration limited to every K-th frame — K=2: +60.0%, K=4: +36.6%, K=8: **−41.5%** (stale phases can even hurt). This quantitatively illustrates the reconfiguration-rate vs coherence-time trade-off. Section 6.3 verifies this qualitative trade-off is robust under per-frame independent Rician fading (K = 10/5/0 dB, 5 seeds each).

### 2.4 Target Models

Ground targets are 16³ voxel templates (5 m voxels; 80 m ROI) with 5 classes: car, UAV, bicycle, pedestrian, train (static building/tank/tower/cubesat templates exist as functions and are pluggable). Each sample is placed at a random position with a random pose (rotation about the vertical axis), modulating the scattering-center distribution. Isotropic scattering is assumed (no RCS angular dependence, polarization, or occlusion modeled).

---

## 3. Dynamic RIS Tracking and the Sensing–Communication Closed Loop

### 3.1 Sensing Layer

- **Diffusion-based 3D reconstruction** (`train_sat.py`): conditional latent diffusion (PointVAE encoder + DiT-style denoiser + CFG; architecture follows latent point diffusion [10, 11, 12]) reconstructs the ROI point cloud; evaluated with Chamfer Distance (CD), F-Score, voxel IoU. Training-scale results: CD 0.137 (sat) / 0.169 (ground) vs 0.233 (no RIS).
- **Classification + localization** (`train_sensing*.py`): MLP over wideband HRRP (K = 512) outputs 5-class labels and normalized 2D position within the ROI; CPU real-time.

### 3.2 Communication Layer and Closed Loop

Given sensing output, the RIS phase pattern is configured toward the sensed target. Received power comparison (reproducible, seed-fixed): random phase 1.00×, sensing-aided **+309.4%**, ideal oracle +319.3% (closed-loop efficiency **97.6%**). Multi-target closed loop: detection 1/2 in the current 5-class setting, RIS pointing gain **+443.8%** (93% of oracle). A robustness observation: even when classification is wrong, coarse localization captures most of the communication gain.

### 3.3 Closed-Loop Demo

The pipeline is packaged as `run_demo.sh` plus a single-file HTML player and GIF animation; a Colab notebook gives a 60-second zero-setup experience.

*Interactive demos (HTML player, GIF, Colab) are companion repository content and are not part of this arXiv submission; static snapshots are available in the repository README.*

---

## 4. Wideband/ISAR Sensing and 3D Multi-Object Tracking

### 4.1 Narrowband → Wideband → ISAR

Classification accuracy progression (early experiments, 6-class templates; reported for reference): narrowband 0.383 → wideband HRRP 0.867 → ISAR sequence 0.933. With the current 5-class templates, wideband HRRP classification is **0.80** (reproducible, `train_sensing.py --wideband`, seed-fixed). The progression quantifies the information added by range resolution and by synthetic aperture over rotation.

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

- **2D-CA-CFAR** (P_fa = 10⁻⁴, convolution-vectorized; CA-CFAR formulation [13]) on the range–Doppler map (slow-time FFT of the ISAR sequence);
- **1D-CFAR** on the absolute-range profile with regression-calibrated bin→meters mapping (20-sample calibration set, R² ≈ 0.83);
- **MUSIC** with an 8-element ULA (λ/2, 64 snapshots) [14]; general radar background per [15].

*Honesty note: MUSIC is validated with synthetically generated point-source snapshots (a·s+n) — it does not share the same signal stream as CFAR/ML; the 0.017° figure verifies algorithmic self-consistency, not end-to-end sensing accuracy.*

### 5.1 Results

| Method | Detection | LOS RMSE | Cross-range RMSE | Class acc. |
|---|---|---|---|---|
| 2D-CFAR (detection) | **1.000** | — | — | — |
| 1D-CFAR (localization) | — | **8.14 m** | — (no angle info) | — |
| MUSIC (ULA-8, synthetic) | — | — | — | DOA MAE **0.017°** |
| ML (absolute-range) | — | **2.27 m** | 11.84 m | 0.733 |
| ML (absolute-range, cls head) | — | — | — | 0.733 |
| ML (centroid-relative, old) | — | — | 2D RMSE **22.63 m** | 0.817 |
| ML (shape, aligned) | — | — | 2D RMSE 21.86 m | 0.700 |

### 5.2 Finding 1: Feature-Construction Defect

The original range-profile function computed delays relative to the voxel centroid, silently discarding the target's absolute position. A controlled single-voxel experiment confirmed this: moving the target across the ROI left the range-profile centroid bin *constant*. Consequently, ML localization trained on centroid-relative features collapses to a class-prior estimate: 2D RMSE **22.63 m** (centroid-relative) and 21.86 m (aligned shape features) vs **12.06 m** with absolute-range (`center='roi'`) features — a ~2× gap.

**Fix and verification**: `compute_range_profile` now supports `center='roi'` (differential delay); `SatROIDataset` uses it when `rp_align=False`. After retraining: classification 0.80, 2D localization RMSE 12.1 m (LOS 2.3 m), and the closed-loop demo is unaffected (+309.4%, 97.6% of oracle). Physics verification and module imports remain green.

### 5.3 Finding 2: The Far-Field Angle-Resolution Wall

At ~695 km slant range, 1 m of cross-range offset subtends ≈ 8×10⁻⁵ degrees; the full 80 m ROI subtends ≈ 0.0066°. An 8-element ULA at λ/2 has a Rayleigh resolution of ≈ 0.886·λ/(Nd) ≈ **12.7°** (upper-bound estimate; even a finer reading λ/D ≈ 14.3° is orders of magnitude larger). Therefore mono-static angle information cannot localize targets within the ROI: ML cross-range RMSE ≈ 11.8 m reflects exactly this wall (its cross-range output is driven by class priors and training statistics, not observable angles). This is a physical geometry bound, not an implementation artifact — but it is a bound on **angle-only mono-static** localization specifically. The dual-station ISAC scenario already contains the escape route: the ground UE is a second range source, and Section 6.4 shows two-station trilateration breaks the wall by more than an order of magnitude.

---

## 6. Experiments and Reproducibility

### 6.1 Summary of Quantitative Results (all reproducible, seed-fixed 42)

| # | Experiment | Result |
|---|---|---|
| 1 | Orbit physics (ISS) | Altitude 418.3 km / 7.66 km/s / 92.9 min — matches real values |
| 2 | Overpass Doppler (30 GHz) | −611…+611 kHz S-curve (real LEO order) |
| 3 | RIS frame-by-frame tracking (K=1) | Power **+89.0%** vs random phase |
| 4 | RIS segmented tracking | K=2: +60.0%, K=4: +36.6%, K=8: **−41.5%** (stale phases harmful) |
| 5 | Sensing–comm closed loop (single) | Class 80%, comm gain **+309.4%** (97.6% of oracle) |
| 6 | Sensing–comm closed loop (multi) | Detection 1/2, RIS gain **+443.8%** (93% of oracle) |
| 7 | Classification (5-class, wideband HRRP) | **0.80** (early 6-class: 0.383→0.867→ISAR 0.933) |
| 8 | 3D MOT | 10 targets / 5 classes, recall **0.60**, class acc. 0.73 |
| 9 | SDR pipeline | IQ→FFT→range profile fidelity **0.998** |
| 10 | Robustness | ISS / Starlink × 30 GHz / 28 GHz: all PASS |
| 11 | 2D-CFAR / 1D-CFAR | Detection 100% (P_fa=10⁻⁴), LOS RMSE **8.14 m** |
| 12 | MUSIC (synthetic) | ULA-8 DOA MAE **0.017°**; far-field angle wall quantified |
| 13 | Feature fix | Localization 2D RMSE 22.63 → **12.06 m**; LOS 2.3 m |
| 14 | K-sweep under Rician fading (K=10/5/0 dB, 5 seeds) | Trade-off qualitative conclusion **ROBUST**: monotone in K, K=8 harmful at K=10 dB (−12.8% mean) |
| 15 | Metric-dependence findings | Headline boost +89.0% (legacy ROI object) vs +148% (data_sat ROI object); strong scattering shrinks the *relative* K=8 penalty |
| 16 | Two-station trilateration (default geometry, σ_ρ=0.15 m) | Cross-range RMSE **0.31 m** vs mono-static wall 11.84 m (**~38×**); break-wall budget σ_ρ < 6.6–8.9 m |
| 17 | Degenerate UE geometry (Δaz=0°) | Rank-deficient: cross-range error explodes to 1817 m — UE must lie outside the BS–target vertical plane |
| 18 | Sionna CDL-D channel cross-validation (K ≈ 9 dB, DS = 100 ns) | \|ρ(1 GHz)\| = 0.888 (LOS floor); NLOS scatter decorrelation ~30–65 µs ≪ 1 s frame (ratio ~10⁻⁴); K-sweep trade-off confirmed at the standard profile K |

### 6.2 Reproducibility

All results are produced by one-command scripts with **fixed global seeds** (`torch.manual_seed` + `np.random.seed` + `random.seed` at every entry point). Reference environment: torch 2.8.0, numpy 1.24+, Python 3.9; two consecutive runs in the same environment produce identical outputs. ML training numbers correspond to the parameters stated in each section (e.g., baseline_classic.py default 60 test / 300 train / 30 epochs); the closed-loop demo numbers are produced by `run_demo.sh` (train + evaluate in one step), which trains the sensing checkpoint in-place.

```bash
cd source_code/isac_sat
../../.venv/bin/python verify_sat.py            # physics verification (ALL PASS)
../../.venv/bin/python verify_tracking.py       # RIS tracking trade-off
../../.venv/bin/python verify_tracking_rician.py # K-sweep under Rician fading (v1.4, Section 6.3)
../../.venv/bin/python verify_twostation_localization.py  # two-station trilateration (v1.5, Section 6.4)
../../.venv/bin/python verify_sionna_channel.py  # Sionna CDL standard-channel cross-validation (v1.6, Section 6.5; needs `pip install sionna`)
../../.venv/bin/python train_sensing.py --wideband  # sensing (class + localization)
../../.venv/bin/python baseline_classic.py      # CFAR + MUSIC vs ML comparison
../../.venv/bin/python demo.py --checkpoint ./isac_demo/sensing_best.pth  # closed loop
../../.venv/bin/python demo_mot.py              # 3D multi-object tracking
```

The companion repository [16] runs a GitHub Actions CI pipeline (import checks, physics smoke tests, SDR fidelity) on every push; a Colab notebook reproduces the core demo in ~60 s.

Since v1.4 the repository also ships **`isac_sim/`**, a layered, pluggable simulation reference library (numpy-only core, no torch required for the classic layers): channels (free-space → Rician → Sionna CDL cross-validation layer), waveforms (OFDM), RIS models (continuous / 1-bit / segmented), communication links (QPSK-over-AWGN), sensing (1D CA-CFAR), tracking (nearest-neighbor + CV), and finding modules with analytic bounds (the far-field angle wall of Section 5.3). Every layer carries a physics sanity check wired into CI (`make smoke-sim`). Since v1.6, cross-stack validation against Sionna 2.x (PyTorch backend) is implemented for the channel layer (`make verify-sionna`, optional dependency, see Section 6.5); MATLAB reference implementations remain future work. `source_code/isac_sat` remains the reference application built on top of these layers.

### 6.3 Robustness under Rician Fading and Metric-Dependence Notes (v1.4)

The K-sweep headline values in Sections 2.3 and 6.1 were obtained under ideal free-space channels. To test whether the qualitative conclusion is an artifact of that idealization, we inject **per-frame independent Rician fading** into every scenario link (per-element power-aligned, same convention as `isac_sim/channels/rician.py`):

H'(t) = sqrt(K/(K+1)) · H + sqrt(1/(K+1)) · |H| ⊙ Z(t),  Z ~ CN(0,1)

with E|H'|² = |H|² (link budgets unchanged, only time selectivity added), for K-factors {10, 5, 0} dB and 5 seeds each (`verify_tracking_rician.py`, `make track-rician`). The free-space in-house rerun reproduces the v1.3 reference values exactly.

| Channel setting | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Free space (reference) | +89.0% | +60.0% | +36.6% | −41.5% |
| Rician 10 dB | +122 ± 21 | +82 ± 10 | +59 ± 9 | −12.8 ± 18 |
| Rician 5 dB | +165 ± 52 | +104 ± 26 | +74 ± 25 | +7 ± 39 |
| Rician 0 dB | +215 ± 91 | +126 ± 52 | +74 ± 43 | +39 ± 59 |

**Finding 3 (robustness).** The qualitative trade-off survives: the boost decreases monotonically in K in every setting, and per-seed K=8 is always far below K=1. At K = 10 dB the K=8 mean remains negative (−12.8%), i.e. stale phases still hurt.

**Finding 4 (metric dependence).** Two caveats when interpreting such headline numbers:
1. **ROI-object dependence.** The boost is relative to a random-phase baseline, and both numerator and denominator depend on the scatterer object. Replacing the legacy 16³ ROI object with the `data_sat.generate_ground_roi` object changes the K=1 boost from **+89.0% to +148%** (free space, identical geometry and seeds). Cross-setting comparisons must fix the object; v1.3 values correspond to the legacy object.
2. **Baseline-relative metric under strong scattering.** At K = 0 dB the *relative* K=8 penalty shrinks (mean +39%, std 59%) because the random-phase baseline power itself fluctuates with fading. The physical statement "stale phases are far worse than per-frame tracking" holds in every seed; the sign of the relative number at large K should not be over-interpreted.

### 6.4 Two-Station Trilateration: Escaping the Angle Wall (v1.5)

Section 5.3 established that mono-static **angle-only** cross-range localization is physically unavailable at LEO ranges. The dual-station ISAC scenario, however, already contains a second range source: the ground UE. Two range measurements (BS and UE, each with resolution ρ = c/2B determined by bandwidth and independent of array aperture) intersect on the known ground plane to localize in 2D — bypassing angle resolution entirely.

**Geometry (v2, corrected).** The LOS angle γ at the target must be computed from the actual station unit vectors, γ = arccos(û_BS · û_UE) — not from the ground baseline divided by slant range (an early small-angle estimate gave 3.9°, wrong by ~35×). In the default pass, the BS sits at 33.7° elevation and the UE is 142.5° away in azimuth (and on the horizon), giving **γ ≈ 131°**. A second, independent geometric condition matters: the UE must lie **outside the BS–target vertical plane**. When the azimuth offset Δaz → 0, the two LOS ground projections become parallel, the 2D problem loses rank, and the error explodes regardless of γ (measured 1817 m at σ_ρ = 0.15 m).

**Experiment** (`verify_twostation_localization.py`, `make twostation`): targets uniformly drawn from the 80 m ROI (ground-constrained), Gaussian range noise per station, Gauss–Newton solve, 2000 Monte Carlo runs per cell, swept over σ_ρ ∈ {0.15, 0.5, 1.5, 5} m × Δaz ∈ {0°, 45°, 90°, 142°} on the real default-pass geometry.

| Δaz (γ) | σ_ρ = 0.15 m | 0.5 m | 1.5 m | 5 m |
|---|---|---|---|---|
| 0° (34°, rank-deficient) | 1817 m | 2594 m | 5364 m | 7062 m |
| 45° (54°) | 0.25 m | 0.86 m | 2.63 m | 8.66 m |
| 90° (90°) | 0.15 m | 0.50 m | 1.51 m | 4.92 m |
| **142.5° (131°, default)** | **0.31 m** | 1.03 m | 3.12 m | 10.04 m |

(cross-range RMSE, i.e. the same axis on which the mono-static wall was measured; first-order theory σ_ρ/sin γ tracks the linear region within ~1.5×, degrading at σ_ρ = 5 m where errors exceed the ROI's linear regime.)

**Finding 5 (escape route + information interpretation).** At the default geometry, two-station trilateration achieves **0.31 m** cross-range RMSE — **~38× better** than the 11.8 m mono-static ML wall, and the break-the-wall budget is σ_ρ < 11.84·sin γ ≈ **6.6–8.9 m** for any non-degenerate UE placement — i.e. even CFAR-grade ranging suffices. The 11.8 m mono-static result therefore reflects **unused UE-side information, not missing information**: the correct target problem for future sensing work in this scenario is two-station range-fusion localization (with the BS–UE link dual-use for both communication and ranging), not further refinement of mono-static angle features.

*Honesty notes: ionospheric/atmospheric delay errors are not modeled (they dominate real star–ground ranging); the experiment answers the geometric/information question, not end-to-end accuracy.*

### 6.5 Cross-Stack Channel Validation Against 3GPP TR 38.901 via Sionna 2.x (v1.6)

All in-house results rest on two channel approximations: (i) **flat fading** (the scenario's per-link channel is a single complex gain per distance; the L1 Rician layer adds co-delay scatter only), and (ii) **per-frame independence** of the injected fading (Section 6.3). To quantify both, we cross-validate against the 3GPP TR 38.901 **CDL-D** standard profile generated with Sionna 2.0.1 (PyTorch backend), at the scenario's carrier (30 GHz) and with the UT speed set to the default pass's median LOS radial velocity (|v_rel| = 25–415 m/s, median 195 m/s → f_d ≈ ±42 kHz within the 8-frame window; the S-curve peak over the full pass reaches the ±611 kHz of Table 6.1, row 2).

*Honest boundary:* Sionna ships TR 38.901 terrestrial profiles only — TR 38.811 NTN profiles are not built in. CDL-D (explicit LOS + scatter clusters) is used as the standard-channel proxy: its LOS structure matches a satellite link, but its delay spread (100 ns default) and angle dispersion are terrestrial values. The Sionna Doppler model is isotropic Jakes scatter, not full orbit geometry; the scenario already models the deterministic LOS Doppler phase explicitly (frame-level f_d phase rotation, Section 2).

**(a) Frequency selectivity (flat-fading check).** From the CDL-D profile (K = 8.98 dB, delay spread 100 ns), the frequency correlation |ρ(Δf)| = |Σ Pℓ e^(−j2πΔfτℓ)| has a floor at the LOS power fraction K/(K+1) = 0.888: with a strong LOS component the channel never fully decorrelates in frequency (the |ρ| = 0.5 coherence bandwidth does not exist). Measured: |ρ| = 0.938 at one subcarrier spacing (1.95 MHz of the 512-subcarrier/1 GHz sensing waveform), |ρ| = 0.888 across the full 1 GHz bandwidth, coherence bandwidth @0.9 = 75 MHz.

**(b) Time selectivity (per-frame-independence check).** From Sionna CIR time series (Jakes Doppler at v = 195 m/s), the NLOS scatter component decorrelates at |ρ| = 0.5 in **~30–65 µs** (Jakes theory first zero: 19.6 µs) — **~10⁴× shorter** than the 1 s frame interval. The full-channel |ρ(Δt)| plateaus at the LOS floor (deterministic phase rotation, already modeled explicitly by the scenario's per-frame Doppler phase). The per-frame-independent Rician injection of Section 6.3 is therefore the correct marginal model for the scatter component at this frame rate.

**(c) K-factor alignment.** The CDL-D profile K-factor is fixed by the 38.901 table at **8.98 dB** — inside the P1 sweep range {10, 5, 0} dB. Rerunning the Section 6.3 tracking experiment at exactly K = 8.98 dB (2 seeds) reproduces the qualitative trade-off: K=1: +263/+296%, K=2: +52/+191%, K=4: +32/+92%, K=8: −7/+63% (relative boost, legacy ROI object; per Finding 4, magnitudes are baseline-relative and not comparable across settings, only the monotone-in-K trend and the K=8 ≪ K=1 gap are the invariant statements).

**Finding 6 (standard-channel coverage).** Both in-house approximations are adequate for the regime the experiments operate in: flat-fading processing loses only the NLOS-induced |ρ| ripple of amplitude ~1/(K+1) ≈ 0.11 across 1 GHz (relevant for future wideband delay-spread studies, not for the current narrow-per-link results), and per-frame independence is justified by a scatter coherence time four orders of magnitude below the frame interval. The K-sweep conclusion is confirmed at the standard profile's exact K-factor. `verify_sionna_channel.py` (`make verify-sionna`, optional dependency `pip install sionna`) reproduces all numbers with fixed seeds; the CDL wrapper lives in `isac_sim/channels/sionna_cdl.py` (lazy import — the rest of `isac_sim` does not require Sionna).

---

## 7. Limitations and Honest Discussion

1. **Absolute attitude estimation is not feasible** in the far-field star–ground setting with simple symmetric templates — a physical upper bound, not an implementation gap.
2. **Single-station multi-target classification is limited** by signal mixing in range profiles (detection/localization remain usable).
3. **Cross-range localization is angle-limited** (Section 5.3): mono-static angle-based cross-range localization is physically unavailable at practical array sizes for ~695 km links. Section 6.4 shows the bound is specific to angle-only mono-static processing: two-station trilateration with the existing UE breaks it by ~38× at CFAR-grade ranging.
4. **Target templates are simple voxel models** (isotropic scattering; no RCS angular dependence/polarization); space-debris/satellite geometry models are planned.
5. **No over-the-air hardware validation yet**: the SDR interface is verified on simulated IQ; RTL-SDR/USRP capture is the natural next step.
6. **Atmosphere/ionosphere effects are not modeled** (free-space far-field approximation).
7. **Evaluation scale is modest** (60–150 test samples, smoke-level training, single seed per experiment; multiple seeds and larger runs are a matter of compute). MOT is evaluated on a single scene.
8. **Historical values**: some early results (6-class classification progression) were produced before the current 5-class templates; they are reported for reference and are reproducible only from earlier commits.
9. **Relative-metric caveats**: RIS boost percentages are relative to a random-phase baseline and depend on the ROI scatterer object (Section 6.3, Finding 4); absolute received-power levels are simulation-calibrated.

---

## 8. Conclusion

We presented an open-source, physics-grounded space ISAC engineering system — real LEO orbits, dynamic RIS tracking, learning-based sensing, 3D MOT, closed loop, SDR interface — with fixed-seed reproducibility and CI verification. Two transferable findings emerge: (i) a feature-construction defect (centroid-relative delays discard absolute position) that we fixed and quantified (~2× localization gap); (ii) a quantitative far-field angle-resolution wall bounding mono-static cross-range localization. The system serves as a practical testbed for space ISAC research; the honest limitation reporting aims to raise the reproducibility bar in this emerging area. v1.4 adds a robustness verification of the RIS tracking trade-off under Rician fading, a metric-dependence analysis, and a layered `isac_sim/` reference library for broader reuse. v1.5 turns the far-field angle wall from a negative result into an actionable one: the scenario's own ground UE breaks the wall as a second range source, defining two-station range-fusion localization as the correct target problem for future sensing work. v1.6 closes the loop on channel fidelity: against a 3GPP TR 38.901 CDL-D profile (Sionna 2.x), the in-house flat-fading and per-frame-independent approximations are quantified and shown adequate for the regime studied, and the RIS tracking trade-off is confirmed at the standard profile's K-factor.

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
17. 3GPP, "Study on NR integrated sensing and communication," TR 38.765, Release 20, 2026.
18. 3GPP, "Service requirements for integrated sensing and communication," TS 22.137, Release 19, 2025.
19. IEEE, "IEEE Standard for Information Technology — Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications — Amendment: WLAN Sensing," IEEE 802.11bf-2025, 2025.
20. ITU-R, "IMT-2030 framework: Overall objectives of the future development of IMT for 2030 and beyond," Recommendation ITU-R M.2160-0, 2023.
21. Y. Liu, Y. Zhang, J. Zhang, Y. Pei, C. Zhao, S. Luo, et al., "A comprehensive survey of 3GPP Release 19 ISAC channel modeling: From empirical features to unified methodology and standardized simulator," arXiv:2512.03506, 2025.
22. M. A. Jamshed, R. Singh, M. M. Saad, et al., "ISAC-enabled non-terrestrial networks for 6G: Design principles, standardization, performance tradeoffs, and use cases," arXiv:2604.11593, 2026.
23. J. Yang, H. Lee, and J. Choi, "Beam training for RIS-aided ISAC systems," arXiv:2607.24003, 2026.
24. A. Gkekas, A. I. Papadopoulos, P. A. Pantazopoulos, A. Lalas, K. Votis, C. Liaskos, "Geometry-informed optimization of binary RIS configurations for communication and sensing," arXiv:2608.04133, 2026.
25. A. Umra, K. Weinberger, A. Khaleel, G. Enzner, and A. Sezgin, "Short blocks, fast sensing: Finite blocklength tradeoffs in RIS-assisted ISAC," arXiv:2511.02673, 2025.
26. M. Farzanullah, H. Zhang, A. B. Sediq, A. Afana, and M. Erol-Kantarci, "Conditional denoising diffusion for ISAC enhanced channel estimation in cell-free 6G," arXiv:2506.06942, 2025 (IEEE PIMRC).
27. X. Wang, Z. Fang, N. Cheng, et al., "RadioDiff-Inverse: Diffusion-enhanced Bayesian inverse estimation for ISAC radio map construction," IEEE Trans. Wireless Commun., 2026.
28. R. Zhang, B. Zeng, S. Wang, F. Zhou, and W. Wang, "RaLD: Generating high-resolution 3D radar point clouds with latent diffusion," arXiv:2511.07067, 2025.
29. J. Kwok, H. Caesar, and A. Palffy, "4D-RaDiff: Latent diffusion for 4D radar point cloud generation," arXiv:2512.14235, 2025.
30. N. C. Luong, N. D. Hai, D. V. Le, H. T. Nguyen, T.-H. Vu, T. Huynh-The, et al., "Diffusion models for future networks and communications: A comprehensive survey," arXiv:2508.01586, 2025 (submitted to Proceedings of the IEEE).
31. N. D. M. Quang, C. Liu, S. Li, et al., "Diffusion model-enhanced environment reconstruction in ISAC," arXiv:2511.19044, 2025 (submitted to IEEE Wireless Communications Letters).
32. X. Dai, Y. Gao, H. Jiang, X. Yuan, and X. Wang, "Conditional diffusion-based point cloud imaging for UAV position and attitude sensing," arXiv:2603.29822, 2026.
33. X. Dai, Y. Gao, H. Jiang, X. Yuan, and X. Wang, "Conditional generative learning enabled wireless UAV sensing and tracking via point cloud imaging," arXiv:2607.14778, 2026.
34. H. Yang, X. Chen, and Q. Wang, "Robust design of integrated sensing and communication in LEO satellite systems," arXiv:2607.12337, 2026.
35. P.-C. Chen, M.-C. Lee, and Y.-C. Huang, "Fundamental limits of MIMO-OTFS and MIMO-OFDM in high-dynamics ISAC: An antenna array architecture perspective," arXiv:2607.20200, 2026.

---

*Report v1.5. All numbers are produced by the scripts in the companion repository with fixed seeds and are reproducible at the commit accompanying this version (2026-09-02).*
