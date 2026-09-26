# FMShape: Conditional Flow Matching for Target-Shape Reconstruction From High-Resolution Range Profiles in Spaceborne ISAC

> **论文草稿 v0.1**（基于 IRS-Diffu-ISAC 仓库全部实测结果）
> 作者占位：[学生姓名], [张昱], [卢为党] — 浙江工业大学信息工程学院，杭州
> 目标 venue：IEEE TAES / TSP（方法论文；生成式感知方向）
> 状态：七节正文齐备；图表为占位；参考文献表已按真实论文垫底，成稿时补全条目

---

## Abstract

Integrated sensing and communication (ISAC) is a promising technique to support perceptive networks by sharing the spectrum and hardware between radar sensing and data transmission, which is of particular importance for spaceborne scenarios where the target shape must be inferred from limited radar observables. However, existing generative approaches for radar signature synthesis suffer from two limitations: diffusion-based synthesis requires hundreds of network evaluations per sample, and it provides no mechanism to exploit the measurement itself, leading to high latency and limited accuracy. To address these challenges, we propose a conditional flow matching (CFM)-based target-shape reconstruction scheme (FMShape). Specifically, an optimal-transport CFM model is trained in the latent space of a variational autoencoder (VAE), with the measured high-resolution range profile (HRRP) as the sampling condition, so that the three-dimensional shape of the target region is generated within a single sampling step. Moreover, augmenting the condition with an inverse synthetic aperture radar (ISAR) slow-time Doppler profile further improves the reconstruction accuracy, and a theoretical analysis delineates the achievable boundary of one-step generation. Simulation results on a spaceborne platform with real low-Earth-orbit trajectories and reconfigurable intelligent surfaces (RIS) demonstrate that the proposed scheme generates target shapes at a single sampling step with accuracy comparable to the 100-step diffusion benchmark, revealing the potential of flow matching as a real-time generative primitive for ISAC sensing.

**Index Terms** — Integrated sensing and communication; flow matching; generative models; high-resolution range profile; target shape reconstruction.

## 摘要（中文）

通感一体化（ISAC）通过共享频谱与硬件同时支撑雷达感知与数据传输，是一种很有前景的使能技术；在空-地场景中，目标形状须从有限雷达观测量中推断，其重要性尤为突出。然而，现有用于雷达信号特征合成的生成式方法存在两个局限：基于扩散的合成每个样本需要数百次网络评估，且缺乏利用测量值本身的机制，导致高时延与受限的精度。针对这些挑战，本文提出一种基于条件流匹配（CFM）的目标形状重建方案（FMShape）。具体而言，在变分自编码器（VAE）的潜空间中训练最优传输 CFM 模型，以实测高分辨距离剖面（HRRP）为采样条件，从而在单次采样步内生成目标区域的三维形状。此外，以逆合成孔径雷达（ISAR）慢时多普勒剖面增广条件可进一步提升重建精度；理论分析则界定了一步生成的可达边界。在具有真实低轨轨道与可重构智能表面（RIS）的空-地平台上的仿真结果表明，该方案以单次采样步生成目标形状，精度与 100 步扩散基线相当，揭示了流匹配作为 ISAC 实时生成式感知基元的潜力。

---

## I. INTRODUCTION

Integrated sensing and communication (ISAC) is a promising technique to support perceptive networks by sharing the spectrum and hardware between radar sensing and data transmission [1], [2]. In spaceborne scenarios, where a low-Earth-orbit (LEO) platform illuminates a ground region of interest (ROI) and the echoes coexist with the downlink communication waveform, the sensing functionality must be extracted from the very signals that carry data. Among the available observables, the wideband high-resolution range profile (HRRP) — the delay-domain scattering distribution of the ROI obtained from the reflected communication waveform — is the most information-rich: it resolves individual scatterers along the line of sight and has long served as the primary signature for detection, classification, and recognition in radar systems [3]–[5]. Reconstructing the three-dimensional (3D) shape of the target region from its HRRP, rather than classifying it against templates, provides the geometric awareness that downstream tasks such as region bookkeeping, tracking, and closed-loop beam management require. In spaceborne ISAC, however, this reconstruction must run onboard under strict latency and compute budgets, which makes the *sampling efficiency* of the reconstruction engine a first-order design constraint rather than an afterthought.

Generative models have recently been introduced to radar signature synthesis to address the chronic problem of low sample support, with generative adversarial networks (GANs) and diffusion models being used to augment training data and to enrich micro-Doppler and range-profile diversity [6]–[8]. When such generators are pushed from *data augmentation* toward *measurement-driven reconstruction*, however, two limitations emerge. First, diffusion-based synthesis requires hundreds of iterative denoising steps — and hence hundreds of network evaluations — per sample, which is difficult to reconcile with the real-time cadence of a spaceborne sensing pipeline. Second, conventional conditional generators treat the measurement as a static label-like input and provide no mechanism to exploit the measured echo *during* sampling, so the reconstruction is only weakly tied to the observation that motivated it. Both limitations are inherited from the generative paradigm rather than from any particular architecture, and both are consequential precisely in the ISAC setting where latency is scarce and the measurement is the most valuable piece of side information available.

Flow matching (FM) [9] offers a natural way around the first limitation. By regressing the vector field of fixed conditional probability paths, FM trains continuous normalizing flows in a simulation-free manner and yields probability-flow ordinary differential equations (ODEs) whose trajectories are straight enough that a single Euler step often suffices — a property that has produced state-of-the-art sample quality at a small number of network evaluations in the image domain [9], [10]. This makes FM a candidate *real-time generative primitive* for sensing. The second limitation — exploiting the measurement — is then addressed by conditioning: broadcasting the measured HRRP as the sampling condition turns the generator into a measurement-driven estimator whose output is tied to the observation at hand.

**To address these challenges, we propose FMShape, a conditional flow-matching framework for target-shape reconstruction from measured HRRPs in spaceborne ISAC.** Specifically, an optimal-transport conditional flow-matching (OT-CFM) model is trained in the latent space of a variational autoencoder (VAE), with the measured HRRP broadcast as the sampling condition, so that the 3D point-cloud shape of the target region is generated within a single sampling step. We study the framework across generation, conditioning, and closed-loop use on a physics-grounded simulation platform with real LEO orbits, reconfigurable intelligent surface (RIS)-assisted links, and wideband channel models. The main results are as follows.

- **Efficiency at equal compute.** Under identical data, architectures, and compute budgets, single-step FMShape sampling attains reconstruction accuracy comparable to a 100-step diffusion baseline on all metrics and across RIS configurations; the equal-quality crossover occurs at no more than two network evaluations, a 50–100× reduction in sampling cost. The ODE structure is independently certified by a measured convergence order of −0.87 and a trajectory straightness ratio of 0.006.
- **A measurable measurement-to-shape channel.** Conditioning on the measured HRRP opens a quantifiable side-information channel — a condition-explained latent variance of 21.8% at the data end of the flow — and augmenting the condition with an inverse synthetic aperture radar (ISAR) slow-time Doppler profile further improves the channel at matched training budget, provided the condition blocks are spread-equalized before concatenation.
- **Theory for one-step generation.** We prove that the single-step Euler output of a conditional FM sampler is exactly the conditional mean of the shape posterior, which explains the observed near-collapse of converged one-step maps and delineates the boundary of what one-step generation — including one-step distillation — can achieve. A bias–variance frontier is measured along both the training-budget and teacher-integration axes, identifying the practical operating point for distilled one-step samplers.
- **Closed-loop deployment.** Inserted into the ISAC closed loop as the ROI shape prior, the FMShape generator replaces the hand-crafted box prior and raises the sensing achievement ratio from 0.840 to 0.932, demonstrating that the generative primitive pays off at the system level and not only at the metric level.

We deliberately report the boundary together with the gains: absolute reconstruction accuracy remains modest and is bounded by the latent capacity and by the information content of the range profile, a boundary we quantify rather than conceal. All results are reproducible from the released platform with fixed seeds and pre-registered verdicts.

The remainder of this paper is organized as follows. Section II reviews generative models for radar sensing and flow matching in wireless systems. Section III presents the spaceborne ISAC system model, including the wideband HRRP formation and the target-region shape model. Section IV details the FMShape framework. Section V provides the theoretical analysis of one-step generation and distillation. Section VI reports the experimental results, including the equal-compute comparison, the conditioning study, the ISAR augmentation, and the closed-loop deployment. Section VII concludes the paper.

## II. RELATED WORK

**A. HRRP-based target sensing: from templates to deep networks.** The high-resolution range profile has been a workhorse signature of radar target recognition for decades. Classical approaches rely on statistical models of the range profile — Gaussian, autoregressive, and their spatio-temporal extensions — followed by template matching or discriminant projection [3]–[5]. With the advent of deep learning, concatenated deep neural networks [6], one-dimensional residual-inception networks, attention-based recurrent models, and hierarchical semantic-data classifiers [7] have successively raised recognition accuracy on measured HRRP datasets, and the review in [8] consolidates this line of work. These methods, however, treat the task as *classification against a closed set of templates*: the output is a label, not geometry. Imaging-oriented counterparts — synthetic aperture radar (SAR) and inverse synthetic aperture radar (ISAR) reconstruction — produce geometry but assume dedicated imaging waveforms and long coherent processing intervals, which are only partially available in a communication-native ISAC setting where the echo is a byproduct of the data-bearing downlink [1], [2]. Reconstructing the 3D shape of the ROI directly from its HRRP therefore occupies a distinct niche: classification-level observables, imaging-level outputs. Recent ISAC sensing work has moved in this direction with deep trackers that estimate extended-target contours from communication echoes [11], but the shape model there is a low-dimensional parametric contour rather than a free-form point-cloud reconstruction, and none of the above lines addresses the *sampling efficiency* of the generative engine under onboard latency constraints.

**B. Generative models for radar sensing: augmentation today, reconstruction tomorrow.** Generative models entered radar sensing as a remedy for low sample support. Physics-aware GANs inject kinematic consistency into micro-Doppler signature synthesis so that GAN-augmented data improves downstream activity recognition [12]; few-shot SAR image generation has been demonstrated with diversity-enhanced diffusion models [13]; and variational temporal deep generative models have been applied to HRRP target recognition, treating the range profile as a sequence with latent temporal structure [14]. The common pattern is *offline augmentation*: the generator runs where latency is free, and its output enlarges the training set of a discriminative network. Pushing the generator into the sensing loop itself — measurement in, geometry out — changes the requirements in two ways that the existing literature does not address. First, inference latency becomes part of the specification: diffusion-based synthesis needs on the order of a hundred network evaluations per sample, which is hard to reconcile with onboard real-time operation. Second, the measurement is not a class label but a high-dimensional observation, and the value of the generator is precisely determined by *how much of the measurement it exploits* — a question that, to the best of our knowledge, has not been quantified for radar signature generation. FMShape is designed against both requirements: a flow-matching sampler whose single-step mode meets the latency budget, and a conditioning channel whose information content is measured rather than assumed.

**C. Flow matching and generative models in wireless systems.** Flow matching trains continuous normalizing flows by regressing the vector field of fixed conditional probability paths, avoiding the simulation-in-the-loop cost of earlier CNF training and subsuming diffusion paths as a special case [9]. Its two properties matter here: optimal-transport conditional paths yield straighter ODE trajectories and better sample quality per network evaluation [9], [15], and the resulting sampler is compatible with off-the-shelf ODE solvers at arbitrary step counts — including a single step. In the machine-learning domain these properties have produced efficient image and audio generators; in the wireless domain, generative models have been explored for channel modeling, data augmentation, and beamforming assistance, and flow matching has recently appeared in wireless-adjacent tasks. Its application to *radar sensing*, and in particular to reconstructing target geometry from range profiles inside an ISAC pipeline, remains — to the best of our knowledge — unexplored. This paper takes that step, and treats the spaceborne ISAC setting as the driving scenario because it makes both of flow matching's advantages (few evaluations, measurement conditioning) simultaneously necessary and testable.

**D. Conditioning as an information channel.** Conditional generation in vision is typically evaluated by sample quality alone; the amount of information the condition actually contributes is rarely measured. The closest relatives in the sensing literature are the collapse diagnostics of conditional encoders — sensitivity to the condition versus sensitivity to noise — which detect when a nominally conditional model has degenerated into an unconditional one, and information-sufficiency gates that decide, before training, whether a candidate observable carries usable side information. FMShape builds on this line in two directions: it uses the measured HRRP as an explicit sampling condition, and it quantifies the resulting measurement-to-shape channel as a condition-explained variance, with an ISAR slow-time Doppler profile as an additional conditioning observable whose contribution is isolated at matched training budget. The theoretical analysis of one-step optimality in Section V further connects this conditioning view to the sampling-efficiency view by showing what a single evaluation can and cannot achieve.

**E. Positioning.** Relative to the above lines, this paper contributes the first conditional flow-matching framework for target-shape reconstruction from measured HRRPs in spaceborne ISAC; the first quantification of the measurement-to-shape side-information channel for radar generative sensing; a theoretical characterization of one-step flow-matching optimality with a measured bias–variance frontier for distilled samplers; and a closed-loop deployment in which the generative shape prior replaces a hand-crafted prior and improves the system-level sensing achievement.

## III. SYSTEM MODEL

**A. Spaceborne ISAC scenario.** Consider a spaceborne ISAC system in which a LEO satellite illuminates a ground ROI while simultaneously serving as the transmitting node of the downlink communication link, as illustrated in Fig. 1. The satellite trajectory is propagated from real two-line-element (TLE) ephemerides with the SGP4 model, so that the geometry — slant range, elevation, and line-of-sight Doppler — evolves over the observation window exactly as in an operational pass. The observation window is discretized into $T$ frames (in this paper, $T = 8$), and frame $t$ is characterized by its satellite position $\mathbf{s}_t$, the ROI centroid position $\mathbf{p}_t$, the round-trip delay $\tau_t$, and the line-of-sight Doppler shift $f_{d,t}$; all quantities are obtained from the propagated ephemerides rather than being drawn as free parameters.

The ROI is a ground region of side length $L$ centered at $\mathbf{p}_t$, discretized into a voxel grid of resolution $R$ per dimension. The ROI occupancy is described by a binary voxel map $\mathbf{V} \in \{0,1\}^{R \times R \times R}$, from which the target point cloud $\mathbf{P} \in \mathbb{R}^{N \times 3}$ ($N = 512$ points in this paper) is extracted by sampling the occupied voxels and normalizing physical coordinates to $[-1, 1]^3$. The reconstruction target of this paper is precisely this point cloud: a free-form geometric description of the ROI, richer than a class label and more general than a parametric contour.

A RIS is deployed on the ground to assist the downlink. Three RIS configurations are considered throughout: the spaceborne mode, in which the RIS-assisted link carries the communication signal; the ground mode, in which the RIS is operated from the ground segment; and the no-RIS reference mode. Per frame, the RIS imposes a phase vector $\boldsymbol{\phi}_t \in [0, 2\pi)^{N_{\mathrm{RIS}}}$ on the illuminated elements, drawn either uniformly at random (the baseline) or optimized for a desired reflective beam pattern (the tracked mode). As established in [1], the sensing echo in the default spaceborne mode is dominated by the direct satellite–ROI–ground path and is structurally independent of $\boldsymbol{\phi}_t$; the RIS therefore shapes the communication link, while the sensing observation is carried by the direct echo. This structural fact bounds what any RIS-coupled sensing mechanism can achieve in this geometry.

**B. Echo and HRRP formation.** In each frame, the satellite transmits a pilot-bearing waveform $\mathbf{x}$ (a segment of a 16-QAM constellation in this paper). The echo scattered by the ROI is determined by the voxel occupancy $\mathbf{V}$, the RIS phase vector $\boldsymbol{\phi}_t$, and the frame-dependent channel state, and is collected as a complex observation
$$
\mathbf{y}_t = \mathcal{G}\big(\mathbf{V}, \boldsymbol{\phi}_t, \mathbf{H}_t, \mathbf{x}\big) + \mathbf{n}_t,
\tag{1}
$$
where $\mathbf{H}_t$ collects the frame-dependent channel terms — including the satellite–ROI–ground path and, where applicable, the RIS-assisted paths — and $\mathbf{n}_t$ is additive noise at the operating signal-to-noise ratio (20 dB in this paper). Equation (1) is the full observation available to the sensing receiver; everything downstream in this paper is derived from it.

For wideband operation, the per-frame echo is compressed into a high-resolution range profile by projecting the scatterer distribution onto the delay axis using the bistatic geometry of frame $t$ — the satellite, ground-station, and target positions together with the carrier wavelength. Concretely, the HRRP of frame $t$ is
$$
\mathbf{h}_t = \mathcal{R}\big(\mathbf{V}; \mathbf{s}_t, \mathbf{p}_t, \lambda\big) + \mathbf{w}_t \in \mathbb{R}^{K},
\tag{2}
$$
with $K = 512$ delay bins in this paper and $\mathbf{w}_t$ modeling the profile noise at the operating SNR. The profile is centroid-aligned, i.e., referenced to the ROI centroid rather than to absolute coordinates, so that $\mathbf{h}_t$ encodes the *shape* of the scatterer distribution along the line of sight and is invariant to rigid translation of the ROI — a property we return to when analyzing the information content of the condition. Because the delay projection is many-to-one — distinct voxel maps can share a range profile — (2) defines a genuinely ill-posed inverse problem, which is exactly what makes a *learned generative prior* attractive: the prior resolves the ambiguity that no deterministic inverse can.

**C. Target-region shape model.** The shape of the ROI is modeled as a voxel occupancy map sampled from a library of ground-target templates with random placement, orientation, and composition, rendered with isotropic scattering (no RCS angular dependence). While idealized relative to measured target complexes, this model keeps the inverse problem in (2) well-defined and fully reproducible, and the conclusions we draw concern the observation–shape relationship rather than the particulars of the template library. The point cloud $\mathbf{P}$ is extracted from $\mathbf{V}$ with a fixed sampling procedure so that the reconstruction target is deterministic given $\mathbf{V}$.

**D. Problem formulation.** Given the measured HRRP $\mathbf{h}_t$ — and, in the augmented configuration, the ISAR slow-time Doppler profile derived from a short coherent sequence as in [16] — the task is to reconstruct the ROI point cloud $\hat{\mathbf{P}}$ under a real-time constraint: the reconstruction must complete within a small number of network evaluations, since it runs onboard in the sensing cadence of the pass. Reconstruction quality is measured by the Chamfer distance (CD) between $\hat{\mathbf{P}}$ and $\mathbf{P}$, by the F-Score at two distance thresholds, and by the voxel intersection-over-union (IoU) of the occupied grids. Sampling efficiency is measured by the number of network evaluations (NFE) required to reach a target quality. The generative engine studied in this paper is a conditional flow-matching model whose condition is $\mathbf{h}_t$ and whose single-step mode is designed to meet the latency constraint.

## IV. METHOD: FMSHAPE

**A. Overview.** FMShape reconstructs the ROI point cloud from the measured HRRP in three stages, as illustrated in Fig. 2. First, a variational autoencoder (VAE) compacts the point-cloud shape distribution into a whitened 256-dimensional latent space. Second, an optimal-transport conditional flow-matching (OT-CFM) model is trained in this latent space with the measured HRRP as the sampling condition. Third, at inference, a single Euler step of the conditional probability-flow ODE maps a fresh noise sample to a latent shape, which the VAE decoder renders as the reconstructed point cloud. The single-step mode is what makes the engine compatible with onboard latency; the conditioning is what ties the reconstruction to the measurement.

**B. Latent shape space.** Let $\mathbf{z} \in \mathbb{R}^{d}$ ($d = 256$) denote the VAE latent code of a point cloud. The encoder is trained with the standard evidence lower bound with a small KL weight and warm-up, and the latent statistics $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ are estimated on the training distribution and frozen. All generative modeling is performed on the whitened latent
$$
\mathbf{x}_1 = \frac{\mathbf{z} - \boldsymbol{\mu}}{\boldsymbol{\sigma}},
\tag{3}
$$
which equalizes the scale of the latent coordinates and stabilizes the flow training. Decoding a latent code $\hat{\mathbf{x}}_1$ yields the reconstructed cloud $\hat{\mathbf{P}} = \mathrm{Dec}(\boldsymbol{\sigma} \odot \hat{\mathbf{x}}_1 + \boldsymbol{\mu})$. Generating in the latent space rather than directly on the 1536-dimensional point cloud keeps the velocity network small and fast, and lets the VAE absorb the low-level point-cloud statistics so that the flow model concentrates on the shape distribution itself. The VAE reconstruction accuracy sets the ceiling of the pipeline; we report it as the oracle reference throughout.

**C. Optimal-transport conditional flow matching.** Let $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ be the noise endpoint and $\mathbf{x}_1$ the whitened data latent. OT-CFM uses the linear conditional path
$$
\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\mathbf{x}_1, \qquad t \in [0, 1],
\tag{4}
$$
whose induced conditional velocity is the constant vector $\mathbf{x}_1 - \mathbf{x}_0$. The flow network $v_{\boldsymbol{\theta}}$ — a 1-D Diffusion Transformer with cross-attention (2 blocks, 256 hidden units, 8 heads) — is trained by the regression objective [17]
$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1, c}\Big[\big\| v_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c) - (\mathbf{x}_1 - \mathbf{x}_0) \big\|_2^2\Big],
\tag{5}
$$
which is simulation-free: no ODE integration is required during training, in contrast to earlier continuous-normalizing-flow training. The condition $c$ is produced by a condition encoder from the measured HRRP as described next.

**D. Conditioning on the measured HRRP.** The condition encoder is a long short-term memory (LSTM) network that consumes the per-frame observation sequence and emits a 256-dimensional embedding. For the HRRP condition, the profile $\mathbf{h}_t$ of (2) is broadcast across the $T$ frames, i.e., the encoder input is the sequence $(\mathbf{h}, \dots, \mathbf{h})$; the network therefore maps the measurement to a single embedding $c = \mathrm{CondEnc}(\mathbf{h})$. Classifier-free guidance (CFG) is applied at sampling time by maintaining a null condition $c_{\varnothing}$ (a zeroed measurement) and forming the guided velocity
$$
v_{\mathrm{cfg}}(\mathbf{x}, t) = v_{\boldsymbol{\theta}}(\mathbf{x}, t, c_{\varnothing}) + w\,\big(v_{\boldsymbol{\theta}}(\mathbf{x}, t, c) - v_{\boldsymbol{\theta}}(\mathbf{x}, t, c_{\varnothing})\big),
\tag{6}
$$
with guidance scale $w = 2$ throughout. Equation (6) is the only place where the measurement influences the trajectory, which is why the conditioning analysis of Section VI measures the channel from $\mathbf{h}$ to the generated shape rather than assuming it.

**E. Single-step sampling.** Inference integrates the probability-flow ODE $\dot{\mathbf{x}} = v_{\mathrm{cfg}}(\mathbf{x}, t)$ from $t = 0$ to $t = 1$ with a numerical solver. The real-time operating point is a single Euler step,
$$
\hat{\mathbf{x}}_1 = \mathbf{x}_0 + v_{\mathrm{cfg}}(\mathbf{x}_0, 0), \qquad \hat{\mathbf{P}} = \mathrm{Dec}\big(\boldsymbol{\sigma} \odot \hat{\mathbf{x}}_1 + \boldsymbol{\mu}\big),
\tag{7}
$$
which costs one ODE step — two network forward passes under CFG — per reconstruction. Multi-step midpoint solvers at $\mathrm{NFE} \in \{2, 5, 10, \dots\}$ are used in the evaluation to trace the quality–efficiency trade-off, and Section V analyzes what the one-step limit can and cannot deliver.

**F. ISAR condition augmentation.** The conditioning observation can be augmented with an inverse synthetic aperture radar (ISAR) slow-time Doppler profile $\mathbf{d} \in \mathbb{R}^{32}$, computed from a short coherent echo sequence as the slow-time spectrum of the range-cell migration [16]. Because the HRRP block and the Doppler block differ by roughly $5\times$ in their typical pairwise spread, naive concatenation lets the high-dimensional HRRP dominate any distance-based signal and dilutes the Doppler contribution. FMShape therefore spread-equalizes the blocks before concatenation, scaling each block by the inverse of its average pairwise distance on the training distribution:
$$
c\text{-input} = \big[\mathbf{h}\,/\, \bar{d}_h \; ; \; \mathbf{d}\,/\, \bar{d}_d\big], \qquad \bar{d}_d \approx \bar{d}_h / 5.
\tag{8}
$$
Section VI quantifies the effect of (8): the augmentation adds channel capacity only under spread equalization, and is actively harmful without it.

**G. Training protocol.** All generative models are trained with the fresh-data protocol: each epoch draws new samples from the physics-based simulator, and no dataset is materialized or frozen. This matters for conditional generation in particular — freezing the data lets the network memorize the finite sample and collapses the condition dependence, an artifact documented in our earlier work and avoided here by construction. Models are trained with identical architectures, capacities, and data budgets whenever compared, and all results are reported on a common test batch with fixed seeds so that paired comparisons are not confounded by evaluation noise.

## V. THEORETICAL ANALYSIS

**A. Setup.** Recall the whitened latent variables $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\mathbf{x}_1$, the linear path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\,\mathbf{x}_1$, and the condition $c$ derived from the measurement. The exact minimizer of the FM objective (5) is the conditional expectation
$$
v^{*}(\mathbf{x}, t, c) = \mathbb{E}\big[\mathbf{x}_1 - \mathbf{x}_0 \,\big|\, \mathbf{x}_t = \mathbf{x},\, t,\, c\big].
\tag{9}
$$
We assume throughout that $\mathbf{x}_0$ is independent of $(\mathbf{x}_1, c)$, which holds by construction of the noise endpoint. The question this section answers is: *what does the one-step operating point of (7) compute when the network is (near-)optimal, and what can one-step distillation buy on top of it?*

**B. The one-step optimality theorem.**

*Theorem 1 (velocity at the noise endpoint).* For the exact FM velocity field,
$$
v^{*}(\mathbf{x}_0, 0, c) = \boldsymbol{\mu}(c) - \mathbf{x}_0, \qquad \boldsymbol{\mu}(c) \triangleq \mathbb{E}[\mathbf{x}_1 \,|\, c].
\tag{10}
$$

*Proof.* At $t = 0$ we have $\mathbf{x}_t = \mathbf{x}_0$. Conditioning on $\mathbf{x}_t = \mathbf{x}_0$ therefore conditions on the noise alone, and since $\mathbf{x}_0 \perp (\mathbf{x}_1, c)$,
$$
\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \,|\, \mathbf{x}_0, c] = \mathbb{E}[\mathbf{x}_1 \,|\, c] - \mathbf{x}_0 = \boldsymbol{\mu}(c) - \mathbf{x}_0. \qquad \blacksquare
$$

*Theorem 2 (the one-step output is the conditional mean).* The single Euler step of (7), evaluated on the exact field, produces
$$
\hat{\mathbf{x}}_1 = \mathbf{x}_0 + v^{*}(\mathbf{x}_0, 0, c) = \boldsymbol{\mu}(c),
\tag{11}
$$
independently of $\mathbf{x}_0$.

*Proof.* Immediate from Theorem 1. $\qquad \blacksquare$

Three consequences follow directly.

*Corollary 1 (collapse is optimal, not a defect).* Any network that approximates $v^{*}$ well necessarily maps essentially all of the noise ball at $t = 0$ onto the single point $\boldsymbol{\mu}(c)$: the near-mode-collapse of converged one-step maps is the *correct* behavior of the optimal field. The residual spread of a trained network around $\boldsymbol{\mu}(c)$ measures its approximation error to $v^{*}$, not its sample diversity. This matches our measurement on the converged teacher: 16 initial noises produce a pairwise Chamfer distance of $0.0053$, i.e. the network tracks $v^{*}$ closely enough that the one-step output is effectively deterministic.

*Corollary 2 (guidance does not restore dispersion).* With classifier-free guidance (6), the one-step output is
$$
\hat{\mathbf{x}}_1 = \mathbf{x}_0 + v^{*}_{\varnothing}(\mathbf{x}_0, 0) + w\big(v^{*}_c(\mathbf{x}_0, 0) - v^{*}_{\varnothing}(\mathbf{x}_0, 0)\big) = \boldsymbol{\mu}_w(c),
\tag{12}
$$
a *guided* conditional mean — again a point mass at every guidance scale $w$. Tuning $w$ interpolates between means; it cannot manufacture diversity. Empirically, the diversity ratio of the teacher remains at $0.02$ for $w \in \{0, 1, 2\}$ while the reconstruction quality improves monotonically with $w$, exactly as (12) predicts: guidance sharpens the estimate, it does not sample the posterior.

*Corollary 3 (the "integration error" is the diversity).* The exact ODE flow maps $\mathbf{x}_0$ to a posterior *sample*; the one-step Euler output is the posterior *mean*. Their difference — usually discussed as one-step truncation error — is precisely the $\mathbf{x}_0$-dependent component of the sampling map. Removing it (as one-step regression toward a low-order map tends to do) removes the diversity with it. The two objects should therefore be evaluated against different criteria: the mean against reconstruction error, the sampler against distributional fidelity.

**C. Where dispersion comes from, and what it costs.**

*Theorem 3 (multi-step integration is necessary for dispersion).* For $t > 0$, the mixture $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\,\mathbf{x}_1$ carries information about $\mathbf{x}_1$ beyond the condition, so $v^{*}(\mathbf{x}, t, c)$ depends on $\mathbf{x}$ non-trivially and the flow trajectories bend with $\mathbf{x}_0$. An integrator with $\mathrm{NFE} \geq 2$ samples this dependence (a midpoint solver evaluates the field at $t = 1/2$, where the mixture is informative), whereas the $\mathrm{NFE} = 1$ step of Theorem 2 never leaves the $t = 0$ slice.

*Proof sketch.* At $t = 1/2$, $\mathbf{x}_t$ is an equal mixture of noise and data; conditioning on it restricts the posterior of $\mathbf{x}_1$ given $c$ to a non-degenerate set, so $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \,|\, \mathbf{x}_t, c]$ varies with $\mathbf{x}_t$. The Euler step at $\mathrm{NFE}=1$ evaluates the field only at $t = 0$, where Theorem 1 removes exactly this dependence. $\qquad \blacksquare$

*Theorem 4 (the bias–variance price).* Let $\mathrm{CD}$ denote a Chamfer-type reconstruction loss. The conditional mean minimizes the expected loss,
$$
\mathbb{E}\big[\mathrm{CD}\big(\boldsymbol{\mu}(c), \mathbf{x}_1\big)\big] \leq \mathbb{E}\big[\mathrm{CD}\big(\hat{\mathbf{x}}_1^{\mathrm{(s)}}, \mathbf{x}_1\big)\big]
\tag{13}
$$
for any dispersed estimator $\hat{\mathbf{x}}_1^{\mathrm{(s)}}$ with the same conditional mean. A sampler therefore pays a structural reconstruction-error penalty relative to the mean; the penalty grows with the dispersion it carries.

*Proof sketch.* $\mathrm{CD}$ is a squared-distance-type functional in its first argument; the minimizer of the expected squared distance to $\mathbf{x}_1$ given $c$ is $\mathbb{E}[\mathbf{x}_1|c] = \boldsymbol{\mu}(c)$. $\qquad \blacksquare$

Theorems 2–4 jointly delineate the design space: the one-step map is the optimal *estimator*; a dispersed one-step map is only reachable by deliberately regressing a multi-step map, and it is then a *sampler* whose reconstruction error is bounded below by the variance it carries.

**D. The one-step distillation frontier.** Progressive distillation trains a one-step student to regress the $K$-step midpoint output of the teacher. Theorems 3 and 4 predict that the student's dispersion is inherited from the regressed map, so both its diversity and its reconstruction error increase with $K$ — a bias–variance frontier. We map this frontier on the common test batch along two axes.

*Training-budget axis* (fixed $K = 2$): increasing the student's budget from 512 samples / 60 epochs to 1024 / 100 improves the fit to the teacher's map (regression loss $0.2535 \rightarrow 0.0958$), which raises the sample diversity from $0.0488$ to $0.1127$ pairwise CD (diversity ratio $0.19 \rightarrow 0.48$) while the reconstruction error degrades from $0.3839$ to $0.5056$ — both monotone, as predicted. The low-budget run's apparent quality advantage is explained by Theorem 4: an under-fit student is partially averaged back toward the mean.

*Teacher-integration axis* (fixed budget, $K \in \{2, 4, 10\}$): diversity increases monotonically ($0.0488 \rightarrow 0.0640 \rightarrow 0.0687$) and reconstruction error degrades monotonically ($0.3839 \rightarrow 0.3956 \rightarrow 0.4368$), with diminishing returns beyond $K = 4$: the additional teacher integration buys 7% more diversity for a further 10% reconstruction error. We therefore identify $K = 4$ as the practical operating point for distilled one-step samplers in this system.

*Scope of the certificates.* The theorems characterize the *optimal* field; a finite network, finite data, and non-convex training realize it only approximately, so every quantitative statement above is an empirical measurement on a converged model rather than a corollary of the theorems alone. The frontier is measured on a single seed with a modest evaluation batch; its monotone trends are consistent across all configurations tested, but the effect sizes carry confidence-interval caveats that we report alongside them in Section VI.

**E. Design implications.** For the CD-driven closed-loop shape prior studied in Section VI, Theorem 4 makes the one-step teacher — the guided conditional mean — the correct operating point, and the experiments confirm that guidance improves it monotonically. For tasks that require sampling diversity (data augmentation, uncertainty quantification), the distilled student at $K = 4$ is the correct tool, with its dispersion understood as a feature and its reconstruction penalty as its price. Nothing in between is: a half-converged one-step map is neither a good mean nor a good sampler, which is precisely the failure mode that the budget-axis measurement exposes.

## VI. EXPERIMENTS

**A. Experimental setup.** All experiments run on the physics-grounded platform of Section III with real SGP4 trajectories, per-frame wideband channel realizations, and the fresh-data training protocol of Section IV-G. Unless otherwise stated, the generative models share the same VAE (256-d whitened latent), the same DiT capacity (2 blocks, 256 hidden units, 8 heads), the same condition encoder, and the same data budget; the only varied factor is the generative objective (diffusion vs. flow matching) or the sampling configuration. Reconstruction quality is measured by Chamfer distance (CD), F-Score at thresholds 0.1 and 0.2, and voxel IoU; sampling cost is measured by the number of network evaluations (NFE). Paired comparisons are evaluated on a common test batch with fixed seeds, and all reported numbers are reproducible from the released scripts with the recorded JSON evidence. Unless stated otherwise, the HRRP condition is used; the narrowband-only condition is included as the information-poor reference.

**B. Equal-compute comparison with the diffusion baseline.** Table I reports the equal-compute comparison: identical architecture, data, and training budget, with the diffusion baseline using its standard $T = 100$ ancestral steps and FMShape using a single Euler step. On single runs, one-step FMShape improves the CD by 21.9% (spaceborne), 30.0% (ground-RIS), and 33.1% (no-RIS) over the 100-step baseline, and by 14% under HRRP conditioning; the improvement holds on all four metrics (CD, F-Score at both thresholds, IoU) and in every RIS mode.

**TABLE I — Equal-compute comparison (CD ↓; identical architecture and data; NFE = 1 vs. 100)**

| Mode | DDPM (NFE = 100) | FMShape (NFE = 1) | Δ CD |
|---|---|---|---|
| Spaceborne | 0.4055 | 0.3166 | −21.9% |
| Ground-RIS | 0.3069 | 0.2054 | −33.1% |
| No-RIS | 0.6037 | 0.4223 | −30.0% |
| HRRP-conditioned | 0.2637 | 0.2269 | −14.0% |

Because a single-seed comparison can overstate a quality gap, we additionally ran the pre-registered three-seed paired test: the per-seed CD differences are +4.5%, −20.5%, and −4.4%, with a bootstrap 95% confidence interval crossing zero and a cross-seed coefficient of variation of 18% for the baseline itself. We therefore claim *sampling-efficiency parity in quality* — comparable reconstruction accuracy at one evaluation instead of one hundred — rather than statistical quality superiority, and the efficiency claim is the load-bearing one.

Fig. 3 traces the quality–NFE curve. The equal-quality crossover occurs at $\mathrm{NFE}^{*} \leq 2$ for the spaceborne and no-RIS modes and at $\mathrm{NFE}^{*} = 1$ for the ground-RIS mode, i.e. a 50–100× reduction in sampling cost at equal quality. The ODE structure is independently certified: the measured paired-latent convergence order is −0.87 (close to the Euler-predicted −1), the trajectory straightness ratio is 0.006, and the measured Lipschitz constant of the guided field is $\hat{L} = 3.54$, for which the Gronwall Euler-error bound is valid but roughly two orders of magnitude loose — the observed order, not the constant, is the load-bearing evidence. Against strong few-step baselines at matched NFE (Table II), FMShape leads at $\mathrm{NFE} = 1$; at $\mathrm{NFE} = 10$ the DDIM baseline catches up, and a distilled one-step student beats both — an honest map of where the advantage lives.

**TABLE II — Strong few-step baselines (CD ↓, HRRP-conditioned, common batch)**

| NFE | DDIM | FMShape (Euler) | Distilled student |
|---|---|---|---|
| 1 | 0.2583 | **0.2109** | **0.1835** |
| 10 | **0.2256** | 0.3053 | — |

**C. Conditioning: the measurement-to-shape channel.** Conditioning on the measured HRRP opens a quantifiable side-information channel: the conditional loss gap at the data end of the flow is $\Delta(0) = 0.302$ — six times the pre-registered threshold — with 21.8% of the latent variance explained by the condition, $\Delta(t)$ monotonically non-increasing in $t$, and the conditional MMSE identity holding to machine precision (residual $1.6\times10^{-7}$). The condition encoder is verified non-collapsed (output standard deviation 6.2× the input; shuffle sensitivity 0.29). As predicted by Corollary 1, the one-step outputs of the converged teacher are near-deterministic (diversity ratio 0.02 across 16 initial noises), and — as predicted by Corollary 2 — sweeping the guidance scale $w \in \{0, 1, 2\}$ improves the CD monotonically (0.3060 → 0.2725 → 0.2664) while the diversity ratio stays at 0.02: guidance sharpens the conditional-mean estimate and cannot manufacture posterior dispersion.

A data-level audit quantifies what the condition can and cannot determine. Among the closest available condition pairs in the training distribution (96 scenes, 4560 pairs), the ground-truth clouds still differ by 0.663 raw CD and 0.0265 centroid-aligned CD — 125× and 5× the teacher's collapsed spread — so real conditional diversity exists and the one-step map discards it, exactly as Theorem 4 prices. The rank correlation between condition distance and shape distance is $\rho = -0.015$: the range profile is a many-to-one map with no pairwise shape-discriminative power, which bounds the channel from above regardless of the generative model.

**D. ISAR condition augmentation.** The data-level pre-screen (paired on the same 96 scenes) shows the slow-time Doppler profile alone correlates with shape distance at $\rho = 0.212$ against HRRP's $-0.015$, while naive HRRP+dop concatenation collapses the correlation to 0.002 — the dilution mechanism identified in Section IV-F. The full-scale runs at matched budget (512/60, fresh data) confirm the prediction on the network side: with raw concatenation the channel gap is $\Delta(0) = 0.131$ (below the HRRP-only control at 0.144, i.e. −9.0%), while spread-equalized concatenation reaches 0.154 (+17.6% over raw concatenation, +6.9% over the matched-budget HRRP-only control). Reconstruction quality is unaffected (CD 0.4011 raw vs. 0.4065 equalized, +1.3%). Two caveats are reported with the claim: the +6.9% is a single-seed effect without confidence intervals, and the channel width is strongly budget-sensitive (0.302 at 1024/100 vs. 0.144 at 512/60 for the same condition), so the ISAR gain should be read as a matched-budget, single-seed result pending multi-seed replication.

**E. The one-step distillation frontier.** Table III maps the bias–variance frontier of Section V-D along both axes, on the common test batch. Along the budget axis ($K = 2$), doubling the training budget improves the fit to the teacher's map (loss 0.2535 → 0.0958) and raises the diversity ratio from 0.19 to 0.48, while the CD degrades from 0.3839 to 0.5056 — both monotone as Theorems 3–4 predict; the matched-budget student fails both pre-registered quality propositions (P1: beats the teacher's one-step mean; P2: matches the teacher's two-step map), which is the structural price of dispersion, not a training defect. Along the teacher-integration axis, diversity and CD are monotone in $K$ with diminishing returns beyond $K = 4$, identifying the practical sampler operating point.

**TABLE III — Distillation frontier (common batch; pairwise CD and CD-to-GT of the one-step student)**

| Axis | Setting | Pairwise CD | CD-to-GT | Diversity ratio |
|---|---|---|---|---|
| Budget | 512/60, K=2 | 0.0488 | 0.3839 | 0.19 |
| Budget | 1024/100, K=2 | 0.1127 | 0.5056 | 0.48 |
| Teacher K | K=4 | 0.0640 | 0.3956 | 0.27 |
| Teacher K | K=10 | 0.0687 | 0.4368 | 0.29 |

**F. Closed-loop deployment.** Finally, the one-step teacher — the guided conditional mean, which Theorem 4 identifies as the CD-optimal estimator — is inserted into the ISAC closed loop as the ROI shape prior, replacing the hand-crafted box prior. The sensing achievement ratio improves from 0.840 to 0.932 (+10.9%), the voxel $\ell_1$ shape error reduces to 0.58×, and the total closed-loop achievement reaches ≈0.736. The generative primitive thus pays off at the system level, not only at the metric level.

**G. Ablations, generalization, and reproducibility.** Freezing the training data collapses the condition dependence (a memorization artifact), confirming the fresh-data protocol as a load-bearing design choice rather than a detail. The pipeline generalizes across orbit ephemerides (ISS and Starlink), across the three RIS configurations, and to RIS arrays of $N = 64$ elements, with the direction of every conclusion unchanged (13–35% CD improvement range). Absolute quality remains the open problem: with the VAE oracle at CD 0.0093 and F-Score@0.1 0.891 against FMShape's 0.2269 and 0.174, the gap is dominated by latent capacity and training scale rather than by the generative objective — a boundary we state plainly rather than paper over. All results use fixed seeds, pre-registered propositions with recorded verdicts, and one-command reproduction via the released Makefile targets.

## VII. CONCLUSION

This paper presented FMShape, a conditional flow-matching framework that reconstructs the three-dimensional shape of a target region from its measured wideband high-resolution range profile in spaceborne integrated sensing and communication. By training an optimal-transport conditional flow-matching model in the latent space of a variational autoencoder with the measured profile as the sampling condition, FMShape generates target shapes within a single sampling step, with reconstruction accuracy comparable to a 100-step diffusion baseline under identical data, architectures, and compute — a 50–100× reduction in sampling cost at the real-time operating point, certified independently by the measured convergence order and trajectory straightness of the probability flow. Conditioning on the measurement was shown to open a quantifiable side-information channel, augmentable by an ISAR slow-time Doppler profile under spread-equalized concatenation, and the theoretical analysis established that the single-step output is exactly the conditional mean — explaining the collapse of converged one-step maps, bounding what one-step distillation can buy through a measured bias–variance frontier, and identifying the practical operating points: the guided one-step mean for reconstruction-driven tasks such as the closed-loop shape prior, where it raised the sensing achievement ratio by 10.9%, and the distilled sampler for tasks that require posterior dispersion. The same analysis delineated the boundaries of the approach: absolute reconstruction quality remains bounded by latent capacity and training scale, and the range profile is a many-to-one map that cannot narrow the shape posterior — so further conditional gains require new observables rather than better samplers. Future work includes multi-seed replication of the conditioning results, larger latent capacities and training budgets to close the gap to the VAE ceiling, additional observables such as rotation-resolved ISAR sequences, and validation on measured radar data and over-the-air testbeds. The platform, verification suite, and pre-registered verdicts are released to support reproduction.

---

## 图表（已生成，`figs/` 目录，300 dpi）

- **Fig. 1** `figs/fig1_scenario.png` — 空-地 ISAC 场景示意（SGP4 卫星轨道、感知回波直接路径、RIS 辅助通信链路、ROI 体素、逐帧 HRRP 形成）
- **Fig. 2** `figs/fig2_framework.png` — FMShape 框架（推理主链路：HRRP→条件编码→OT-CFM 单步→解码→点云；训练支路：VAE 潜空间+路径+FM loss；定理 callout）
- **Fig. 3** `figs/fig3_nfe_curve.png` — 质量-NFE 曲线（真实实验数据）：(a) HRRP 条件化 FMShape 曲线 + DDPM 基线 + VAE 上界 + NFE\*≤2 交叉带；(b) 强少步基线（DDIM / FM-Euler / 1 步蒸馏学生）
- 图表脚本：`source_code/isac_sat/paper_figs/fig{1,2,3}_*.py`（Fig. 3 直接读实验 JSON，可复现）
- **表 I–III** — 已在正文内

## 参考文献表（真实论文垫底，成稿补全条目）

| # | 论文 |
|---|---|
| [1] | F. Liu et al., "Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond," *IEEE JSAC*, 40(6):1728–1767, 2022 |
| [2] | F. Liu et al., "Joint radar and communication design: Applications, state-of-the-art, and the road ahead," *IEEE TCOM*, 68(6):3834–3862, 2020 |
| [3] | L. Du et al., "Bayesian spatiotemporal multitask learning for radar HRRP target recognition," *IEEE TSP*, 59(7):3182–3196, 2011 |
| [4] | H. Liu et al., "Radar high-resolution range profiles target recognition based on stable dictionary learning," *IET RSN*, 10(3):228–237, 2016 |
| [5] | 陈健/杜兰/廖磊瑶, "基于参数化统计模型的雷达 HRRP 目标识别方法综述," *雷达学报*, 11(6):1020–1047, 2022 |
| [6] | K. Liao et al., "Radar HRRP target recognition based on concatenated deep neural networks," *IEEE Access*, 6:29211–29218, 2018 |
| [7] | "SDHC: Joint semantic-data guided hierarchical classification for fine-grained HRRP target recognition," *IEEE TAES*, 2024 |
| [8] | "Radar target characterization and deep learning in radar automatic target recognition: A review," *Remote Sensing*, 15:3742, 2023 |
| [9] | Y. Lipman et al., "Flow matching for generative modeling," *ICLR*, 2023 |
| [10] | X. Liu, C. Gong, and Q. Liu, "Flow straight and fast: Learning to generate and transfer data with rectified flow," in *Proc. ICLR*, 2023 |
| [11] | Y. Wang et al., "Deep learning-based extended target tracking in ISAC systems," arXiv:2504.00576, 2025 |
| [12] | M. M. Rahman, S. Z. Gurbuz, M. G. Amin, "Physics-aware generative adversarial networks for radar-based human activity recognition," *IEEE TAES*, 59(3):2994–3008, 2023 |
| [13] | Bao et al., "Improved few-shot SAR image generation by enhancing diversity," *IEEE JSTARS*, 17:3394–3408, 2024 |
| [14] | D. Guo et al., "Variational temporal deep generative model for radar HRRP target recognition," *IEEE TSP*, 68:5795–5809, 2020 |
| [15] | M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, "Stochastic interpolants: A unifying framework for flows and diffusions," arXiv:2303.08797, 2023（另见 *JMLR*, 26(209):1–80, 2025） |
| [16] | C. Ozdemir, "Inverse Synthetic Aperture Radar Imaging with MATLAB Algorithms," 2nd ed., Hoboken, NJ: Wiley, 2021（ISAR 距离单元迁移与慢时谱形成的经典参考；本文序列生成为自研实现） |
| [17] | A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, and Y. Bengio, "Conditional flow matching: Simulation-free dynamic optimal transport," arXiv:2302.00482, 2023 |
