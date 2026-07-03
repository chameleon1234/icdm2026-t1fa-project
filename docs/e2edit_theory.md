# E2EDiT Theory Design

E2EDiT stands for **End-to-End frequency-Decoupled Disease-informed Texture-preserving T1-to-FA Synthesis**.

## Motivation

T1-to-FA synthesis is not a standard image-to-image translation task. A single T1 slice provides anatomical structure and partial white-matter cues, but it does not fully contain the microscopic diffusion texture measured by FA. Therefore, a model optimized only for PSNR/SSIM can collapse to a posterior-mean solution, whereas a model optimized only for visual sharpness may hallucinate unsupported texture.

Prior explored routes have clear limitations:

- Posterior-mean / MSE models are stable but blurry.
- LPIPS / GAN models can be sharp but may create over-bright white-matter artifacts and fake texture.
- Direct T1-start MeanFlow / K25 models produce strong high-frequency content but can preserve T1-like texture and unstable medical fidelity.
- PriorFlow K8/K25 has a template/prior shortcut risk.
- The previous FADiT / A080+DS pipeline depends on offline base/detail prediction folders; its detail source depends on PriorFlow, so it is not an end-to-end final method.
- ReFA-Flow is theoretically meaningful but failed the smoke anti-shortcut validation.

## Core Hypothesis

A reliable T1-to-FA synthesis model should decouple low-frequency FA structure and high-frequency FA detail within an end-to-end network, while using a bounded disease-informed corrector to improve medical consistency without hallucinating unsupported texture.

## Model Formulation

Stage 1 predicts two branches from the current T1 slice:

\[
B_i = B(T1_i)
\]

\[
H_i = H(T1_i)
\]

\[
P_i = clamp(B_i + H_i, -1, 1)
\]

\(B_i\) is a low-frequency FA base responsible for global FA structure and intensity space. \(H_i\) is a high-frequency detail branch responsible for FA residual texture.

Stage 2 predicts a bounded correction:

\[
\Delta_i = C(T1_i, P_i, LP(P_i), HP(P_i), Edge(T1_i), WMproxy_i, Coord)
\]

\[
FA_i^{final} = clamp(P_i + \Delta_i, -1, 1)
\]

The corrector is not a full generator. It applies small corrections to improve fidelity and medical consistency while preserving the Stage1 texture.

## Smoke Ablations

The smoke test includes four models:

- M0: single-branch T1-to-FA baseline.
- M1: naive dual-branch \(B(T1)+H(T1)\).
- M2: dual-branch with anti-mean low/high frequency constraints.
- M3: M2 plus bounded corrector.

M2 must improve or at least not degrade against M0 while showing meaningful low/high branch separation. M3 must improve medical and reconstruction metrics over M2 without destroying texture. If either condition fails, full-data training is not allowed.

## Explicit Exclusions

The first E2EDiT smoke does not use A080, PriorFlow, template sources, offline blend folders, GAN, LPIPS, disease labels as inference inputs, or test-set tuning.
