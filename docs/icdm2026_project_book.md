# ICDM 2026 T1-to-FA Project Book

## 1. Project Identity

**Working title:** Mining Missing Diffusion Biomarkers from Structural MRI: Posterior-Mean Rectified Flow for T1-to-FA Synthesis and Alzheimer's Staging

**Target venue:** IEEE ICDM 2026 Applied Data Science Track.

**Target deadline:** ICDM 2026 Applied Track lists paper submission deadline as 2026-06-06 AoE. In Beijing time this is 2026-06-07 19:59:59, but the internal deadline should be 2026-06-05 for a complete paper, figures, and reproducibility check.

**Target links:**

- ICDM 2026 main page: https://icdm2026.neu.edu.cn/main.htm
- ICDM 2026 Applied Track CFP: https://icdm2026.neu.edu.cn/CallforAppliedTrackPapers/list.htm

## 2. One-Sentence Goal

Build a deterministic T1-to-FA synthesis and data mining framework that reconstructs voxel-aligned FA maps from routine T1 MRI and verifies whether synthesized FA preserves Alzheimer's disease staging signals.

## 3. Positioning Shift

The current paper is mostly a medical image synthesis paper. For ICDM Applied Track, the story should become an applied data mining paper:

> Missing diffusion biomarker mining from structural MRI for Alzheimer's staging.

This framing keeps the existing T1-to-FA generation work, but makes the contribution broader and more conference-aligned:

- **Data mining problem:** recover missing diffusion-derived white-matter biomarkers from widely available structural MRI.
- **Applied domain:** Alzheimer's spectrum staging, using CN, SCD, MCI, and AD labels.
- **Methodological contribution:** posterior-mean guided deterministic rectified flow for paired cross-modality biomarker synthesis.
- **Utility validation:** downstream diagnosis/classification and brain-region analysis, not only image fidelity.

## 4. Dataset Specification

**Dataset root:** `data/`

**Raw image data:** `data/ZHU_T1_and_FA_space-MNI152NLin6Asym_res-02`

**Processed slices:** `data/processed`

**Subject split:** `data/processed/dataset_splits.json`

- Train: 173 subjects, 8650 slices
- Validation: 37 subjects, 1850 slices
- Test: 38 subjects, 1900 slices
- Each subject currently contributes 50 axial slices

**Clinical label file:** `data/data_information.xlsx`

Use sheet `re_order` as the subject-level label source.

**Confirmed group mapping:**

- `1 = CN`
- `2 = SCD`
- `3 = MCI`
- `4 = AD`

**Label counts from Excel:**

- CN: 92
- SCD: 56
- MCI: 70
- AD: 30
- Total: 248

**Clinical covariates available:**

- `groups`
- `gender`
- `age`
- `edu`
- `MMSE`

**Important rule:** all train/validation/test operations must be subject-level, never slice-level random splitting. Slice-level splitting would leak subject identity.

## 5. Current Assets

**Main first-version experiment:**

- `train_pure_meanflow_v2_multi_loss.py`
- Current idea: deterministic image-to-image rectified flow, trained on straight T1-to-FA bridge.
- Current issue: the paper still over-emphasizes FID/KID, which is weak for quantitative FA maps.

**Second-version PMRF work:**

- `pmrf_t1fa/models/pmrf_t1fa.py`
- `pmrf_t1fa/train_pmrf_t1fa_stage1.py`
- `pmrf_t1fa/train_pmrf_t1fa_stage2.py`
- `pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py`

This code already implements a useful direction:

- Stage 1: Restormer-style posterior-mean/coarse FA predictor.
- Stage 2: refinement rectified flow from coarse FA to target FA.
- Evaluation already includes PSNR, SSIM, MSE, MAE, MS-SSIM, brain-masked MAE, white-matter masked MAE, ROI concordance, gradient error, and white-matter histogram distance.

**Existing paper draft:**

- `papers/IEEE_TMI (7).pdf`
- `main.tex`
- `_paper_text.txt`

**Relevant local references:**

- `papers/PMRF.pdf`
- `papers/相同任务/Zhang 等 - 2025 - Diffusion Bridge Models for 3D Medical Image Translation.pdf`
- `papers/相同任务/Yazdani 等 - 2025 - Flow Matching for Medical Image Synthesis Bridging the Gap Between Speed and Quality.pdf`
- `papers/相同任务/Kwon 等 - 2026 - Generative Synthesis of Fractional Anisotropy Maps from T1 MRI Using Transfer Learning for White Mat.pdf`
- `papers/方法设计/Liu 等 - 2026 - DIReCT Domain-Informed Rectified Flow for Controllable Brain MRI to PET Translation.pdf`
- `papers/方法设计/Susladkar 等 - 2025 - ViCTr Vital Consistency Transfer for Pathology Aware Image Synthesis.pdf`

## 6. Main Problems To Fix

### Problem A: Metric mismatch

FID is designed for natural image distribution comparison and relies on Inception features. It is not the right main metric for a scalar quantitative medical map such as FA.

**Decision:** main quantitative table must use:

- PSNR
- SSIM
- MSE
- MAE

Useful medical/white-matter metrics should be added:

- Brain-masked MAE
- White-matter masked MAE
- ROI concordance correlation coefficient
- Gradient error
- White-matter histogram Wasserstein distance

FID/KID can be removed from the main paper or moved to supplementary analysis only.

### Problem B: ICDM fit is weak without a downstream task

Pure synthesis is less aligned with data mining than synthesis plus data utility mining.

**Decision:** add downstream Alzheimer's staging experiments:

- 4-class classification: CN vs SCD vs MCI vs AD.
- Binary diagnostic variants:
  - CN vs AD
  - CN vs MCI+AD
  - CN+SCD vs MCI+AD
- Optional ordinal staging analysis: CN -> SCD -> MCI -> AD.

### Problem C: PSNR/SSIM gaps are small

A small PSNR/SSIM gain is common in paired medical image synthesis. The paper should not depend on only a tiny PSNR/SSIM margin.

**Decision:** strengthen the claim by triangulating:

- Paired fidelity metrics.
- White-matter ROI preservation.
- Downstream classification utility.
- Visual case analysis around AD-sensitive white-matter regions.
- Inference speed and deterministic reproducibility.

### Problem D: Dataset labels are not yet integrated into dataset loading

`src/datasets.py` currently returns slice pairs and filenames, but not subject labels or clinical covariates.

**Decision:** build a subject index layer that maps each slice to:

- `subject_id`
- `slice_id`
- `group_id`
- `group_name`
- `age`
- `gender`
- `edu`
- `MMSE`
- `split`

## 7. Proposed Method

### Name

Use a name that naturally bridges DIRF and PMRF:

**PM-DIRF: Posterior-Mean Guided Deterministic Image-to-Image Rectified Flow**

### Architecture

PM-DIRF has two stages:

**Stage 1: Posterior-mean FA predictor**

- Input: T1 slice.
- Output: coarse FA prediction.
- Backbone: Restormer-style encoder-decoder from `pmrf_t1fa/models/pmrf_t1fa.py`.
- Loss: MSE + L1 + SSIM + gradient + high-frequency loss.
- Role: minimize paired reconstruction error and provide stable posterior-mean estimate.

**Stage 2: deterministic refinement rectified flow**

- Source: Stage 1 coarse FA.
- Target: ground-truth FA.
- Optional condition: coarse FA or T1/coarse concatenation.
- Flow state: interpolation from source to target.
- Loss: velocity MSE + endpoint MSE/L1/SSIM + gradient/high-frequency/detail loss.
- Inference: deterministic Euler, default 1 step.

### Why This Is Explainable

- Stage 1 estimates the conditional mean FA from T1, which is appropriate for low MSE.
- Stage 2 refines the posterior-mean estimate toward sharper target FA while remaining deterministic.
- The bridge is source-to-target in paired image space, avoiding pure Gaussian corruption.
- White-matter metrics and downstream classification test whether the synthesis preserves disease-relevant structure.

## 8. Baselines

### Must keep

- CNN U-Net
- Pix2Pix
- CycleGAN
- DDIM
- Diffusion Bridge Model
- Restormer direct image-to-image
- DIRF first version from `train_pure_meanflow_v2_multi_loss.py`

### Add if time permits

- NAFNet image-to-image baseline, because the repo already has `train_nafnet_meanflow.py` and related logs.
- Swin/UMamba baselines if previous results are stable and reproducible.

### Related work to cite even if not rerun

- PMRF, ICLR 2025: posterior-mean rectified flow for restoration.
- Diffusion Bridge Models for 3D Medical Image Translation, 2025: close medical image translation baseline and downstream task motivation.
- Flow Matching for Medical Image Synthesis, 2025: speed-quality medical flow matching context.
- DIReCT, 2026: domain-informed rectified flow for brain modality translation.
- ViCTr, 2025: pathology-aware consistency in medical synthesis.
- T1-to-FA synthesis papers from 2024-2026 in `papers/相同任务`.

## 9. Evaluation Design

### Main synthesis metrics

Compute on test set, subject split fixed:

- PSNR, mean and standard deviation.
- SSIM, mean and standard deviation.
- MSE, mean and standard deviation.
- MAE, mean and standard deviation.

### Medical preservation metrics

- Brain-masked MAE.
- White-matter masked MAE.
- Gradient error.
- ROI-CCC over white-matter ROIs.
- White-matter histogram Wasserstein distance.

### Efficiency metrics

- Inference latency per slice.
- Number of function evaluations.
- Determinism check: repeated inference with same input and checkpoint must produce identical outputs within numerical tolerance.

### Downstream classification metrics

Subject-level classification only.

Inputs to compare:

- T1 only.
- Real FA only.
- Synthetic FA only.
- T1 + synthetic FA.
- T1 + real FA.

Metrics:

- Balanced accuracy.
- Macro-F1.
- AUROC for binary tasks.
- Confusion matrix.
- Bootstrap confidence interval over test subjects.

### Statistical tests

Use paired testing on subject-level metrics:

- Wilcoxon signed-rank test for PSNR/SSIM/MSE/MAE comparisons between methods.
- Bootstrap confidence intervals for classification metrics.

## 10. Ablation Plan

Run ablations in a way that each row answers one paper question:

| ID | Variant | Question |
| --- | --- | --- |
| A0 | DIRF first version | What does the original deterministic straight bridge achieve? |
| A1 | Stage 1 posterior-mean only | How good is pure posterior-mean estimation? |
| A2 | Stage 1 + Stage 2 flow | Does PMRF-style refinement improve synthesis? |
| A3 | A2 without gradient/HF/detail losses | Do white-matter detail losses matter? |
| A4 | A2 without coarse conditioning | Does conditioning stabilize refinement? |
| A5 | A2 with 1, 2, 4, 8 steps | Is one-step inference sufficient? |
| A6 | A2 plus auxiliary classification loss | Does disease-aware supervision improve downstream utility? |

The auxiliary classification loss is optional and should only be used as a training regularizer. Group label must not be used as an inference-time input.

## 11. Brain-Region And Case Analysis

Focus on AD-relevant white-matter regions rather than vague "lesions":

- Cingulum bundle.
- Corpus callosum.
- Fornix.
- Uncinate fasciculus.
- Superior longitudinal fasciculus.
- Periventricular white matter.
- Hippocampal and parahippocampal adjacent white matter.

Case analysis figure should show:

- T1 input.
- Ground-truth FA.
- Synthetic FA.
- Absolute error map.
- ROI zoom.
- Classification saliency or Grad-CAM.

Case selection:

- One CN case correctly classified.
- One SCD or MCI case where synthetic FA improves classification confidence.
- One AD case with visible white-matter degradation.
- One failure case with explanation.

## 12. Paper Structure

### Abstract

State the applied data mining problem first: missing FA biomarker mining for Alzheimer's staging. Then describe PM-DIRF and report synthesis plus downstream utility.

### Introduction

Motivate:

- FA is clinically useful but DTI is harder to acquire.
- T1 is routine.
- Missing-modality biomarker mining can support large-scale AD studies.
- Current synthesis metrics alone do not prove utility.

### Related Work

Organize into:

- T1-to-FA and structural-to-diffusion synthesis.
- Diffusion/bridge/flow matching for medical image translation.
- Posterior mean and rectified flow restoration.
- Downstream utility evaluation in medical data mining.

### Method

Explain PM-DIRF:

- Subject-level dataset and labels.
- Stage 1 posterior-mean predictor.
- Stage 2 rectified refinement.
- Losses.
- Deterministic inference.
- Downstream classifier.

### Experiments

Include:

- Dataset and preprocessing.
- Baselines.
- Main fidelity results.
- Medical ROI metrics.
- Downstream classification.
- Ablation.
- Case analysis.

### Discussion

Explain:

- Why PSNR/SSIM alone are not enough.
- Why PMRF-style two-stage design helps.
- Why downstream classification provides ICDM relevance.
- Limitations: 2D slices, single dataset, FA only, label imbalance.

## 13. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| AD class has only 30 subjects | Unstable 4-class metrics | Report balanced accuracy, macro-F1, binary variants, bootstrap CI |
| Slice-level classifier leaks subject identity | Invalid downstream claim | Enforce subject-level splits and aggregate slice features per subject |
| PSNR/SSIM gains remain small | Weak main table | Add ROI metrics, downstream utility, determinism and latency |
| Stage 2 worsens Stage 1 | PMRF claim weak | Keep Stage 1 as strong baseline and tune Stage 2 with paired score |
| FID removed from main results | Existing paper claims need rewrite | Reframe as quantitative FA fidelity, keep FID only as supplementary |
| No external validation | Generalization limitation | State clearly and use robust internal subject-level validation |

## 14. Immediate Next Actions

1. Build subject label index from `data/data_information.xlsx`.
2. Update dataset loader to return subject metadata.
3. Create unified metrics script for all methods.
4. Re-evaluate existing checkpoints with PSNR/SSIM/MSE/MAE.
5. Train/evaluate PM-DIRF Stage 2 only after Stage 1 metrics are stable.
6. Add downstream classification on subject-level aggregated features.
7. Generate main tables and figures.
8. Rewrite `main.tex` for ICDM Applied Track.

