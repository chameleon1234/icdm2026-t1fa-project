# ADNI Dual-Dataset Experiment Design

## Goal

Use the private 248-subject dataset for method development and ADNI as the public reproducibility dataset for final comparison, ablation, and downstream data-mining validation.

## Data Facts From Inspection

- Archive: `data/ADNI_data.7z`
- Internal folders:
  - `ADNI_data/T1/*.nii.gz`
  - `ADNI_data/FA/*.nii.gz`
- Image space: `MNI152NLin6Asym`, `res-02`
- Volume shape from samples: `91 x 109 x 91`
- Resolution: `2 mm`
- Files:
  - T1: 542
  - FA: 538
  - Paired T1-FA subjects: 538
  - T1-only subjects: 4
- Label file: `data/subject_group_cleaned_filtered.csv`
- Labeled subjects: 405
- Paired labeled group counts:
  - CN: 214
  - MCI: 113
  - AD: 25
  - EMCI: 24
  - LMCI: 13
  - `_S_MC`: 16
- Paired unlabeled subjects: 133

## Label Policy

For generation, use all paired T1-FA subjects, including unlabeled ADNI subjects.

For downstream classification:

- Keep `CN` as normal control.
- Map `MCI`, `EMCI`, and `LMCI` to `MCI_spectrum`.
- Keep `AD`.
- Exclude `_S_MC` from downstream tasks until its meaning is verified.
- Exclude unlabeled subjects from downstream tasks, but keep them in generation train/val/test splits.

## Split Policy

All splits are subject-level, never slice-level.

The first split manifest will use:

- train: 70%
- val: 10%
- test: 20%
- stratification:
  - labeled subjects stratified by normalized group
  - unlabeled paired subjects split separately as `UNLABELED`

This gives the generation model the largest paired training set while preserving a labeled public test set for disease-utility evaluation.

## Evaluation Strategy

The ADNI public dataset should be used to show larger and more stable differences than the private 38-subject test set.

Image fidelity:

- PSNR
- SSIM
- MSE
- MAE

Detail and anatomy:

- SharpRatio
- Gradient error
- WM-MAE
- ROI-MAE / ROI-CCC

Downstream utility:

- CN vs AD
- CN vs MCI_spectrum
- CN vs MCI_spectrum+AD
- MCI_spectrum vs AD
- three-class CN / MCI_spectrum / AD as an auxiliary task

Incremental information tests:

- T1 only
- synthetic FA only
- T1 + synthetic FA
- real FA only
- T1 + real FA

## Main Paper Framing

The private dataset remains the method-development dataset. ADNI becomes the public validation dataset. A method is considered strong only if it is balanced across:

- paired image fidelity
- visual/white-matter detail
- ROI consistency
- downstream disease utility

This avoids overclaiming from small private-set PSNR/SSIM differences and gives the ICDM paper a stronger data-mining story.
