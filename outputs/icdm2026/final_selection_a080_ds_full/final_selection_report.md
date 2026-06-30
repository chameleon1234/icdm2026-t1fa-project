# ADNI A080+DS Full Final Unified Validation

## Current Decision

Recommended main method: **A080+DS Full (Ours)**.

It is not the winner on every isolated downstream task, but it is the most balanced final candidate: highest PSNR/SSIM, lower WM-MAE than A080 base and Pix2Pix, SharpRatio close to 1, and ROI-CCC improved from 0.768 to 0.842 over A080 base while preserving acceptable downstream utility. Old Fidelity Flow remains a strong ROI-consistency reference, but A080+DS full is stronger for the final combined image-quality and medical-consistency story.

## Summary Table

| Method | PSNR | SSIM | WM-MAE | ROI-CCC | Sharp | MIL-F1 | TestCV-F1 | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA_GT | - | 1.000 | 0.000 | 1.000 | 1.000 | 0.582 | 0.607 | - |
| A080+DS Full (Ours) | 28.816 | 0.914 | 0.055 | 0.842 | 1.023 | 0.594 | 0.515 | 0.921 |
| Old Fidelity Flow | 28.092 | 0.905 | 0.057 | 0.867 | 0.889 | 0.634 | 0.506 | 0.871 |
| Pix2Pix | 28.011 | 0.901 | 0.061 | 0.827 | 0.913 | - | 0.517 | 0.783 |
| A080 Base | 28.371 | 0.909 | 0.061 | 0.768 | 0.996 | 0.556 | 0.478 | 0.740 |
| U-Net | 28.444 | 0.907 | 0.058 | 0.812 | 0.519 | - | 0.493 | 0.696 |
| CycleGAN | 26.256 | 0.871 | 0.078 | 0.691 | 0.877 | - | 0.453 | 0.380 |
| DDIM | 25.783 | 0.862 | 0.088 | 0.580 | 0.569 | - | 0.512 | 0.174 |
| DBM | 25.880 | 0.877 | 0.093 | 0.506 | 0.489 | - | 0.527 | 0.149 |

## New Outputs

- A080+DS full train/test MIL evaluation: `outputs/icdm2026/downstream_adni_a080_ds_full_e12_final/classification_subject_summary.csv`
- ADNI all-method test-CV downstream evaluation: `outputs/icdm2026/downstream_adni_final_testcv_all_methods/classification_repeated_summary.csv`
- Final visual panels: `outputs/icdm2026/figures/adni_final_a080_ds_panels`
- Combined CSV: `outputs/icdm2026/final_selection_a080_ds_full/final_selection_metrics.csv`
