# Pure Comparison Audit

## Conclusion

`Pix2Pix_CurrentSplit_E100` and `UNet_CurrentSplit_E100` are not copied GT predictions and are not copied T1 images. Their train/val/test subjects are disjoint:

- train: 173 subjects / 8650 slices
- val: 37 subjects / 1850 slices
- test: 38 subjects

Their apparently strong PSNR is explainable:

- U-Net is a strong regression baseline and tends toward smooth conditional-mean outputs.
- Pix2Pix uses standard `lambda_l1=100`, so it is also strongly constrained by paired L1 regression.
- Their sharpness is limited: U-Net SharpRatio is 0.456, Pix2Pix SharpRatio is 0.641.

The results that must be removed from pure comparison tables are:

- `UNet_E99`: PSNR=36.99, suspicious legacy/unverified result.
- `Pix2Pix_E100`: legacy/unverified split.
- `CycleGAN_E100`: legacy CycleGAN; replace with `CycleGAN_CurrentSplit_E100` when finished.
- `DDIM_E100_K50`: legacy or unverified split; retrain before pure comparison.
- `FREQ_FLOWBASE_PMLOW_B035`: our method, not a pure baseline.
- `PM_STAGE1`, `PM_STAGE1_LPIPS_GAN`, `PM_DIRF_FIDELITY_FLOW`: our ablations, not external baselines.

## Clean Table Script

Pure baselines only:

```powershell
python scripts/build_pure_comparison_tables.py `
  --output_root outputs/icdm2026/tables/pure_comparison_clean
```

Baselines plus ours/references:

```powershell
python scripts/build_pure_comparison_tables.py `
  --include_ours `
  --include_references `
  --output_root outputs/icdm2026/tables/pure_comparison_with_ours
```

## Current Pure Baseline Image Metrics

| Method | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net current split | 28.236 | 0.897 | 0.001606 | 0.017834 | 0.456 | 0.057 | 0.836 |
| Pix2Pix current split | 28.114 | 0.904 | 0.001641 | 0.017847 | 0.641 | 0.058 | 0.871 |
| CycleGAN current split | pending | pending | pending | pending | pending | pending | pending |
| DIRF_V5 K6 | 28.399 | 0.909 | 0.001531 | 0.017188 | 0.455 | 0.054 | 0.884 |

## Current Pure Baseline Downstream

Task: `CN+SCD vs MCI+AD`

| Method | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| U-Net current split | 0.532 | 0.536 | 0.568 |
| Pix2Pix current split | 0.568 | 0.571 | 0.562 |
| CycleGAN current split | pending | pending | pending |
| DIRF_V5 K6 | 0.460 | 0.461 | 0.374 |

## With Our Method

`FREQ_FLOWBASE_B035` is the current main candidate, not a pure baseline.

| Method | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| FREQ_FLOWBASE_B035 | 0.598 | 0.604 | 0.571 |

Image metrics:

| Method | PSNR | SSIM | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|
| FREQ_FLOWBASE_B035 | 28.180 | 0.902 | 0.842 | 0.057 | 0.850 |
