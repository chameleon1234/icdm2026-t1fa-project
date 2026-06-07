# Current Method and ADNI Result Audit

## Conclusion

The previous ADNI numbers were not our method results and were not trained comparison-model results. They were only two references:

- `ADNI_T1_ONLY`: using ADNI test T1 slices directly as predicted FA.
- `ADNI_FA_GT`: real FA upper bound.

ADNI U-Net, Pix2Pix, CycleGAN, DIRF/PMRF, and frequency-fusion methods have not been trained yet. The GPU is still occupied by the private-dataset fair CycleGAN retraining job.

## Fixes Applied

1. ADNI subject IDs must not be compressed with the private-dataset `sub-001` logic.
   - Risk: `sub-002_S_0413` could become `sub-002`, mixing multiple ADNI subjects.
   - Fixed: `normalize_subject_id()` now preserves ADNI IDs such as `sub-002_S_0413`.

2. ADNI downstream references must not read private-dataset config paths.
   - Risk: `--include_t1` / `--include_fa_gt` could point to `data/processed/test/...`.
   - Fixed: when `--adni_slice_manifest` is provided, default references point to the manifest parent directory.

3. Image-metric evaluation can now read ADNI slice manifests through `--adni_slice_manifest`.

## Current Private-Dataset Main Method

The current balanced main-method candidate is:

`FREQ_FLOWBASE_PMLOW_B035`

It is not a newly trained single network. It is frequency-fusion inference:

- low source: `PM_STAGE1`
- high/source image: `PM_DIRF_FIDELITY_FLOW_FULL`
- script: `scripts/export_frequency_fusion_predictions.py`
- mode: `sharp_base_low_residual`
- sigma: `1.5`
- low_residual_gain: `0.35`

Formula:

```text
output = high_image + 0.35 * (LP(PM_STAGE1) - LP(high_image))
```

It keeps high-frequency detail from `Fidelity Flow` and injects low-frequency correction from `PM_STAGE1`.

## Private-Dataset Image Metrics

| Method | Type | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DIRF_V5_3SLICE_K6 | flow comparison | 28.399 | 0.909 | 0.001531 | 0.017188 | 0.455 | 0.054 | 0.884 |
| PM_STAGE1 | PMRF posterior mean | 27.972 | 0.903 | 0.001696 | 0.018168 | 0.519 | 0.063 | 0.800 |
| PM_STAGE1_WMROI_DETAIL_5SLICE | Stage1 detail baseline | 27.741 | 0.898 | 0.001771 | 0.018417 | 0.681 | 0.060 | 0.855 |
| PM_STAGE1_LPIPS_GAN_5SLICE_FINAL | sharp Stage1 | 27.562 | 0.891 | 0.001850 | 0.018915 | 0.882 | 0.062 | 0.834 |
| PM_DIRF_FIDELITY_FLOW_FULL | sharp + flow correction | 27.581 | 0.892 | 0.001842 | 0.018896 | 0.869 | 0.059 | 0.867 |
| FREQ_FLOWBASE_PMLOW_B035 | current main candidate | 28.180 | 0.902 | 0.001609 | 0.017632 | 0.842 | 0.057 | 0.850 |
| UNet_CurrentSplit_E100 | fair U-Net | 28.236 | 0.897 | 0.001606 | 0.017834 | 0.456 | 0.057 | 0.836 |
| Pix2Pix_CurrentSplit_E100 | fair Pix2Pix | 28.114 | 0.904 | 0.001641 | 0.017847 | 0.641 | 0.058 | 0.871 |
| CycleGAN_E100 | legacy CycleGAN | 25.004 | 0.858 | 0.003306 | 0.025436 | 1.168 | 0.076 | 0.878 |
| DDIM_E100_K50 | DDIM comparison | 27.479 | 0.888 | 0.001876 | 0.019173 | 0.667 | 0.063 | 0.827 |

## Private-Dataset Main Downstream Task With AUC

Main task: `CN+SCD vs MCI+AD`, implemented as `cn_scd_vs_mci_ad`.

| Method | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| T1_ONLY | 0.526 | 0.526 | 0.452 |
| PM_STAGE1 | 0.604 | 0.603 | 0.609 |
| Stage1_Baseline | 0.491 | 0.493 | 0.377 |
| Stage1_LPIPS_GAN | 0.491 | 0.493 | 0.499 |
| Fidelity_Flow | 0.559 | 0.559 | 0.519 |
| Fidelity_Direct | 0.512 | 0.514 | 0.484 |
| FREQ_FLOWBASE_PMLOW_B035 | 0.598 | 0.604 | 0.571 |
| UNet_CurrentSplit_E100 | 0.532 | 0.536 | 0.568 |
| Pix2Pix_CurrentSplit_E100 | 0.568 | 0.571 | 0.562 |
| CycleGAN_E100 | 0.559 | 0.559 | 0.612 |
| FA_GT | 0.537 | 0.538 | 0.641 |

## ADNI Current State

ADNI currently has reference results only:

| Method | Meaning | Trained model? |
|---|---|---|
| ADNI_T1_ONLY | direct T1-as-FA reference | No |
| ADNI_FA_GT | real FA upper bound | No |

ADNI reference image metrics:

| Method | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADNI_T1_ONLY | 14.052 | 0.748 | 0.042848 | 0.105173 | 3.616 | 0.317 | 0.051 |
| ADNI_FA_GT | inf | 1.000 | 0.000000 | 0.000000 | 1.000 | 0.000 | 1.000 |

ADNI reference main downstream task `CN vs MCI_spectrum+AD`:

| Method | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| T1_ONLY | 0.563 | 0.565 | 0.550 |
| FA_GT | 0.548 | 0.548 | 0.622 |

The next step is to train ADNI U-Net, Pix2Pix, CycleGAN, and the current main method before making any ADNI method claims.
