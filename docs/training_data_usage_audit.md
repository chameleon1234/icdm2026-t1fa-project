# Training Data Usage Audit and Full-Data Supplement

Date: 2026-06-29 / 2026-06-30

## Core Conclusion

The ADNI A080 + disease-sensitive corrector has now been supplemented with a strict full-data training run. It is no longer only a 4096-slice training branch.

Definitions:

- Full training: `train_limit=0` and `val_limit=0`, using the complete train/val split.
- Limited training: explicit limits such as `train_limit=4096` and `val_limit=1024`.
- Full evaluation: prediction folders cover the complete test split, 5616 ADNI test slices or 1900 private test slices.

## Dataset Size

| Dataset | Split | Slices |
| --- | --- | ---: |
| ADNI | train | 19552 |
| ADNI | val | 2808 |
| ADNI | test | 5616 |
| Private | train | 8650 |
| Private | val | 1850 |
| Private | test | 1900 |

## Supplemented ADNI A080 + DS Pipeline

The full-data chain generated these folders:

| Stage | Output folder | Count |
| --- | --- | ---: |
| Fidelity Flow train source | `outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL` | 19552 |
| Fidelity Flow val source | `outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_VAL_FULL` | 2808 |
| LightGuard train source | `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_TRAIN_FULL_K8` | 19552 |
| LightGuard val source | `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_VAL_FULL_K8` | 2808 |
| LowGuard train base | `outputs/icdm2026/predictions/ADNI_LOWGUARD_TEMPLATE_STRONG_TRAIN_FULL` | 19552 |
| LowGuard val base | `outputs/icdm2026/predictions/ADNI_LOWGUARD_TEMPLATE_STRONG_VAL_FULL` | 2808 |
| A080 train base | `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL` | 19552 |
| A080 val base | `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_VAL_FULL` | 2808 |

A080 fusion parameters were unchanged: `alpha=0.8`, `kernel=9`, `clip_delta=0.06`.

## Full DS Corrector Training

Run name: `adni_ds_corrector_from_a080_disease_roi_hfpreserve_full_e12`

Training log confirmed:

```text
DS Stage2 single-slice training | variant=multihead | train=19552 val=2808 | stage1=none | coarse_pred_dir=outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL | roi_weights=yes | device=cuda
```

Main parameters:

- `variant=multihead`
- `epochs=12`
- `batch_size=2`
- `num_workers=0`
- `lr=6e-5`
- `width=48`
- `num_blocks=8`
- `mixed_precision=bf16`
- `correction_scale=0.08`
- `disease_roi_csv=outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv`
- `hf_preserve_weight=4.0`
- `sharp_retention_weight=3.0`
- `train_limit=0`
- `val_limit=0`

## Full ADNI Test Results

| Method | Training | Test slices | PSNR | SSIM | MAE | WM-MAE | SharpRatio | ROI-CCC | ROI-Spearman | Slice Consistency Error | Template Residual MAE | WM Skeleton Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A080 base | derived | 5616 | 28.371 | 0.9086 | 0.01744 | 0.06135 | 0.996 | 0.768 | 0.842 | 0.03288 | 0.09238 | 0.08270 |
| A080 + DS full e12 gate-best | full train 19552 / val 2808 | 5616 | 28.765 | 0.9137 | 0.01667 | see summary | see summary | 0.835 | see summary | see summary | see summary | see summary |
| A080 + DS full e12 score-best | full train 19552 / val 2808 | 5616 | 28.816 | 0.9142 | 0.01656 | 0.05455 | 1.023 | 0.842 | 0.853 | 0.03198 | 0.08657 | 0.07367 |
| Old Fidelity Flow full | full train 19552 / val 2808 | 5616 | 28.092 | 0.9053 | 0.01769 | 0.05697 | 0.889 | 0.867 | not recorded | not recorded | not recorded | not recorded |

## Interpretation

1. A080 + DS full e12 score-best is now a true full-training candidate.
2. Compared with A080 base, the full DS corrector improves PSNR, SSIM, MAE, WM-MAE, ROI-CCC, and sharpness retention.
3. Compared with Old Fidelity Flow full, A080 + DS full e12 has stronger PSNR, SSIM, MAE, WM-MAE, and SharpRatio, while Old Fidelity Flow still has higher ROI-CCC.
4. This makes score-best a strong candidate for the ?clear + WM/PSNR? story, while Old Fidelity Flow remains the strongest ROI-consistency reference.

## Output Paths

Predictions:

- `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST`
- `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`

Metrics:

- `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST_summary.json`
- `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json`

Visualizations:

- `outputs/icdm2026/figures/method_slices/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST`
- `outputs/icdm2026/figures/method_slices/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`

## Next Steps

1. Visually inspect score-best on the fixed hard cases.
2. Build a panel comparing GT, Old Fidelity Flow, A080 base, and A080+DS score-best.
3. If the visual quality is stable, report A080+DS full e12 as the clear/WM/PSNR candidate and Old Fidelity Flow as the ROI-CCC reference.
4. Run downstream ADNI and private classification/MIL protocols for this full-data candidate.
