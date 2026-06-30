# Training Data Usage Audit and Final Status

Date: 2026-06-30

## Core conclusion

A080+DS Full has been supplemented with ADNI full-data training and is no longer a 4096-slice branch. The final main method is fixed as:

`ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`

## Dataset size

| Dataset | Split | Slices |
| --- | --- | ---: |
| ADNI | train | 19552 |
| ADNI | val | 2808 |
| ADNI | test | 5616 |
| Private | train | 8650 |
| Private | val | 1850 |
| Private | test | 1900 |

## Full-data confirmation

- Old Fidelity Flow is a full-data training run.
- A080+DS Full is now also a full-data training run.
- A080+DS Full used the full ADNI train/val split: train = 19552 slices, val = 2808 slices.
- All final image metrics are evaluated on the full independent ADNI test split: 5616 slices.
- PriorFlow safe 4096/e8 is an ablation/texture exploration result and is not the final main method.

## A080+DS Full training chain

- A080 base: `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL`
- DS corrector run: `adni_ds_corrector_from_a080_disease_roi_hfpreserve_full_e12`
- Final prediction: `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- Final metrics: `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json`

## Final test results

| Method | PSNR | SSIM | MAE | WM-MAE | SharpRatio | ROI-CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A080 base | 28.371 | 0.9086 | 0.01744 | 0.06135 | 0.996 | 0.768 |
| A080+DS Full score-best | 28.816 | 0.9142 | 0.01656 | 0.05455 | 1.023 | 0.842 |
| Old Fidelity Flow full | 28.092 | 0.9053 | 0.01769 | 0.05697 | 0.889 | 0.867 |

## Final wording

A080+DS Full provides the best overall balance among reconstruction fidelity, white-matter fidelity, sharpness preservation, and downstream utility. Old Fidelity Flow remains a strong ROI-consistency baseline, but A080+DS Full is more suitable as the final integrated method.
