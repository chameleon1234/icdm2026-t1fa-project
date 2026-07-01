# Private Full-Heavy Downstream Audit Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: private_fair_full_heavy_v1

All required private methods were audited for separate train/test prediction folders. All eligible: **True**. The downstream tasks are aligned with ADNI: CN vs MCI-spectrum, CN vs AD, and MCI-spectrum vs AD. For the private dataset, MCI-spectrum is explicitly SCD + MCI.

## Coverage

| method                                      | train_png_count | test_png_count | checkpoint_exists | can_export_train_predictions | eligible_for_fair_train_test_MIL | reason_if_not_eligible |
| ------------------------------------------- | --------------- | -------------- | ----------------- | ---------------------------- | -------------------------------- | ---------------------- |
| T1_ONLY                                     | 8650.0000       | 1900.0000      |                   | 0.0000                       | 1.0000                           |                        |
| FA_GT                                       | 8650.0000       | 1900.0000      |                   | 0.0000                       | 1.0000                           |                        |
| FidelityFlow                                | 8650.0000       | 1900.0000      | 0.0000            | 0.0000                       | 1.0000                           |                        |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | 8650.0000       | 1900.0000      | 0.0000            | 0.0000                       | 1.0000                           |                        |
| UNet                                        | 8650.0000       | 1900.0000      | 0.0000            | 0.0000                       | 1.0000                           |                        |
| Pix2Pix                                     | 8650.0000       | 1900.0000      | 0.0000            | 0.0000                       | 1.0000                           |                        |
| CycleGAN                                    | 8650.0000       | 1900.0000      | 0.0000            | 0.0000                       | 1.0000                           |                        |
| Stage1_LPIPS_GAN                            | 8650.0000       | 1900.0000      | 1.0000            | 1.0000                       | 1.0000                           |                        |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 8650.0000       | 1900.0000      | 1.0000            | 1.0000                       | 1.0000                           |                        |

## Downstream Averages

| Method                                      | Accuracy | Macro_AUC | Macro_F1 | Rows   |
| ------------------------------------------- | -------- | --------- | -------- | ------ |
| UNet                                        | 0.6580   | 0.7244    | 0.5918   | 9.0000 |
| FA_GT                                       | 0.6556   | 0.7046    | 0.5908   | 9.0000 |
| Pix2Pix                                     | 0.6229   | 0.6997    | 0.5296   | 9.0000 |
| Stage1_LPIPS_GAN                            | 0.6335   | 0.6950    | 0.5621   | 9.0000 |
| T1_ONLY                                     | 0.6343   | 0.6895    | 0.5761   | 9.0000 |
| FidelityFlow                                | 0.6730   | 0.6862    | 0.5992   | 9.0000 |
| CycleGAN                                    | 0.6323   | 0.6650    | 0.5597   | 9.0000 |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 0.6106   | 0.5975    | 0.5329   | 9.0000 |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | 0.6331   | 0.5913    | 0.5424   | 9.0000 |

## Interpretation

PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL is not the best single downstream method by Macro-AUC. Its paper-safe framing should be integrated balance/trade-off rather than universal downstream superiority.

- `classification_repeated_summary.csv` is not a test-CV result in this run. It is a compatibility copy of the subject-level train/test summary; the primary outputs are `classification_subject_summary.csv` and `method_average_summary.csv`.
