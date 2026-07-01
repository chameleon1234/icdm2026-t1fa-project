# ??? full-heavy ??????

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: private_fair_full_heavy_v1

## 1. Coverage Audit

?????????? train/test prediction folder????? eligible: **True**?

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

## 2. ??????

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

## 3. ??????

| method                                      | task               | mean_accuracy | mean_macro_auc | mean_macro_f1 |
| ------------------------------------------- | ------------------ | ------------- | -------------- | ------------- |
| FA_GT                                       | cn_vs_ad           | 0.6984        | 0.7892         | 0.6312        |
| UNet                                        | cn_vs_ad           | 0.6984        | 0.7794         | 0.6354        |
| Pix2Pix                                     | cn_vs_ad           | 0.6508        | 0.7696         | 0.5868        |
| T1_ONLY                                     | cn_vs_ad           | 0.6825        | 0.7598         | 0.6314        |
| Stage1_LPIPS_GAN                            | cn_vs_ad           | 0.6984        | 0.7451         | 0.6222        |
| FidelityFlow                                | cn_vs_ad           | 0.7619        | 0.7304         | 0.6769        |
| CycleGAN                                    | cn_vs_ad           | 0.7778        | 0.7304         | 0.6954        |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | cn_vs_ad           | 0.6984        | 0.6667         | 0.6106        |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | cn_vs_ad           | 0.7302        | 0.6373         | 0.6221        |
| FA_GT                                       | cn_vs_mci_spectrum | 0.6176        | 0.6874         | 0.5782        |
| UNet                                        | cn_vs_mci_spectrum | 0.5294        | 0.6390         | 0.4922        |
| Stage1_LPIPS_GAN                            | cn_vs_mci_spectrum | 0.5196        | 0.5998         | 0.4543        |
| Pix2Pix                                     | cn_vs_mci_spectrum | 0.5196        | 0.5894         | 0.4073        |
| FidelityFlow                                | cn_vs_mci_spectrum | 0.5588        | 0.5732         | 0.5108        |
| CycleGAN                                    | cn_vs_mci_spectrum | 0.5000        | 0.5686         | 0.4394        |
| T1_ONLY                                     | cn_vs_mci_spectrum | 0.4902        | 0.5490         | 0.4320        |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | cn_vs_mci_spectrum | 0.4706        | 0.4406         | 0.4112        |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | cn_vs_mci_spectrum | 0.4510        | 0.4002         | 0.3932        |
| T1_ONLY                                     | mci_spectrum_vs_ad | 0.7302        | 0.7598         | 0.6647        |
| FidelityFlow                                | mci_spectrum_vs_ad | 0.6984        | 0.7549         | 0.6100        |
| UNet                                        | mci_spectrum_vs_ad | 0.7460        | 0.7549         | 0.6476        |
| Pix2Pix                                     | mci_spectrum_vs_ad | 0.6984        | 0.7402         | 0.5948        |
| Stage1_LPIPS_GAN                            | mci_spectrum_vs_ad | 0.6825        | 0.7402         | 0.6097        |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | mci_spectrum_vs_ad | 0.6825        | 0.7255         | 0.5951        |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | mci_spectrum_vs_ad | 0.6984        | 0.6961         | 0.5938        |
| CycleGAN                                    | mci_spectrum_vs_ad | 0.6190        | 0.6961         | 0.5443        |
| FA_GT                                       | mci_spectrum_vs_ad | 0.6508        | 0.6373         | 0.5629        |

## 4. ?????????

- ??????? ADNI ??? fair subject-level train/test downstream?
- ????: `cn_vs_mci_spectrum`, `cn_vs_ad`, `mci_spectrum_vs_ad`??????? MCI-spectrum ????? `SCD + MCI`?
- ??? test-as-train??????????? train prediction folder ? test prediction folder?
- `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` ??? Macro-AUC ??? **9**??????????
- `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` ? integrated score ??? **1**??????????? best integrated balance ? trade-off?????????????

- `classification_repeated_summary.csv` ????? test-CV ???????????????????? subject-level train/test summary ??????? `classification_subject_summary.csv` ? `method_average_summary.csv` ???
