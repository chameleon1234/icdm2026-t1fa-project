# Private Per-Task Sanity Check

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: private_per_task_sanity_v1

## PRIVATE_DS_HYBRID Per-Task Performance

| task               | mean_accuracy | mean_macro_auc | mean_macro_f1 | rows   |
| ------------------ | ------------- | -------------- | ------------- | ------ |
| cn_vs_mci_spectrum | 0.4706        | 0.4406         | 0.4112        | 3.0000 |
| cn_vs_ad           | 0.7302        | 0.6373         | 0.6221        | 3.0000 |
| mci_spectrum_vs_ad | 0.6984        | 0.6961         | 0.5938        | 3.0000 |

The weakest task is **cn_vs_mci_spectrum**: Accuracy=0.4706, Macro-AUC=0.4406, Macro-F1=0.4112. This task is the main reason why the average Macro-AUC of PRIVATE_DS_HYBRID is low.

## Interpretation

- CN vs AD is valid but not leading for PRIVATE_DS_HYBRID.
- CN vs MCI-spectrum is the weakest task, suggesting limited robustness for early-spectrum discrimination.
- MCI-spectrum vs AD is moderate but still below several baselines.

## Method Averages

| method                                      | mean_accuracy | mean_macro_auc | mean_macro_f1 | rows   | mean_n_train_subjects | mean_n_test_subjects |
| ------------------------------------------- | ------------- | -------------- | ------------- | ------ | --------------------- | -------------------- |
| UNet                                        | 0.6580        | 0.7244         | 0.5918        | 9.0000 | 134.5556              | 25.3333              |
| FA_GT                                       | 0.6556        | 0.7046         | 0.5908        | 9.0000 | 134.5556              | 25.3333              |
| Pix2Pix                                     | 0.6229        | 0.6997         | 0.5296        | 9.0000 | 134.5556              | 25.3333              |
| Stage1_LPIPS_GAN                            | 0.6335        | 0.6950         | 0.5621        | 9.0000 | 134.5556              | 25.3333              |
| T1_ONLY                                     | 0.6343        | 0.6895         | 0.5761        | 9.0000 | 134.5556              | 25.3333              |
| FidelityFlow                                | 0.6730        | 0.6862         | 0.5992        | 9.0000 | 134.5556              | 25.3333              |
| CycleGAN                                    | 0.6323        | 0.6650         | 0.5597        | 9.0000 | 134.5556              | 25.3333              |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 0.6106        | 0.5975         | 0.5329        | 9.0000 | 134.5556              | 25.3333              |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | 0.6331        | 0.5913         | 0.5424        | 9.0000 | 134.5556              | 25.3333              |

## Per-Task Results

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

## Subject Count Consistency

| protocol       | task               | train_subject_counts | test_subject_counts | methods | consistent_across_methods |
| -------------- | ------------------ | -------------------- | ------------------- | ------- | ------------------------- |
| attention_mil  | cn_vs_ad           | [81]                 | [21]                | 9.0000  | 1.0000                    |
| attention_mil  | cn_vs_mci_spectrum | [152]                | [34]                | 9.0000  | 1.0000                    |
| attention_mil  | mci_spectrum_vs_ad | [113]                | [21]                | 9.0000  | 1.0000                    |
| multi_task_mil | cn_vs_ad           | [173]                | [21]                | 9.0000  | 1.0000                    |
| multi_task_mil | cn_vs_mci_spectrum | [173]                | [34]                | 9.0000  | 1.0000                    |
| multi_task_mil | mci_spectrum_vs_ad | [173]                | [21]                | 9.0000  | 1.0000                    |
| slice_svm_vote | cn_vs_ad           | [81]                 | [21]                | 9.0000  | 1.0000                    |
| slice_svm_vote | cn_vs_mci_spectrum | [152]                | [34]                | 9.0000  | 1.0000                    |
| slice_svm_vote | mci_spectrum_vs_ad | [113]                | [21]                | 9.0000  | 1.0000                    |

Subject counts are consistent across methods within each protocol/task. Different tasks use different numbers of subjects because they include different diagnostic groups.

## Label and MCI-Spectrum Definition

The project label mapping is `1=CN, 2=SCD, 3=MCI, 4=AD`. In this audit, `MCI-spectrum` is explicitly defined as `SCD + MCI`, aligned with the ADNI downstream task semantics.

| task               | included_groups | label_0_slice_count | label_1_slice_count | n_subjects | binary_valid |
| ------------------ | --------------- | ------------------- | ------------------- | ---------- | ------------ |
| cn_vs_mci_spectrum | CN,MCI,SCD      | 3000.0000           | 4600.0000           | 152.0000   | 1.0000       |
| cn_vs_ad           | AD,CN           | 3000.0000           | 1050.0000           | 81.0000    | 1.0000       |
| mci_spectrum_vs_ad | AD,MCI,SCD      | 4600.0000           | 1050.0000           | 113.0000   | 1.0000       |

## Leakage and Aggregation Check

| method                                      | train_subjects | test_subjects | overlap_subjects | leakage_risk |
| ------------------------------------------- | -------------- | ------------- | ---------------- | ------------ |
| CycleGAN                                    | 173.0000       | 38.0000       | 0.0000           | no           |
| FA_GT                                       | 173.0000       | 38.0000       | 0.0000           | no           |
| FidelityFlow                                | 173.0000       | 38.0000       | 0.0000           | no           |
| Pix2Pix                                     | 173.0000       | 38.0000       | 0.0000           | no           |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL     | 173.0000       | 38.0000       | 0.0000           | no           |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 173.0000       | 38.0000       | 0.0000           | no           |
| Stage1_LPIPS_GAN                            | 173.0000       | 38.0000       | 0.0000           | no           |
| T1_ONLY                                     | 173.0000       | 38.0000       | 0.0000           | no           |
| UNet                                        | 173.0000       | 38.0000       | 0.0000           | no           |

No train/test subject overlap was detected. The evaluation uses subject-level aggregation / MIL-level protocols, and no slice-level leakage or subject aggregation anomaly was found.
