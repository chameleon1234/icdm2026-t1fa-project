# ADNI Final Downstream Audit Report

## Conclusion

The downstream evaluation for the final method `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` has been completed under:

`outputs/icdm2026/downstream_adni_final_a080_ds_full_complete`

`classification_subject_summary.csv` corresponds to a fair train/test MIL protocol using `data/adni_processed/adni_slice_manifest.csv`, subject-level split, and subject-level aggregation. No slice-level leakage is used.

## Protocol 1: fair train/test MIL

Included methods: T1_ONLY, FA_GT, ADNI_OLD_FIDELITY_FLOW, ADNI_A080_BASE, and ADNI_A080_DS_FULL.

ADNI_UNet_E50, ADNI_Pix2Pix_E50, and ADNI_CycleGAN_E50 are not included in this MIL table because train-full prediction folders are not available. They are not forced into train/test MIL to avoid test-as-train leakage.

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| ADNI_A080_BASE | 0.736 | 0.702 | 0.600 | 9 |
| ADNI_A080_DS_FULL | 0.710 | 0.723 | 0.593 | 9 |
| T1_ONLY | 0.690 | 0.709 | 0.582 | 9 |
| FA_GT | 0.724 | 0.676 | 0.573 | 9 |
| ADNI_OLD_FIDELITY_FLOW | 0.678 | 0.713 | 0.573 | 9 |

## Protocol 2: all-method subject-level test-CV supplement

To include U-Net, Pix2Pix, and CycleGAN, we additionally ran subject-level repeated CV inside the ADNI test split. This remains subject-level, but should be interpreted as supplemental comparison rather than the fair train/test MIL protocol.

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| FA_GT | 0.715 | 0.704 | 0.607 | 3 |
| ADNI_PIX2PIX | 0.623 | 0.466 | 0.517 | 3 |
| ADNI_A080_DS_FULL | 0.687 | 0.533 | 0.515 | 3 |
| ADNI_OLD_FIDELITY_FLOW | 0.658 | 0.617 | 0.506 | 3 |
| T1_ONLY | 0.689 | 0.577 | 0.504 | 3 |
| ADNI_UNET | 0.631 | 0.485 | 0.493 | 3 |
| ADNI_A080_BASE | 0.629 | 0.439 | 0.478 | 3 |
| ADNI_CYCLEGAN | 0.583 | 0.423 | 0.453 | 3 |

## Metrics

Accuracy, Macro-AUC, and Macro-F1 are available. Per-class F1 is not emitted by the current downstream scripts.

## Recommended wording

A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.
