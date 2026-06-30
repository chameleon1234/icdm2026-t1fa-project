# Final Downstream Audit Report

## ADNI final method status

Completed. The final method `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` has been evaluated under `outputs/icdm2026/downstream_adni_final_a080_ds_full_complete`.

- Fair train/test MIL: `classification_subject_summary.csv`
- All-method subject-level test-CV supplement: `classification_repeated_summary.csv`

## ADNI train/test MIL coverage

The fair MIL table includes T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, and A080+DS Full.

U-Net, Pix2Pix, and CycleGAN do not currently have train-full prediction folders, so they are not included in train/test MIL. They are included in the subject-level test-CV supplement.

## ADNI train/test MIL mean results

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| ADNI_A080_BASE | 0.736 | 0.702 | 0.600 | 9 |
| ADNI_A080_DS_FULL | 0.710 | 0.723 | 0.593 | 9 |
| T1_ONLY | 0.690 | 0.709 | 0.582 | 9 |
| FA_GT | 0.724 | 0.676 | 0.573 | 9 |
| ADNI_OLD_FIDELITY_FLOW | 0.678 | 0.713 | 0.573 | 9 |

## ADNI all-method test-CV mean results

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

## Private audit

The private dataset has existing train/test MIL and repeated-CV results. See `outputs/icdm2026/downstream_private_final_audit/downstream_private_audit_cn.md`.

## Paper wording

A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.
