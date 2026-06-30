## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-30
- Verification Status: ANALYZED
- Version Label: final_downstream_audit_v1

## Final Downstream Evaluation Audit

- **Audited method**: `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **Prediction directory**: `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **ADNI manifest**: `data/adni_processed/adni_slice_manifest.csv`
- **Main conclusion**: the latest ADNI fair train/test MIL audit is complete; `ADNI_A080_DS_FULL` is the final-method alias; UNet/Pix2Pix/CycleGAN lack train-full prediction folders and are therefore only included in the test-CV supplement.

## Key Results

| Protocol Family | Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---|---:|---:|---:|---:|
| fair_train_test_MIL | `ADNI_A080_BASE` | 9 | 0.736 | 0.702 | 0.600 |
| fair_train_test_MIL | `ADNI_A080_DS_FULL` | 9 | 0.710 | 0.723 | 0.593 |
| fair_train_test_MIL | `ADNI_OLD_FIDELITY_FLOW` | 9 | 0.678 | 0.713 | 0.573 |
| fair_train_test_MIL | `FA_GT` | 9 | 0.724 | 0.676 | 0.573 |
| fair_train_test_MIL | `T1_ONLY` | 9 | 0.690 | 0.709 | 0.582 |

`ADNI_A080_DS_FULL` has the highest mean Macro-AUC, but it is not the top method by mean ACC or mean Macro-F1. The manuscript should therefore avoid claiming that it is best on all downstream metrics. Recommended wording:

> A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.

## Output Files

- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_audit_cn.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_audit.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.csv`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_protocol_coverage.csv`

## Private Dataset Status

Private downstream evidence exists from historical audits, but it was not freshly rerun in this pass. It can be used as existing/private evidence. A fully symmetric private rerun would require checking train/test prediction folders for `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` and rerunning the same subject-level protocol.

## Reproducibility

- **Method**: environment-sensitive artifact audit
- **Verdict**: PARTIALLY_REPRODUCIBLE
- **Reason**: file structure, final-method alias, manifest path, and result tables were checked; byte-level deterministic reproduction was not performed.

