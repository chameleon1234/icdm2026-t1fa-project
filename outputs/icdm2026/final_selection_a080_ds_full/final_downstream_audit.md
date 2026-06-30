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
- **Audit output directory**: `outputs/icdm2026/final_selection_a080_ds_full`
- **Overall confidence**: SOLID for ADNI fair train/test MIL; CAUTION for private downstream because it is an existing historical audit rather than a fresh rerun in this pass.

## Main Findings

1. The latest ADNI downstream evaluation for the final method has been audited under the fair train/test MIL protocol. The final method appears in the tables as `ADNI_A080_DS_FULL`.
2. `ADNI_A080_DS_FULL` is confirmed to map to `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`, with both test and train-full prediction folders available.
3. The fair train/test MIL table includes only methods with valid train and test prediction sources: `T1_ONLY`, `FA_GT`, `ADNI_OLD_FIDELITY_FLOW`, `ADNI_A080_BASE`, and `ADNI_A080_DS_FULL`.
4. `ADNI_UNET`, `ADNI_PIX2PIX`, and `ADNI_CYCLEGAN` have test prediction folders but no train-full prediction folders, so they are not eligible for the fair train/test MIL main table. They are included only in the subject-level test-CV supplement.
5. The available metrics are Accuracy, Macro-AUC, and Macro-F1. Per-class F1 is not emitted by the current downstream scripts and should not be reported as if it exists.
6. The results do not support the claim that the final method is best on every downstream metric. A safer statement is that A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.

## Artifact Completeness

| File | Status | Notes |
|---|---:|---|
| `outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_subject_summary.csv` | present | Latest ADNI fair train/test MIL subject-level summary |
| `outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_repeated_summary.csv` | present | ADNI subject-level test-CV supplement |
| `outputs/icdm2026/final_selection_a080_ds_full/final_selection_metrics.csv` | present | Final image-quality selection table |
| `outputs/icdm2026/final_selection_a080_ds_full/adni_final_main_table.csv` | present | ADNI final image-quality table |
| `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.csv` | generated | Corrected final downstream table |
| `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_protocol_coverage.csv` | generated | Method coverage and protocol eligibility table |

## ADNI Fair Train/Test MIL Results

Protocols: `slice_svm_vote`, `attention_mil`, `multi_task_mil`  
Tasks: `cn_vs_mci_spectrum`, `cn_vs_ad`, `mci_spectrum_vs_ad`  
Aggregation level: subject-level / MIL-level  
Training data: train split  
Test data: test split  
Leakage risk: no evidence of slice-level train/test leakage was found.

| Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---:|---:|---:|---:|
| `ADNI_A080_BASE` | 9 | 0.736 | 0.702 | 0.600 |
| `ADNI_A080_DS_FULL` | 9 | 0.710 | 0.723 | 0.593 |
| `ADNI_OLD_FIDELITY_FLOW` | 9 | 0.678 | 0.713 | 0.573 |
| `FA_GT` | 9 | 0.724 | 0.676 | 0.573 |
| `T1_ONLY` | 9 | 0.690 | 0.709 | 0.582 |

Interpretation: under the fair train/test MIL protocol, `ADNI_A080_DS_FULL` has the highest mean Macro-AUC, but it is not the top method by mean ACC or mean Macro-F1. It should therefore be described as competitive downstream evidence, not as uniformly superior.

## ADNI Supplemental Subject-Level Test-CV Results

This protocol is used to include every method with an available test prediction folder, including `ADNI_UNET`, `ADNI_PIX2PIX`, and `ADNI_CYCLEGAN`. Because these baselines lack train-full prediction folders, this supplement cannot replace the fair train/test MIL main table.

| Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---:|---:|---:|---:|
| `FA_GT` | 3 | 0.715 | 0.704 | 0.607 |
| `ADNI_PIX2PIX` | 3 | 0.623 | 0.466 | 0.517 |
| `ADNI_A080_DS_FULL` | 3 | 0.687 | 0.533 | 0.515 |
| `T1_ONLY` | 3 | 0.689 | 0.577 | 0.504 |
| `ADNI_OLD_FIDELITY_FLOW` | 3 | 0.658 | 0.617 | 0.506 |
| `ADNI_UNET` | 3 | 0.631 | 0.485 | 0.493 |
| `ADNI_A080_BASE` | 3 | 0.629 | 0.439 | 0.478 |
| `ADNI_CYCLEGAN` | 3 | 0.583 | 0.423 | 0.453 |

Interpretation: this supplement provides broader method coverage. It must be labeled as a different protocol and should not be merged into the fair MIL ranking.

## Private Dataset Audit

Private downstream results exist from previous runs, but no fresh private full-data rerun was performed in this audit pass. Existing private outputs include:

- `T1_ONLY`
- `FA_GT`
- `FidelityFlow`
- `Stage1`
- `UNet`
- `Pix2Pix`
- `CycleGAN`
- `T1_PLUS_*` fusion variants

Relevant directories:

- `outputs/icdm2026/downstream_private_slice_mil_finalpdf_train_test_seed2026`
- `outputs/icdm2026/downstream_finalpdf_compatible_private`
- `outputs/icdm2026/downstream_private_final_audit/downstream_private_audit_cn.md`

Conclusion: the private results can be cited as existing/private audit evidence. If the manuscript requires a fully symmetric private result for the latest final candidate, `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` still needs a dedicated train/test prediction-folder check and rerun under the same subject-level protocol.

## Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Protocol mixing | CAUTION | Fair train/test MIL and test-CV supplement use different protocols. | Keep the main and supplemental tables separate. |
| Data leakage | NOTE | Latest ADNI MIL uses train/test split and subject-level aggregation; no slice-level leakage evidence found. | Preserve manifest and command records. |
| Overclaiming | CAUTION | A080+DS Full is not best on all downstream metrics. | Use competitive downstream utility wording. |
| Missing per-class F1 | NOTE | Current scripts do not export per-class F1. | Do not report per-class F1 unless newly computed. |
| Baseline eligibility | CAUTION | UNet/Pix2Pix/CycleGAN lack train-full prediction folders. | Treat them as test-CV supplement only. |
| Final-method alias | NOTE | `ADNI_A080_DS_FULL` is an alias for the final method. | State the full method name in table notes. |
| Private/ADNI asymmetry | CAUTION | Private results are historical; ADNI was newly audited. | Label result provenance in the dual-dataset table. |
| Single metric ranking | CAUTION | AUC, ACC, and F1 rankings differ. | Avoid single-metric superiority claims. |
| Sample dependence | NOTE | MIL aggregates by subject. | Report subject-level protocol explicitly. |
| Multiple tasks | NOTE | The three binary tasks have different difficulty. | Report per-task metrics plus averages. |
| External validity | NOTE | ADNI and private datasets have different distributions. | Report both datasets separately. |

## Reproducibility

- **Method**: environment-sensitive rerun / artifact audit
- **Verdict**: PARTIALLY_REPRODUCIBLE

| Item | Status | Evidence |
|---|---|---|
| Final method prediction directory | MATCH | test and train-full folders exist |
| ADNI manifest | MATCH | `data/adni_processed/adni_slice_manifest.csv` |
| Fair MIL result files | MATCH | subject summary and repeated summary exist |
| Corrected downstream table | MATCH | `final_downstream_table_corrected.csv` |
| Byte-level deterministic rerun | Not checked | This audit verifies structure and metrics, not byte-for-byte determinism |

## Safe Manuscript Wording

A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.

