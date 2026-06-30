# 最终下游审计总报告

## ADNI 最新主方法是否已完整评估

已完成。最终主方法 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` 已在以下目录完成补跑：

`outputs/icdm2026/downstream_adni_final_a080_ds_full_complete`

输出包括：

- 公平 train/test MIL：`classification_subject_summary.csv`
- 全方法 subject-level test-CV 补充：`classification_repeated_summary.csv`

## ADNI train/test MIL 覆盖范围

公平 train/test MIL 包含：

- `T1_ONLY`
- `FA_GT`
- `ADNI_OLD_FIDELITY_FLOW`
- `ADNI_A080_BASE`
- `ADNI_A080_DS_FULL`

该协议使用 `data/adni_processed/adni_slice_manifest.csv`，按 subject-level split 和 subject-level aggregation 评估，不使用 slice-level 泄漏。

`ADNI_UNet_E50`、`ADNI_Pix2Pix_E50`、`ADNI_CycleGAN_E50` 当前缺少 ADNI train-full prediction folder，因此不能放入公平 train/test MIL。它们已放入 subject-level test-CV 补充结果中。

## ADNI train/test MIL 平均结果

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| ADNI_A080_BASE | 0.736 | 0.702 | 0.600 | 9 |
| ADNI_A080_DS_FULL | 0.711 | 0.722 | 0.586 | 9 |
| FA_GT | 0.724 | 0.676 | 0.572 | 9 |
| T1_ONLY | 0.675 | 0.705 | 0.570 | 9 |
| ADNI_OLD_FIDELITY_FLOW | 0.578 | 0.686 | 0.506 | 9 |

## ADNI 全方法 test-CV 平均结果

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| FA_GT | 0.593 | 0.688 | 0.607 | 3 |
| ADNI_PIX2PIX | 0.604 | 0.600 | 0.517 | 3 |
| ADNI_A080_DS_FULL | 0.519 | 0.497 | 0.515 | 3 |
| T1_ONLY | 0.504 | 0.522 | 0.505 | 3 |
| ADNI_OLD_FIDELITY_FLOW | 0.473 | 0.545 | 0.506 | 3 |
| ADNI_UNET | 0.471 | 0.439 | 0.493 | 3 |
| ADNI_A080_BASE | 0.461 | 0.403 | 0.478 | 3 |
| ADNI_CYCLEGAN | 0.445 | 0.402 | 0.453 | 3 |

## Private 审计

私有集已有 train/test MIL 与 repeated CV 结果，详见：

`outputs/icdm2026/downstream_private_final_audit/downstream_private_audit_cn.md`

## 论文写法建议

A080+DS Full 提供了下游可用性证据，并保持有竞争力的分类表现；其主要优势仍然体现在重建质量、白质误差和清晰度保持上。

不要写成“所有下游任务第一”。更稳妥的表述是：最终方法在图像重建质量、白质一致性、清晰度保持和下游可用性之间取得较好的综合平衡。
