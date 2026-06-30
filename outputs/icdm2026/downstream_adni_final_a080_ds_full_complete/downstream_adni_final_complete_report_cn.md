# ADNI 最终下游完整审计报告

## 结论

ADNI 最终主方法 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` 的下游任务已经补跑到指定目录：

`outputs/icdm2026/downstream_adni_final_a080_ds_full_complete`

其中 `classification_subject_summary.csv` 是公平 train/test MIL 协议，使用 `data/adni_processed/adni_slice_manifest.csv`，按 subject-level split 和 subject-level aggregation 评估，不存在 slice-level 数据泄漏。

## 协议 1：公平 train/test MIL

包含方法：

- `T1_ONLY`
- `FA_GT`
- `ADNI_OLD_FIDELITY_FLOW`
- `ADNI_A080_BASE`
- `ADNI_A080_DS_FULL`

不包含 `ADNI_UNet_E50`、`ADNI_Pix2Pix_E50`、`ADNI_CycleGAN_E50`，原因是当前只有 test prediction folder，没有 ADNI train-full prediction folder。为了避免 test 当 train，本报告不把它们放入 train/test MIL。

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| ADNI_A080_BASE | 0.736 | 0.702 | 0.600 | 9 |
| ADNI_A080_DS_FULL | 0.711 | 0.722 | 0.586 | 9 |
| FA_GT | 0.724 | 0.676 | 0.572 | 9 |
| T1_ONLY | 0.675 | 0.705 | 0.570 | 9 |
| ADNI_OLD_FIDELITY_FLOW | 0.578 | 0.686 | 0.506 | 9 |

## 协议 2：全方法 subject-level test-CV 补充

为了覆盖 U-Net、Pix2Pix、CycleGAN，补跑了 subject-level repeated CV。该协议仍按 subject 聚合，不按 slice 泄漏，但它是在 ADNI test split 内部做 CV，因此只能作为补充对比，不能替代 train/test MIL。

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

## 指标完整性

已包含 Accuracy、Macro-AUC、Macro-F1。当前脚本不输出 per-class F1，因此 per-class F1 标记为缺失。

## 写法建议

A080+DS Full 提供了下游可用性证据，并保持有竞争力的分类表现；其主要优势仍然体现在重建质量、白质误差和清晰度保持上。
