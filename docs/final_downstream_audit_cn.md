## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-30
- Verification Status: ANALYZED
- Version Label: final_downstream_audit_v1

## 最终下游任务审计报告

- **审计对象**: `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **预测目录**: `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **ADNI manifest**: `data/adni_processed/adni_slice_manifest.csv`
- **主要结论**: ADNI 最新公平 train/test MIL 已完成；`ADNI_A080_DS_FULL` 是最终方法别名；UNet/Pix2Pix/CycleGAN 因缺少 train-full 预测目录，只能作为 test-CV 补充对比。

## 关键结果

| Protocol Family | Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---|---:|---:|---:|---:|
| fair_train_test_MIL | `ADNI_A080_BASE` | 9 | 0.736 | 0.702 | 0.600 |
| fair_train_test_MIL | `ADNI_A080_DS_FULL` | 9 | 0.710 | 0.723 | 0.593 |
| fair_train_test_MIL | `ADNI_OLD_FIDELITY_FLOW` | 9 | 0.678 | 0.713 | 0.573 |
| fair_train_test_MIL | `FA_GT` | 9 | 0.724 | 0.676 | 0.573 |
| fair_train_test_MIL | `T1_ONLY` | 9 | 0.690 | 0.709 | 0.582 |

`ADNI_A080_DS_FULL` 的平均 Macro-AUC 最高，但平均 ACC 和 Macro-F1 不是第一。因此最终论文应避免写成“所有下游指标第一”，推荐写成:

> A080+DS Full 提供了下游可用性证据，并保持有竞争力的分类表现；其主要优势仍然体现在重建质量、白质误差和清晰度保持上。

## 输出文件

- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_audit_cn.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_audit.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.csv`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.md`
- `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_protocol_coverage.csv`

## 私有集状态

私有集已有历史下游审计结果，但本轮没有重新全量补跑。已有结果可作为 existing/private evidence；如需与 ADNI 最新最终方法完全对称，需要单独确认 `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 的 train/test 预测目录，并按同一 subject-level 协议补跑。

## Reproducibility

- **Method**: environment-sensitive artifact audit
- **Verdict**: PARTIALLY_REPRODUCIBLE
- **Reason**: 文件结构、方法别名、manifest、结果表均已核查；未进行逐字节确定性复现。

