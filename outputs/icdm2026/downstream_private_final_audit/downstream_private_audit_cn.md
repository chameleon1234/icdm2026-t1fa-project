# 私有集最终下游审计

## 结论

私有集最终候选为 `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL`。已有下游结果覆盖 train/test MIL 和 repeated subject-level CV，但这些结果来自历史评估文件，本次没有重新训练新模型。

## 已有 train/test MIL

来源：

`outputs/icdm2026/downstream_private_slice_mil_finalpdf_train_test_seed2026/classification_subject_summary.csv`

包含方法：

- `T1_ONLY`
- `FA_GT`
- `FidelityFlow`
- `Stage1`
- `UNet`
- `Pix2Pix`
- `CycleGAN`
- 以及 `T1_PLUS_*` 融合结果

平均结果如下：

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| T1_PLUS_GT | 0.754 | 0.640 | 0.626 | 9 |
| T1_PLUS_Stage1 | 0.721 | 0.586 | 0.609 | 9 |
| T1_PLUS_Unet | 0.708 | 0.551 | 0.586 | 9 |
| FA_GT | 0.681 | 0.560 | 0.557 | 9 |
| T1_ONLY | 0.646 | 0.545 | 0.539 | 9 |
| FidelityFlow | 0.646 | 0.540 | 0.538 | 9 |

## 已有 repeated subject-level CV

来源：

`outputs/icdm2026/downstream_finalpdf_compatible_private/classification_repeated_summary.csv`

该协议用于补充稳定性观察，不替代 train/test MIL。

## 公平性说明

当前可公平沿用的 private train/test MIL 结果来自已有文件。若要重新补跑 private train/test MIL，需要确保每个方法都有对应 train/test prediction folder，不能使用 test prediction folder 作为 train。

当前检查到的 prediction folders 包括：

- `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL`
- `PM_DIRF_FIDELITY_FLOW_FULL`
- `UNet_CurrentSplit_E100`
- `Pix2Pix_CurrentSplit_E100`
- `CycleGAN_CurrentSplit_E100`

本次任务要求不训练新模型，因此只做审计和汇总。

## 缺失项

当前 private 审计没有生成新的 per-class F1；现有脚本主要输出 Accuracy、Macro-AUC 和 Macro-F1。
