# 下游疾病分型 Utility 评估

## 目的

这个实验用于验证生成 FA 是否保留了与疾病分期相关的信息，服务 ICDM/data mining 叙事，而不是只看 PSNR/SSIM。实验严格按被试级别做：一个被试的所有切片先聚合成一个 subject feature vector，再做分类或回归，避免把 1900 张切片当独立样本造成泄漏。

## 当前方法

- `T1_ONLY`：输入 T1 参考基线。
- `Stage1_Baseline`：保守的 Stage1 PMRF-style 预测。
- `Stage1_LPIPS_GAN`：保留高频纹理的清晰 Stage1。
- `Fidelity_Flow`：频率约束的 Stage2 flow 医学矫正器。
- `Fidelity_Direct`：直接低频矫正器消融。
- `FA_GT`：真实 FA upper bound。
- `T1_PLUS_*`：T1 与生成/真实 FA 的特征级融合。

## 任务

- `four_class`：CN / SCD / MCI / AD 四分类。
- `cn_vs_ad`：CN vs AD。
- `cn_vs_mci_ad`：CN vs MCI+AD。
- `cn_scd_vs_mci_ad`：CN+SCD vs MCI+AD。
- `MMSE` 回归：辅助认知评分回归。

当前论文中最适合作为正结果的是疾病阶段分类任务，而不是 MMSE 回归。

## 执行命令

```powershell
conda activate dinov3test

python scripts/evaluate_downstream_classification.py `
  --include_t1 `
  --include_fa_gt `
  --method Stage1_Baseline=outputs\icdm2026\predictions\PM_STAGE1_WMROI_DETAIL_5SLICE `
  --method Stage1_LPIPS_GAN=outputs\icdm2026\predictions\PM_STAGE1_LPIPS_GAN_5SLICE_FINAL `
  --method Fidelity_Flow=outputs\icdm2026\predictions\PM_DIRF_FIDELITY_FLOW_FULL `
  --method Fidelity_Direct=outputs\icdm2026\predictions\PM_DIRF_FIDELITY_DIRECT_FULL `
  --fusion T1_PLUS_STAGE1_BASELINE=T1_ONLY+Stage1_Baseline `
  --fusion T1_PLUS_STAGE1_LPIPS_GAN=T1_ONLY+Stage1_LPIPS_GAN `
  --fusion T1_PLUS_FIDELITY_FLOW=T1_ONLY+Fidelity_Flow `
  --fusion T1_PLUS_FIDELITY_DIRECT=T1_ONLY+Fidelity_Direct `
  --fusion T1_PLUS_FA_GT=T1_ONLY+FA_GT `
  --tasks four_class,cn_vs_ad,cn_vs_mci_ad,cn_scd_vs_mci_ad `
  --regression_targets MMSE `
  --n_splits 5 `
  --max_features 12 `
  --output_root outputs\icdm2026\downstream_classification
```

`--max_features 12` 表示在每个交叉验证 fold 内做单变量特征选择。这个设置很重要，因为测试集只有 38 个被试，直接拼接高维特征容易过拟合。

## 输出文件

- `outputs/icdm2026/downstream_classification/subject_features.csv`
- `outputs/icdm2026/downstream_classification/classification_summary.csv`
- `outputs/icdm2026/downstream_classification/classification_predictions.csv`
- `outputs/icdm2026/downstream_classification/regression_summary.csv`
- `outputs/icdm2026/downstream_classification/regression_predictions.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.png`

## 当前结果解读

目前不应该把生成 FA 说成强四分类器。更稳妥的结论是疾病阶段 utility：

- `Fidelity_Flow` 在 fold 内特征选择后，是 `CN+SCD vs MCI+AD` 上最好的生成 FA 结果，`Macro-F1=0.559`，高于 T1-only 的 `0.526` 和模糊 Stage1 baseline 的 `0.491`。
- `T1_PLUS_STAGE1_LPIPS_GAN` 在 `CN vs MCI+AD` 上有价值，`Macro-F1=0.611`，接近真实 FA upper bound（`FA_GT=0.683`，`T1_PLUS_FA_GT=0.624`）。
- `MMSE` 回归目前不是生成 FA 的正结果。T1-only 和真实 FA 更强，所以 MMSE 应作为负结果/补充分析，不建议作为主结论。

这支撑当前论文主线：

1. 保守 posterior-mean FA 稳定但视觉模糊。
2. Sharp Stage1 恢复高频结构。
3. Frequency-preserving Stage2 在不破坏锐度的前提下改善医学一致性。
4. 下游 utility 以被试级分类评估，把生成质量和疾病阶段挖掘联系起来。
