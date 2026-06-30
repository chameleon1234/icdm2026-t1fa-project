# 最终 Integrated Comparison 修复报告

## 修复内容

已修复 ADNI Pix2Pix / U-Net / CycleGAN 的 downstream merge 缺失问题。修复后的 alias mapping 为：

- `Pix2Pix -> ADNI_PIX2PIX`
- `U-Net -> ADNI_UNET`
- `CycleGAN -> ADNI_CYCLEGAN`

修复后，三种方法在 `final_integrated_comparison_table_fixed.csv` 中均已包含 `downstream_ACC`、`downstream_AUC`、`downstream_F1`。

## Primary Integrated Composite Score

固定权重：

- 图像重建保真度：25%
- 医学一致性：30%
- 纹理真实性：20%
- 下游可用性：25%

ADNI 中 A080+DS Full 排名第 1，primary score 为 `0.9642`。私有集中 PRIVATE_DS_HYBRID 排名第 1，primary score 为 `0.7784`。

## 安全结论

A080+DS Full 被选为最终方法，是因为它在重建保真度、白质/ROI 医学一致性、纹理真实性和下游可用性之间取得最佳综合平衡，而不是因为它在每一个单项指标上都第一。

完整表格见：

- `outputs/icdm2026/final_dual_downstream_closure/final_integrated_comparison_table_fixed.csv`
- `outputs/icdm2026/final_dual_downstream_closure/final_integrated_comparison_table_fixed.md`
