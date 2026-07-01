# 模型结构图绘制 Checklist

这个 checklist 用于论文主图、附录图和 PPT 图。当前只围绕最终主方法 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` 绘制。

## 1. 必须使用的最终方法名称

- 代码/结果名：`ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- 论文简称：`A080+DS Full`
- 方法全称：`Frequency-aware sharp FA prior + Disease-sensitive high-frequency-preserving correction`

不要在最终结构图中把方法称为：

- 纯 PMRF。
- 纯 Fidelity Flow。
- 纯 GAN。
- StackUNet frequency fusion。

## 2. 必须引用的核心文件

模型与导出：

- `scripts/blend_highpass_detail.py`
- `pmrf_t1fa/train_pmrf_t1fa_stage2_ds_corrector.py`
- `scripts/export_ds_corrector_predictions.py`
- `outputs/adni_ds_corrector_from_a080_disease_roi_hfpreserve_full_e12/checkpoints/best_score_ds_corrector.pt`

最终预测和评估：

- `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST/export_summary.json`
- `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json`
- `outputs/icdm2026/final_dual_downstream_closure/final_integrated_comparison_table_fixed.md`

ROI 权重：

- `outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv`

A080 prior：

- `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080`
- `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080/blend_manifest.csv`

## 3. 主方法图必须出现的模块

- [ ] T1 single slice input。
- [ ] A080 frequency-aware sharp FA prior。
- [ ] Stable low-frequency FA source。
- [ ] FA-space high-frequency detail source。
- [ ] Clipped high-frequency delta。
- [ ] Disease-sensitive corrector。
- [ ] 9-channel condition。
- [ ] WM / disease-sensitive ROI gate。
- [ ] Low / high / stripe correction heads。
- [ ] Final synthetic FA。
- [ ] Reconstruction / medical / texture / downstream evaluation。

## 4. A080 图必须写清楚的公式

```text
base_hp   = Base - LP(Base)
detail_hp = Detail - LP(Detail)
delta_hp  = clip(detail_hp - base_hp, -0.06, 0.06)
A080      = Base + 0.8 * delta_hp
```

必须标注：

- `alpha=0.8`
- `kernel=9`
- `clip_delta=0.06`

证据：

- `scripts/blend_highpass_detail.py`
- `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080/blend_manifest.csv`

## 5. Stage2 图必须写清楚的结构

输入条件：

```text
Condition = concat(
  T1,
  A080,
  LP(A080),
  HP(A080),
  Edge(T1),
  DiseaseROIMap,
  |Edge(T1)-HP(A080)|,
  CoordX,
  CoordY
)
```

网络：

```text
Conv2d(9 -> 48)
8 x NAFBlock(48)
Conv2d(48 -> 4)
```

输出：

- 3 个 correction heads：low / high / stripe。
- 1 个 uncertainty/log-sigma head。

最终：

```text
correction = low_corr + high_corr + stripe_corr
final = clamp(A080 + correction, -1, 1)
```

## 6. 形状检查

- [x] T1 input：`[B, 1, 224, 224]`
- [x] A080 prior：`[B, 1, 224, 224]`
- [x] Stage2 condition：`[B, 9, 224, 224]`
- [x] Stage2 raw output：`[B, 4, 224, 224]`
- [x] Final FA：`[B, 1, 224, 224]`
- [ ] Downstream feature count：未在当前文件中确认，不要写死。

## 7. 结果图建议

论文主文建议 4 张图：

1. 总体框架图。
2. A080 sharp FA prior 构建图。
3. Disease-sensitive high-frequency-preserving corrector 结构图。
4. 定量 + 定性结果图。

PPT 建议 7 张图：

1. 任务动机。
2. 失败路线与经验。
3. 最终方法概念。
4. A080 频域融合。
5. DS corrector 结构。
6. 双数据集/对比实验结果。
7. 下游任务和局限性。

## 8. 论文图中建议放的结果数字

ADNI final test：

- `n_slices = 5616`
- `n_subjects = 108`

A080+DS Full：

- PSNR = 28.8156
- SSIM = 0.9142
- MSE = 0.0014
- MAE = 0.0166
- WM-MAE = 0.0545
- ROI-CCC = 0.8423
- ROI-Spearman = 0.8531
- Sharpness Ratio = 1.0226
- Downstream ACC = 0.7100
- Downstream Macro-AUC = 0.7226
- Downstream Macro-F1 = 0.5931

Stage2 相对 A080 Base 的提升：

- PSNR：28.3714 -> 28.8156
- SSIM：0.9086 -> 0.9142
- WM-MAE：0.0613 -> 0.0545
- ROI-CCC：0.7677 -> 0.8423
- Sharpness Ratio：0.9961 -> 1.0226

## 9. 必须避免的说法

- [ ] 不写“所有指标第一”。
- [ ] 不写“双数据集下游都第一”。
- [ ] 不写“synthetic FA 替代真实 FA”。
- [ ] 不把 `disease-sensitive grid ROI` 写成真实 anatomical atlas 分割。
- [ ] 不把 FID 写成最终主评估或最终选优指标，因为最终 `fid_eval_every=0`。
- [ ] 不把 A080 画成单一在线神经网络 Stage1。

推荐安全说法：

> A080+DS Full 在重建质量、白质/ROI 医学一致性和清晰度保持之间取得最佳综合平衡，并提供有竞争力的下游可用性证据。

## 10. 视觉设计建议

颜色：

- T1：灰色。
- A080 prior：蓝色。
- 高频纹理支路：紫色或青色。
- WM / ROI：绿色。
- Stage2 correction：橙色。
- 输出 FA：黑白灰。

版式：

- 论文图尽量用横向 left-to-right pipeline。
- A080 构建图建议用上下两条支路：Base low-frequency 和 Detail high-frequency。
- DS corrector 图建议把 9-channel condition 放左侧，把 multihead correction 放右侧。
- Loss 不要全堆在主图里，可用四个小标签表示：reconstruction、medical、texture、artifact。

## 11. 附录图建议

- 失败路线总结图：posterior mean blur、LPIPS+GAN over-bright、T1 high-pass mismatch。
- Ablation 表：A080 Base、A080+DS Full、Old Fidelity Flow、Pix2Pix、U-Net。
- Downstream protocol 图：slice-level features -> subject-level aggregation / MIL。
- Private dataset supplementary table。

## 12. 画图前最终核对

- [ ] 图中最终方法名是否统一为 A080+DS Full。
- [ ] 是否明确 A080 是 frozen prior。
- [ ] 是否明确 Stage2 是 bounded correction，不是重新生成。
- [ ] 是否画出 disease-sensitive ROI gate。
- [ ] 是否画出 high-frequency preserving loss/constraint。
- [ ] 是否避免夸大下游结论。
- [ ] 是否区分 ADNI final method 和 private final candidate。
- [ ] 是否标注最终结果来自 full ADNI test set：5616 slices / 108 subjects。

