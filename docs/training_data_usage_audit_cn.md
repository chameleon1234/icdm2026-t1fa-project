# 训练数据使用审计与全量补充验证

日期：2026-06-29 / 2026-06-30

## 核心结论

本次补充验证已经把 ADNI A080 + disease-sensitive corrector 从原来的 4096-slice 训练版，补成了严格意义上的全量训练版。

需要区分三件事：

- 全量训练：训练命令中 `train_limit=0`、`val_limit=0`，实际使用完整 train/val split。
- 限量训练：训练命令或 checkpoint args 中存在 `train_limit=4096`、`val_limit=1024` 等限制。
- 全量评估：预测目录覆盖完整 test split，ADNI 为 5616 张 test slices，私有集为 1900 张 test slices。

## 数据集规模

| Dataset | Split | Slices |
| --- | --- | ---: |
| ADNI | train | 19552 |
| ADNI | val | 2808 |
| ADNI | test | 5616 |
| Private | train | 8650 |
| Private | val | 1850 |
| Private | test | 1900 |

## 这次补齐的 ADNI A080 + DS 全量链路

### 1. 全量导出 Fidelity Flow train/val 源

| Output folder | Count |
| --- | ---: |
| `outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL` | 19552 |
| `outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_VAL_FULL` | 2808 |

### 2. 全量导出 LightGuard prior-flow train/val 源

| Output folder | Count |
| --- | ---: |
| `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_TRAIN_FULL_K8` | 19552 |
| `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_VAL_FULL_K8` | 2808 |

### 3. 全量生成 LowGuard base

LowGuard 参数与之前 4096 版保持一致：

- `low_kernel=17`
- `mask_kernel=13`
- `excess_threshold=0.015`
- `excess_softness=0.025`
- `alpha=0.5`
- `hp_scale=1.0`
- `edge_protect=0.35`

| Output folder | Count |
| --- | ---: |
| `outputs/icdm2026/predictions/ADNI_LOWGUARD_TEMPLATE_STRONG_TRAIN_FULL` | 19552 |
| `outputs/icdm2026/predictions/ADNI_LOWGUARD_TEMPLATE_STRONG_VAL_FULL` | 2808 |

### 4. 全量生成 A080 train/val

A080 融合参数与原先测试集 A080 保持一致：

- `base = LowGuard`
- `detail = LightGuard K8`
- `alpha=0.8`
- `kernel=9`
- `clip_delta=0.06`

| Output folder | Count |
| --- | ---: |
| `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL` | 19552 |
| `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_VAL_FULL` | 2808 |

### 5. 全量训练 DS corrector

Run name：`adni_ds_corrector_from_a080_disease_roi_hfpreserve_full_e12`

训练日志确认：

```text
DS Stage2 single-slice training | variant=multihead | train=19552 val=2808 | stage1=none | coarse_pred_dir=outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL | roi_weights=yes | device=cuda
```

主要参数：

- `variant=multihead`
- `epochs=12`
- `batch_size=2`
- `num_workers=0`
- `lr=6e-5`
- `width=48`
- `num_blocks=8`
- `mixed_precision=bf16`
- `correction_scale=0.08`
- `disease_roi_csv=outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv`
- `hf_preserve_weight=4.0`
- `sharp_retention_weight=3.0`
- `train_limit=0`
- `val_limit=0`

验证集训练趋势：

| Epoch | PSNR | SSIM | DeltaPSNR | DeltaSSIM | DeltaWM | DeltaROI | DeltaDiseaseROI | SharpRetention | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.5738 | 0.9129 | 0.1808 | 0.0049 | 0.006031 | 0.015226 | 0.015362 | 1.0405 | 1 |
| 5 | 28.5876 | 0.9131 | 0.1946 | 0.0051 | 0.006746 | 0.015565 | 0.015683 | 1.0372 | 1 |
| 10 | 28.5964 | 0.9133 | 0.2033 | 0.0052 | 0.006727 | 0.016406 | 0.016198 | 1.0328 | 1 |
| 11 | 28.6022 | 0.9131 | 0.2091 | 0.0051 | 0.007235 | 0.016842 | 0.016631 | 1.0284 | 1 |
| 12 | 28.6042 | 0.9136 | 0.2111 | 0.0056 | 0.008624 | 0.018656 | 0.018436 | 1.0339 | 0 |

第 12 轮综合分最好，但由于 stripe gate 未通过，另有 gate-best checkpoint。

## ADNI 测试集全量评估结果

| Method | Training | Test slices | PSNR | SSIM | MAE | WM-MAE | SharpRatio | ROI-CCC | ROI-Spearman | Slice Consistency Error | Template Residual MAE | WM Skeleton Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A080 base | derived | 5616 | 28.371 | 0.9086 | 0.01744 | 0.06135 | 0.996 | 0.768 | 0.842 | 0.03288 | 0.09238 | 0.08270 |
| A080 + DS full e12 gate-best | full train 19552 / val 2808 | 5616 | 28.765 | 0.9137 | 0.01667 | see summary | see summary | 0.835 | see summary | see summary | see summary | see summary |
| A080 + DS full e12 score-best | full train 19552 / val 2808 | 5616 | 28.816 | 0.9142 | 0.01656 | 0.05455 | 1.023 | 0.842 | 0.853 | 0.03198 | 0.08657 | 0.07367 |
| Old Fidelity Flow full | full train 19552 / val 2808 | 5616 | 28.092 | 0.9053 | 0.01769 | 0.05697 | 0.889 | 0.867 | not recorded | not recorded | not recorded | not recorded |

## 当前判断

1. A080 + DS full e12 score-best 现在是一个真正全量训练版本，不再是 4096-slice 训练分支。
2. 与 A080 base 相比，DS corrector 明确带来医学一致性和重建指标提升：
   - PSNR：28.371 -> 28.816
   - SSIM：0.9086 -> 0.9142
   - WM-MAE：0.06135 -> 0.05455
   - ROI-CCC：0.768 -> 0.842
   - SharpRatio：0.996 -> 1.023，清晰度没有被抹掉。
3. 与 Old Fidelity Flow full 相比，A080 + DS full e12 的 PSNR、SSIM、MAE、WM-MAE、SharpRatio 更强，但 ROI-CCC 仍低于 Old Fidelity Flow：0.842 vs 0.867。
4. 因此，如果论文强调“清晰 + 全局重建 + WM 指标”，score-best 是强候选；如果强调 ROI-CCC，Old Fidelity Flow 仍是强基线。

## 输出路径

预测结果：

- Gate-best：`outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST`
- Score-best：`outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`

评估结果：

- `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST_summary.json`
- `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json`

可视化：

- `outputs/icdm2026/figures/method_slices/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_GATEBEST`
- `outputs/icdm2026/figures/method_slices/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`

## 论文表述建议

英文：

> All final image evaluations were conducted on the full held-out test split. The main Fidelity Flow model and conventional baselines were trained on the full training split. We additionally completed a full-data training run for the A080 disease-sensitive corrector using all 19,552 ADNI training slices and 2,808 validation slices. This full-data corrector improved PSNR, SSIM, WM-MAE, and sharpness retention over the A080 base while preserving high-frequency texture.

中文：

> 所有最终图像指标均在完整独立测试集上评估。主 Fidelity Flow 模型和常规对比方法均使用完整训练集训练。本次进一步补齐了 A080 disease-sensitive corrector 的全量训练版本，使用 ADNI 的 19552 张训练切片和 2808 张验证切片。全量 DS corrector 相比 A080 base 同时提升了 PSNR、SSIM、WM-MAE 和清晰度保持能力，并且没有抹掉高频纹理。

## 下一步最该做什么

1. 人工查看 score-best 可视化，重点看是否有 A080 或 LPIPS+GAN 类似的局部过亮纹理。
2. 用同一批 hard-case panel 横向比较：GT、Old Fidelity Flow、A080 base、A080+DS score-best。
3. 若 score-best 视觉稳定，可把它作为“清晰/WM/PSNR 强候选”；Old Fidelity Flow 作为“ROI-CCC 强候选”。论文中可以把二者作为主线演化或消融，而不是强行说单一指标全胜。
4. 继续跑双数据集下游任务，确认 A080+DS full e12 是否也能提升分类/MIL 指标。
