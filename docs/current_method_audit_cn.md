# 当前方法与 ADNI 结果核查

## 结论

上一轮 ADNI 数字不是我们方法的结果，也不是对比方法训练结果，而是两个 reference：

- `ADNI_T1_ONLY`: 直接把 ADNI test T1 切片当成预测 FA。
- `ADNI_FA_GT`: 真实 FA 上界。

ADNI 的 U-Net、Pix2Pix、CycleGAN、DIRF/PMRF/频率融合主方法还没有开始训练。原因是 GPU 仍在跑私有数据集的 CycleGAN 公平重训。

## 已修复的问题

1. ADNI subject ID 不能用私有数据集的 `sub-001` 逻辑压缩。
   - 错误风险：`sub-002_S_0413` 会被压成 `sub-002`，导致多个 ADNI subject 混在一起。
   - 已修复：`normalize_subject_id()` 现在保留 `sub-002_S_0413` 这种 ADNI ID。

2. ADNI 下游评估不能默认读取私有数据集 config。
   - 错误风险：`--include_t1` / `--include_fa_gt` 会跑到 `data/processed/test/...`。
   - 已修复：传入 `--adni_slice_manifest` 时，默认 reference 路径改为 manifest 所在目录下的 `test/t1_slices` 和 `test/fa_slices`。

3. 图像指标评估脚本不能只读私有 Excel。
   - 已修复：`evaluate_method_folder.py` 支持 `--adni_slice_manifest`。

## 当前私有数据集主方法

当前最平衡的主方法候选是：

`FREQ_FLOWBASE_PMLOW_B035`

它不是重新训练的单一网络，而是频率融合推理：

- low source: `PM_STAGE1`
- high/source image: `PM_DIRF_FIDELITY_FLOW_FULL`
- script: `scripts/export_frequency_fusion_predictions.py`
- mode: `sharp_base_low_residual`
- sigma: `1.5`
- low_residual_gain: `0.35`

公式语义：

```text
output = high_image + 0.35 * (LP(PM_STAGE1) - LP(high_image))
```

也就是说，它保留 `Fidelity Flow` 的清晰高频，同时用 `PM_STAGE1` 的低频结构做校正。

## 私有数据集图像指标

| 方法 | 类型 | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DIRF_V5_3SLICE_K6 | 流模型对比 | 28.399 | 0.909 | 0.001531 | 0.017188 | 0.455 | 0.054 | 0.884 |
| PM_STAGE1 | PMRF posterior mean | 27.972 | 0.903 | 0.001696 | 0.018168 | 0.519 | 0.063 | 0.800 |
| PM_STAGE1_WMROI_DETAIL_5SLICE | Stage1 detail baseline | 27.741 | 0.898 | 0.001771 | 0.018417 | 0.681 | 0.060 | 0.855 |
| PM_STAGE1_LPIPS_GAN_5SLICE_FINAL | 锐化 Stage1 | 27.562 | 0.891 | 0.001850 | 0.018915 | 0.882 | 0.062 | 0.834 |
| PM_DIRF_FIDELITY_FLOW_FULL | 锐化 + flow 校正 | 27.581 | 0.892 | 0.001842 | 0.018896 | 0.869 | 0.059 | 0.867 |
| FREQ_FLOWBASE_PMLOW_B035 | 当前主方法候选 | 28.180 | 0.902 | 0.001609 | 0.017632 | 0.842 | 0.057 | 0.850 |
| UNet_CurrentSplit_E100 | 公平 U-Net | 28.236 | 0.897 | 0.001606 | 0.017834 | 0.456 | 0.057 | 0.836 |
| Pix2Pix_CurrentSplit_E100 | 公平 Pix2Pix | 28.114 | 0.904 | 0.001641 | 0.017847 | 0.641 | 0.058 | 0.871 |
| CycleGAN_E100 | 旧 CycleGAN | 25.004 | 0.858 | 0.003306 | 0.025436 | 1.168 | 0.076 | 0.878 |
| DDIM_E100_K50 | DDIM 对比 | 27.479 | 0.888 | 0.001876 | 0.019173 | 0.667 | 0.063 | 0.827 |

关键差异：

- `DIRF_V5` PSNR/SSIM 很高，但 SharpRatio 只有 0.455，视觉高频不足。
- `LPIPS+GAN Stage1` 很锐，SharpRatio 0.882，但 PSNR/SSIM/ROI 降低。
- `FREQ_FLOWBASE_PMLOW_B035` 把 PSNR 拉回 28.18，同时 SharpRatio 保持 0.842，是目前最平衡版本。
- `UNet_CurrentSplit_E100` PSNR 接近最高，但 SharpRatio 只有 0.456，说明它仍偏平滑。
- `Pix2Pix_CurrentSplit_E100` ROI-CCC 最高，但 SharpRatio 只有 0.641，不如当前主方法清晰。

## 私有数据集主下游任务 AUC

主任务：`CN+SCD vs MCI+AD`，对应任务名 `cn_scd_vs_mci_ad`。

| 方法 | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| T1_ONLY | 0.526 | 0.526 | 0.452 |
| PM_STAGE1 | 0.604 | 0.603 | 0.609 |
| Stage1_Baseline | 0.491 | 0.493 | 0.377 |
| Stage1_LPIPS_GAN | 0.491 | 0.493 | 0.499 |
| Fidelity_Flow | 0.559 | 0.559 | 0.519 |
| Fidelity_Direct | 0.512 | 0.514 | 0.484 |
| FREQ_FLOWBASE_PMLOW_B035 | 0.598 | 0.604 | 0.571 |
| UNet_CurrentSplit_E100 | 0.532 | 0.536 | 0.568 |
| Pix2Pix_CurrentSplit_E100 | 0.568 | 0.571 | 0.562 |
| CycleGAN_E100 | 0.559 | 0.559 | 0.612 |
| FA_GT | 0.537 | 0.538 | 0.641 |

解释：

- 如果只看 Macro-F1，`PM_STAGE1` 和 `FREQ_FLOWBASE_PMLOW_B035` 最强。
- 如果看 AUC，`FA_GT`、CycleGAN、PM_STAGE1、FREQ/UNet/Pix2Pix 接近，但排序和 F1 不完全一致。
- 这说明下游小测试集受切分和类别样本数影响明显，论文里不能只押一个下游数值。
- 当前主方法优势应讲成“图像保真 + 清晰度 + WM/ROI + 下游 utility 的平衡”，而不是单项下游第一。

## ADNI 当前结果状态

ADNI 当前只有 reference：

| 方法 | 含义 | 是否训练模型 |
|---|---|---|
| ADNI_T1_ONLY | 直接用 T1 当预测 FA | 否 |
| ADNI_FA_GT | 真实 FA 上界 | 否 |

ADNI reference 图像指标：

| 方法 | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADNI_T1_ONLY | 14.052 | 0.748 | 0.042848 | 0.105173 | 3.616 | 0.317 | 0.051 |
| ADNI_FA_GT | inf | 1.000 | 0.000000 | 0.000000 | 1.000 | 0.000 | 1.000 |

ADNI reference 主任务 `CN vs MCI_spectrum+AD`：

| 方法 | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| T1_ONLY | 0.563 | 0.565 | 0.550 |
| FA_GT | 0.548 | 0.548 | 0.622 |

这不是模型坏了，而是因为现在没有 ADNI 训练模型。下一步必须训练 ADNI 上的 U-Net / Pix2Pix / CycleGAN / 当前主方法，再比较。

## 下一步

1. 等当前私有数据集 CycleGAN 公平重训结束。
2. 启动 ADNI U-Net baseline。
3. 导出 ADNI U-Net 预测，跑图像指标和下游 AUC。
4. 再跑 ADNI Pix2Pix、CycleGAN。
5. 最后训练/迁移当前主方法到 ADNI，并与以上对比。

ADNI 表格里在这些模型训练完成前，只能写 `T1_ONLY` 和 `FA_GT` reference，不能写我们的主方法结果。
