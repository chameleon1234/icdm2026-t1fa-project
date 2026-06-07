# 纯对比实验核查

## 结论

`Pix2Pix_CurrentSplit_E100` 和 `UNet_CurrentSplit_E100` 不是直接复制 GT，也不是直接复制 T1。它们的 train/val/test subject 是分离的，训练 summary 中的划分为：

- train: 173 subjects / 8650 slices
- val: 37 subjects / 1850 slices
- test: 38 subjects

它们看起来“指标好”的主要原因不是泄漏，而是：

- U-Net 是强回归 baseline，倾向于输出平滑的条件均值，所以 PSNR/MAE 容易高。
- Pix2Pix 使用标准 `lambda_l1=100`，本质上也有很强的 L1 回归约束，因此 PSNR/SSIM 不会很差。
- 两者的清晰度并不强：U-Net SharpRatio 只有 0.456，Pix2Pix SharpRatio 只有 0.641。

真正不该放进纯对比实验的是旧结果：

- `UNet_E99`: PSNR=36.99，明显异常，旧 split/旧流程结果，必须排除。
- `Pix2Pix_E100`: 旧结果，未证明当前 split 公平重训，排除。
- `CycleGAN_E100`: 旧 CycleGAN，已由公平 current split 的 `CycleGAN_CurrentSplit_E100` 替换。
- `DDIM_E100_K50`: 当前无法证明是同一公平 split 重训，暂时排除，后面需要重新训练。
- `FREQ_FLOWBASE_PMLOW_B035`: 这是我们的方法，不是纯 baseline。
- `PM_STAGE1` / `PM_STAGE1_LPIPS_GAN` / `PM_DIRF_FIDELITY_FLOW`: 都是我们方法的消融，不是外部对比方法。

## 新的纯对比实验表

脚本：

```powershell
python scripts/build_pure_comparison_tables.py `
  --output_root outputs/icdm2026/tables/pure_comparison_clean
```

默认只生成纯 baseline：

| 方法 | 类型 | 状态 |
|---|---|---|
| `UNet_CurrentSplit_E100` | CNN | 公平 current split 已完成 |
| `Pix2Pix_CurrentSplit_E100` | GAN | 公平 current split 已完成 |
| `CycleGAN_CurrentSplit_E100` | GAN | 公平 current split 已完成 |
| `DIRF_V5_3SLICE_K6` | Flow | 项目内 flow baseline |

如果需要论文主方法对比，另生成带 ours/reference 的表：

```powershell
python scripts/build_pure_comparison_tables.py `
  --include_ours `
  --include_references `
  --output_root outputs/icdm2026/tables/pure_comparison_with_ours
```

## 当前纯 baseline 图像指标

| 方法 | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net current split | 28.236 | 0.897 | 0.001606 | 0.017834 | 0.456 | 0.057 | 0.836 |
| Pix2Pix current split | 28.114 | 0.904 | 0.001641 | 0.017847 | 0.641 | 0.058 | 0.871 |
| CycleGAN current split | 26.166 | 0.871 | 0.002556 | 0.022473 | 0.864 | 0.068 | 0.811 |
| DIRF_V5 K6 | 28.399 | 0.909 | 0.001531 | 0.017188 | 0.455 | 0.054 | 0.884 |

## 当前纯 baseline 主下游任务

任务：`CN+SCD vs MCI+AD`

| 方法 | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| U-Net current split | 0.532 | 0.536 | 0.568 |
| Pix2Pix current split | 0.568 | 0.571 | 0.562 |
| CycleGAN current split | 0.627 | 0.638 | 0.594 |
| DIRF_V5 K6 | 0.460 | 0.461 | 0.374 |

## 加入我们方法后的对比

`FREQ_FLOWBASE_B035` 是当前主方法候选，不属于纯 baseline，但可以放在论文主表里作为 ours：

| 方法 | Macro-F1 | Balanced Acc | AUC |
|---|---:|---:|---:|
| FREQ_FLOWBASE_B035 | 0.598 | 0.604 | 0.571 |

图像指标：

| 方法 | PSNR | SSIM | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|
| FREQ_FLOWBASE_B035 | 28.180 | 0.902 | 0.842 | 0.057 | 0.850 |

## 后续动作

1. `CycleGAN_CurrentSplit_E100` 已完成公平重训、导出、图像评估和主下游任务，可以进入纯对比实验。
2. DDIM 必须重新按 current split 训练，旧 `DDIM_E100_K50` 暂时不进纯对比实验。
3. 论文表格分三类：
   - reference: `T1_ONLY`, `FA_GT`
   - pure baselines: U-Net / Pix2Pix / CycleGAN current split / DDIM current split / DIRF
   - ours and ablations: FREQ、PM_STAGE1、LPIPS+GAN、Fidelity Flow
