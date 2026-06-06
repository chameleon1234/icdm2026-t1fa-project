# 公平 U-Net 当前划分结果

## 训练设置

U-Net 已经按照当前 ICDM split 重新训练：

- Train：`data/processed/train`，173 个受试者，8650 张切片
- Validation：`data/processed/val`，37 个受试者，1850 张切片
- Test：`data/processed/test`，38 个受试者，1900 张切片
- 受试者重叠：train/val = 0，train/test = 0，val/test = 0
- checkpoint：`outputs/unet_current_split_e100/checkpoints/best_unet.pt`

训练过程中按 validation PSNR 自动选择 best checkpoint。

## 测试集图像指标

| Method | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net current split | 28.236 | 0.8973 | 0.001606 | 0.017834 | 0.456 | 0.0575 | 0.836 |

## 下游任务

主任务：CN+SCD vs MCI+AD，20 个随机种子的 repeated CV。

| Method | Macro-F1 |
|---|---:|
| T1 only | 0.519 +/- 0.057 |
| U-Net current split | 0.610 +/- 0.054 |
| FREQ_FLOWBASE_B035 | 0.572 +/- 0.048 |
| PM Stage1 | 0.582 +/- 0.049 |
| FA GT | 0.519 +/- 0.043 |

## 解释

重新训练后的 U-Net 是公平 baseline。它的图像指标回到正常范围，不再有泄漏导致的虚高。它在主下游任务上表现不错，但 SharpRatio 只有 0.456，视觉上仍然偏糊，接近 DIRF V5 的低清晰度状态。因此论文里应该把它描述为“强低频监督回归 baseline”，而不是清晰生成器。
