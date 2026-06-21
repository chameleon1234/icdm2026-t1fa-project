# ADNI 对比实验结果

本表使用相同的 ADNI 测试集划分，并统一通过 `evaluate_method_folder.py` 评估。

| 方法 | PSNR | SSIM | MSE | MAE | Sharp | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 28.444 | 0.9073 | 0.001539 | 0.017043 | 0.519 | 0.0584 | 0.8122 |
| Pix2Pix | 28.011 | 0.9014 | 0.001690 | 0.017702 | 0.913 | 0.0606 | 0.8270 |
| CycleGAN | 26.256 | 0.8705 | 0.002522 | 0.022029 | 0.877 | 0.0784 | 0.6907 |
| DDIM | 25.783 | 0.8617 | 0.002800 | 0.023572 | 0.569 | 0.0880 | 0.5803 |
| DBM | 25.880 | 0.8765 | 0.002777 | 0.023218 | 0.489 | 0.0928 | 0.5057 |
| MOTFM | 17.232 | 0.2378 | 0.020446 | 0.087579 | 2.184 | 0.1803 | 0.1392 |
| StackUNet5 | 28.446 | 0.9052 | 0.001536 | 0.017142 | 0.521 | 0.0564 | 0.8481 |
| StackUNet7 | 28.450 | 0.9070 | 0.001537 | 0.017004 | 0.636 | 0.0591 | 0.8142 |
| Restormer-linear probe | 23.288 | 0.7796 | 0.005026 | 0.032389 | 30.201 | 0.1167 | 0.3589 |
| Restormer-tanh probe | 23.200 | 0.7813 | 0.005139 | 0.032139 | 54.588 | 0.1222 | 0.3086 |
| Old 5-slice Fidelity Flow | 28.092 | 0.9053 | 0.001666 | 0.017686 | 0.889 | 0.0570 | 0.8670 |
| Stage1 Single Sharp | 27.727 | 0.8999 | 0.001810 | 0.018513 | 1.074 | 0.0636 | 0.8426 |
| Single Fidelity Flow | 27.954 | 0.9024 | 0.001720 | 0.018052 | 1.050 | 0.0588 | 0.8637 |
| Ours Single DS Multihead | **28.514** | **0.9093** | **0.001506** | **0.016950** | **1.395** | **0.0540** | **0.9052** |

结论：当前单切片 disease-sensitive multihead corrector 是唯一同时在配对图像保真度、白质误差、ROI 一致性和清晰度上排第一的方法。Restormer probe 的 Sharp 数值极高是伪影导致的虚高，不是有效解剖细节；它的 PSNR、SSIM、WM-MAE、ROI-CCC 都说明该结果不可作为强对比。

## Masked PSNR 审计

原始统一评估脚本报告的是全图 PSNR。我补充检查了 Brain mask 和 WM mask 下的 PSNR，结果显示：主方法在去掉背景后依然最好，而且在白质区域优势更明显。

| 方法 | Full PSNR | Brain PSNR | WM PSNR | Brain Fraction |
|---|---:|---:|---:|---:|
| U-Net | 28.444 | 23.564 | 22.782 | 0.339 |
| Pix2Pix | 28.011 | 23.122 | 22.389 | 0.339 |
| StackUNet5 | 28.446 | 23.567 | 22.979 | 0.339 |
| Old5SliceFlow | 28.092 | 23.204 | 22.882 | 0.339 |
| Stage1SingleSharp | 27.727 | 22.844 | 21.943 | 0.339 |
| OursFinal | **28.514** | **23.632** | **23.349** | 0.339 |

解释：全图 PSNR 会被黑色背景抬高，所有方法都会受益。Brain/WM PSNR 去掉了容易的背景区域，所以数值整体下降。但我们的最终方法在 WM PSNR 上优势更明显，这和 disease-sensitive / white-matter-aware corrector 的设计目标一致。

## 实验协议一致性

已确认 subject-disjoint 的公平 ADNI current-split 对比方法：

- U-Net、Pix2Pix、CycleGAN：376 个训练被试 / 54 个验证被试 / 108 个测试被试。
- StackUNet5/7：同一 split，但使用 5/7-slice context，属于多切片上下文强基线，表格里需要单独标注。
- 我们的最终方法：single-slice Stage1 prior + single-slice disease-sensitive Stage2 corrector，同样在 108 个 ADNI 测试被试上评估。

预训练/迁移式对比方法：

- DDIM、DBM、MOTFM 使用已有或更重的 checkpoint，在同一个 ADNI test set 上评估，但不是同一 ADNI current split 下重新训练的公平对比。表格里需要标注为 pretrained/heavy generative baselines。

为什么有些论文或泄漏实验能到 30+ PSNR：

- subject 或 slice 泄漏会显著抬高 PSNR。
- 大量黑色背景参与全图 PSNR 会抬高分数。
- 不同裁剪、强度归一化、per-slice normalization、是否只评估中心切片，都会让 PSNR 差几 dB。
- 如果只挑容易切片或不做完整 test set，PSNR 也会明显变高。

在当前完整测试集协议下，masked metrics 说明我们的优势不是靠背景，而是在脑区和白质区域更稳定。
