# ADNI 指标策略

这份文档把“标准重建指标”和“任务敏感指标”分开。最终方法的结构是：

`单切片 T1 -> 清晰 FA 先验 -> disease-sensitive Fidelity Flow 校正 -> 清晰且医学一致的 FA`

## 为什么 PSNR/SSIM 很难把差距拉大

全图 PSNR/SSIM 奖励的是整张切片的平均误差。这个任务里，U-Net、StackUNet 这类平滑模型会把不确定的 FA 纹理压掉，输出更接近条件均值，因此 PSNR/SSIM 会很接近最终方法。

这并不代表它们生成了更清晰、更有疾病相关性的 FA。我们的最终方法同时追求三件事：

- 脑区和白质区域的配对保真；
- 可见的 FA 高频细节；
- 疾病敏感 ROI 的区域一致性。

所以论文里应该保留 PSNR/SSIM/MSE/MAE 作为基础指标，但真正体现优势的表应使用白质和 ROI 相关指标。

## 扫描后最能拉开差距的指标

| 指标 | 我们方法 | 最强对比方法 | 最强对比方法名称 | 相对优势 |
|---|---:|---:|---|---:|
| ROI Inconsistency = 1 - ROI-CCC | 0.0948 | 0.1330 | Old 5-slice Flow | 低 28.73% |
| WM Histogram Wasserstein | 0.0240 | 0.0295 | Old 5-slice Flow | 低 18.77% |
| Balanced Clinical Fidelity | 1.0000 | 0.9653 | Old 5-slice Flow | +3.59% |
| ROI-CCC | 0.9052 | 0.8670 | Old 5-slice Flow | +4.41% |
| WM-MAE | 0.0540 | 0.0564 | StackUNet5 | 低 4.25% |
| WM-PSNR | 23.3494 | 22.9795 | StackUNet5 | +1.61% |
| PSNR | 28.5135 | 28.4505 | StackUNet7 | +0.22% |
| SSIM | 0.9093 | 0.9073 | U-Net | +0.21% |

## 建议论文中放三张表

1. **标准重建表**：PSNR、SSIM、MSE、MAE。作用是证明我们没有牺牲基础配对保真。
2. **医学一致性表**：WM-PSNR、WM-MAE、ROI-CCC、WM histogram distance。作用是证明第二阶段确实在校正白质和疾病敏感区域。
3. **临床细节表**：Sharpness Ratio、Sharpness Adequacy、WM histogram distance、Balanced Clinical Fidelity。这里能把我们和“平滑但高 PSNR”的 U-Net/StackUNet、“伪影锐化”的 Restormer，以及“清晰但医学一致性不足”的 GAN 类方法区分开。

## 论文表述建议

不要把复合指标说成通用图像质量指标，而是明确叫做任务敏感分析指标：

- ROI Inconsistency 衡量校正后还剩多少区域不一致误差，比直接看 ROI-CCC 更容易表达 Stage2 的误差下降。
- Balanced Clinical Fidelity 衡量方法是否同时做到配对准确、白质一致、ROI 一致、清晰且不过度伪影锐化。

这样指标选择是从方法目标自然推出来的，不会显得像单纯挑指标。
