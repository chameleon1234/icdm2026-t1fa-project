# U-Net 低频融合探针实验

## 目的

这个实验检查公平重训后的 U-Net 是否可以作为更强的低频医学校正来源，用在频率融合分支里。当前参考主方法仍然是 `FREQ_FLOWBASE_B035`，也就是以 Flow 输出作为清晰视觉基底，再注入保守的 posterior-mean 低频残差。

## 对比变体

| 方法 | 低频来源 | 高频/基底来源 | 融合设置 |
| --- | --- | --- | --- |
| `FREQ_FLOWBASE_B035` | PM Stage1 残差 | Flow base | `low_residual_gain=0.35` |
| `FREQ_FLOWBASE_UNETLOW_B015` | 公平 U-Net 残差 | Flow base | `low_residual_gain=0.15` |
| `FREQ_FLOWBASE_UNETLOW_B025` | 公平 U-Net 残差 | Flow base | `low_residual_gain=0.25` |
| `FREQ_FLOWBASE_UNETLOW_B035` | 公平 U-Net 残差 | Flow base | `low_residual_gain=0.35` |
| `FREQ_UNETLOW_GANHIGH` | 公平 U-Net 低频 | Stage1 LPIPS+GAN 高频 | `high_gain=1.0` |

## 图像指标

| 方法 | PSNR | SSIM | MAE | SharpRatio | WM-MAE | ROI-CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FREQ_FLOWBASE_B035` | 28.180 | 0.902 | 0.0176 | 0.842 | 0.0571 | 0.850 |
| `UNet_CurrentSplit_E100` | 28.236 | 0.897 | 0.0178 | 0.456 | 0.0575 | 0.836 |
| `FREQ_FLOWBASE_UNETLOW_B015` | 28.358 | 0.904 | 0.0173 | 0.837 | 0.0562 | 0.851 |
| `FREQ_FLOWBASE_UNETLOW_B025` | 28.450 | 0.905 | 0.0171 | 0.829 | 0.0557 | 0.851 |
| `FREQ_FLOWBASE_UNETLOW_B035` | 28.520 | 0.907 | 0.0170 | 0.818 | 0.0554 | 0.850 |
| `FREQ_UNETLOW_GANHIGH` | 28.387 | 0.900 | 0.0175 | 0.761 | 0.0566 | 0.839 |

## 下游 utility

主任务：`CN+SCD vs MCI+AD`，20 个随机种子重复 5 折交叉验证。

| 方法 | Macro-F1 |
| --- | ---: |
| `UNet_CurrentSplit_E100` | 0.610 +/- 0.054 |
| `PM_STAGE1` | 0.582 +/- 0.049 |
| `FREQ_FLOWBASE_B035` | 0.572 +/- 0.048 |
| `FREQ_UNETLOW_GANHIGH` | 0.587 +/- 0.042 |
| `FREQ_FLOWBASE_UNETLOW_B015` | 0.556 +/- 0.056 |
| `FREQ_FLOWBASE_UNETLOW_B025` | 0.545 +/- 0.054 |
| `FREQ_FLOWBASE_UNETLOW_B035` | 0.552 +/- 0.052 |
| `T1_ONLY` | 0.519 +/- 0.057 |
| `FA_GT` | 0.519 +/- 0.043 |

## 解释

U-Net 低频残差变体随着残差增益增大，可以稳定提升 PSNR、SSIM、MAE 和 WM-MAE，同时 SharpRatio 仍保持在可接受范围内。但是它们没有在主下游任务上超过 `FREQ_FLOWBASE_B035`。

这说明公平 U-Net 确实是一个有效的低频保真来源，但它带来的平滑疾病表征会削弱 Flow-base 图像在主分类任务上的 utility。换句话说，更高的 paired fidelity 不必然带来更强的疾病相关 utility。

`FREQ_UNETLOW_GANHIGH` 的下游 Macro-F1 高于 `FREQ_FLOWBASE_B035`，但 SharpRatio 和 ROI-CCC 更低，所以它更适合作为 utility-oriented 探索变体，而不是主论文里的平衡方法。

## 决策

保留 `FREQ_FLOWBASE_B035` 作为论文主方法，因为它在视觉清晰度、图像保真、ROI 一致性和下游 utility 之间最均衡。U-Net-low 变体作为消融，用来证明“提高配对保真并不自动等于提高疾病 utility”。
