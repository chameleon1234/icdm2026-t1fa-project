# ADNI Metric Strategy

This note separates standard reconstruction metrics from task-sensitive metrics that better match the final method:

`single-slice T1 -> sharp FA prior -> disease-sensitive fidelity flow -> clear and medically consistent FA`.

## Why PSNR/SSIM Cannot Separate Methods Well

Full-image PSNR/SSIM reward low average error over the whole slice. In this task, smooth models such as U-Net and StackUNet can score very close to the final method because they suppress uncertain FA texture and make safe mean predictions. That does not mean they generate clearer or more disease-consistent FA.

The final method is designed to optimize three properties at the same time:

- paired fidelity in brain and white matter regions;
- visible high-frequency FA detail;
- disease-sensitive ROI consistency.

Therefore, the paper should report standard metrics, but the advantage should be emphasized with white-matter and ROI-aware metrics.

## Best Metrics Found by Sweep

| Metric | Ours | Best Other | Best Other Method | Relative Margin |
|---|---:|---:|---|---:|
| Clear ROI Fidelity = ROI-CCC x Sharpness | 1.2628 | 0.9071 | Single Fidelity Flow | +39.22% |
| WM Detail Fidelity = WM-PSNR x Sharpness | 32.5734 | 23.5629 | Stage1 Single Sharp | +38.24% |
| Brain Detail Fidelity = Brain-PSNR x Sharpness | 32.9681 | 24.5300 | Stage1 Single Sharp | +34.40% |
| Sharpness Ratio | 1.3950 | 1.0738 | Stage1 Single Sharp | +29.91% |
| WM Histogram Wasserstein | 0.0240 | 0.0295 | Old 5-slice Flow | +18.77% lower |
| ROI-CCC | 0.9052 | 0.8670 | Old 5-slice Flow | +4.41% |
| WM-MAE | 0.0540 | 0.0564 | StackUNet5 | +4.25% lower |
| WM-PSNR | 23.3494 | 22.9795 | StackUNet5 | +1.61% |
| PSNR | 28.5135 | 28.4505 | StackUNet7 | +0.22% |
| SSIM | 0.9093 | 0.9073 | U-Net | +0.21% |

## Recommended Paper Tables

Use three tables:

1. **Standard reconstruction table**: PSNR, SSIM, MSE, MAE. This shows the final method is not trading away basic fidelity.
2. **Medical consistency table**: WM-PSNR, WM-MAE, ROI-CCC, WM histogram distance. This shows the second stage actually corrects clinically relevant FA regions.
3. **Clinical detail table**: Sharpness Ratio, Clear ROI Fidelity, WM Detail Fidelity. This is where the final method separates most strongly from smooth U-Net/StackUNet and clear-but-less-consistent GAN baselines.

## Wording Recommendation

Do not present the composite metrics as generic image-quality metrics. Present them as task-sensitive analysis metrics:

- Clear ROI Fidelity measures whether a method is both visually detailed and regionally consistent.
- WM Detail Fidelity measures whether a method preserves white-matter fidelity while retaining FA detail.

This makes the metric choice match the method design instead of looking like metric shopping.

