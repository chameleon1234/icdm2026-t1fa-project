# U-Net Low-Frequency Fusion Probe

## Purpose

This probe checks whether the fair current-split U-Net can serve as a stronger low-frequency medical correction source for the frequency-fusion branch. The reference method remains `FREQ_FLOWBASE_B035`, which uses the sharp flow-based output as the visual base and injects a conservative posterior-mean low-frequency residual.

## Compared Variants

| Method | Low-frequency source | High-frequency / base source | Fusion setting |
| --- | --- | --- | --- |
| `FREQ_FLOWBASE_B035` | PM Stage1 residual | Flow base | `low_residual_gain=0.35` |
| `FREQ_FLOWBASE_UNETLOW_B015` | Fair U-Net residual | Flow base | `low_residual_gain=0.15` |
| `FREQ_FLOWBASE_UNETLOW_B025` | Fair U-Net residual | Flow base | `low_residual_gain=0.25` |
| `FREQ_FLOWBASE_UNETLOW_B035` | Fair U-Net residual | Flow base | `low_residual_gain=0.35` |
| `FREQ_UNETLOW_GANHIGH` | Fair U-Net low-pass | Stage1 LPIPS+GAN high-pass | `high_gain=1.0` |

## Image Metrics

| Method | PSNR | SSIM | MAE | SharpRatio | WM-MAE | ROI-CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FREQ_FLOWBASE_B035` | 28.180 | 0.902 | 0.0176 | 0.842 | 0.0571 | 0.850 |
| `UNet_CurrentSplit_E100` | 28.236 | 0.897 | 0.0178 | 0.456 | 0.0575 | 0.836 |
| `FREQ_FLOWBASE_UNETLOW_B015` | 28.358 | 0.904 | 0.0173 | 0.837 | 0.0562 | 0.851 |
| `FREQ_FLOWBASE_UNETLOW_B025` | 28.450 | 0.905 | 0.0171 | 0.829 | 0.0557 | 0.851 |
| `FREQ_FLOWBASE_UNETLOW_B035` | 28.520 | 0.907 | 0.0170 | 0.818 | 0.0554 | 0.850 |
| `FREQ_UNETLOW_GANHIGH` | 28.387 | 0.900 | 0.0175 | 0.761 | 0.0566 | 0.839 |

## Downstream Utility

Main task: `CN+SCD vs MCI+AD`, repeated 5-fold CV with 20 seeds.

| Method | Macro-F1 |
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

## Interpretation

The U-Net low residual variants improve PSNR, SSIM, MAE, and WM-MAE monotonically as the residual gain increases, while preserving acceptable sharpness. However, they do not improve the main downstream task compared with `FREQ_FLOWBASE_B035`.

This means the fair U-Net is useful as a low-frequency fidelity source, but it also injects a smoothed disease representation that weakens the main classification utility when used as a residual correction for the flow-base image.

`FREQ_UNETLOW_GANHIGH` has stronger downstream utility than `FREQ_FLOWBASE_B035`, but its sharpness and ROI-CCC are lower. It should be treated as a utility-leaning exploratory variant, not the main balanced method.

## Decision

Keep `FREQ_FLOWBASE_B035` as the balanced paper method because it has the best trade-off among visual sharpness, image fidelity, ROI consistency, and downstream utility. Use the U-Net-low variants as an ablation showing that higher paired fidelity does not automatically imply stronger disease utility.
