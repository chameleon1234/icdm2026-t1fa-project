# Fair U-Net Current-Split Result

## Training

The U-Net baseline was retrained on the current ICDM split:

- Train: `data/processed/train`, 173 subjects, 8650 slices
- Validation: `data/processed/val`, 37 subjects, 1850 slices
- Test: `data/processed/test`, 38 subjects, 1900 slices
- Subject overlap: train/val = 0, train/test = 0, val/test = 0
- Checkpoint: `outputs/unet_current_split_e100/checkpoints/best_unet.pt`

The best validation PSNR was selected automatically during training.

## Test Image Metrics

| Method | PSNR | SSIM | MSE | MAE | SharpRatio | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net current split | 28.236 | 0.8973 | 0.001606 | 0.017834 | 0.456 | 0.0575 | 0.836 |

## Downstream Utility

Main task: CN+SCD vs MCI+AD, repeated 20-seed CV.

| Method | Macro-F1 |
|---|---:|
| T1 only | 0.519 +/- 0.057 |
| U-Net current split | 0.610 +/- 0.054 |
| FREQ_FLOWBASE_B035 | 0.572 +/- 0.048 |
| PM Stage1 | 0.582 +/- 0.049 |
| FA GT | 0.519 +/- 0.043 |

## Interpretation

The retrained U-Net is now a fair baseline. Its image fidelity is normal and no longer inflated by leakage. It performs well on the main downstream task, but its sharpness is very low, close to DIRF V5. In the paper, it should be described as a strong low-frequency supervised regression baseline, not a visually sharp generator.
