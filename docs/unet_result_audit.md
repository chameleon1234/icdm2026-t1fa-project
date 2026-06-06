# U-Net Result Audit

## Conclusion

The current `UNet_E99` result is not a valid fair comparison for the ICDM split.

The checkpoint `outputs/unet_training/checkpoints/unet_epoch_0099.pth` was trained with an older subject split from `train_unet.py`, not the current `data/processed/dataset_splits.json` split. In the current ICDM test set, 34 of 38 subjects appear in the legacy U-Net training-subject list.

## Evidence

| Group within current test set | Subjects | Slices | PSNR | SSIM | MAE | WM-MAE | SharpRatio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seen by legacy U-Net training | 34 | 1700 | 38.03 | 0.9786 | 0.00558 | 0.01732 | 0.6826 |
| Unseen by legacy U-Net training | 4 | 200 | 28.10 | 0.9055 | 0.01767 | 0.05906 | 0.5924 |

The original U-Net validation file also reports a normal value: PSNR 27.8819, SSIM 0.9035, MAE 0.0177. This matches the unseen-subject subset and contradicts the inflated full-test result.

## Interpretation

The high full-test value, PSNR 36.99 and SSIM 0.9709, is caused by subject-level leakage from the old U-Net split into the current ICDM test split. It should be treated as an invalid/leakage-contaminated reference, not as a SOTA baseline.

## Action

Retrain U-Net on the current `data/processed/train` split, validate on `data/processed/val`, export to the same PNG format, and rerun image and downstream metrics. Until then, use the current U-Net only as a leakage sanity check.
