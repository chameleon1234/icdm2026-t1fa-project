# Pix2Pix Current-Split Probe

## Purpose

After detecting subject leakage in the legacy U-Net checkpoint, the comparison baselines need to be audited and, where necessary, retrained on the current `data/processed/train`, `data/processed/val`, and `data/processed/test` subject split.

This probe adds a reproducible current-split Pix2Pix training entry and runs a short 10-epoch sanity training before committing to a long 100-epoch baseline run.

## Code Added

- `scripts/train_pix2pix_current_split.py`
- `tests/test_train_pix2pix_current_split.py`

The training script checks subject-level split disjointness before training and saves `best_pix2pix_generator.pt` by validation PSNR.

## Probe Command

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m scripts.train_pix2pix_current_split `
  --run_name pix2pix_current_split_probe_e10 `
  --epochs 10 `
  --batch_size 16 `
  --lr 2e-4 `
  --lambda_l1 100 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 5
```

Best validation checkpoint was selected at epoch 8:

- `val_PSNR=27.3103`
- checkpoint: `outputs/pix2pix_current_split_probe_e10/checkpoints/best_pix2pix_generator.pt`

## Test-Set Image Metrics

| Method | PSNR | SSIM | MAE | SharpRatio | WM-MAE | ROI-CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Pix2Pix_CurrentSplit_E10` | 27.575 | 0.893 | 0.0188 | 1.242 | 0.0600 | 0.882 |
| `Pix2Pix_E100` | 27.760 | 0.895 | 0.0183 | 0.891 | 0.0601 | 0.856 |
| `UNet_CurrentSplit_E100` | 28.236 | 0.897 | 0.0178 | 0.456 | 0.0575 | 0.836 |
| `FREQ_FLOWBASE_B035` | 28.180 | 0.902 | 0.0176 | 0.842 | 0.0571 | 0.850 |

## Downstream Utility

Main task: `CN+SCD vs MCI+AD`, repeated 5-fold CV with 20 seeds.

| Method | Macro-F1 |
| --- | ---: |
| `Pix2Pix_CurrentSplit_E10` | 0.441 +/- 0.048 |
| `Pix2Pix_E100` | 0.530 +/- 0.055 |
| `UNet_CurrentSplit_E100` | 0.610 +/- 0.054 |
| `FREQ_FLOWBASE_B035` | 0.572 +/- 0.048 |
| `T1_ONLY` | 0.519 +/- 0.057 |
| `FA_GT` | 0.519 +/- 0.043 |

## Interpretation

The 10-epoch current-split Pix2Pix model is extremely sharp on the test set (`SharpRatio=1.242`) but its downstream utility collapses (`Macro-F1=0.441`). This is likely an early/unstable GAN state: the generator can add high-frequency contrast quickly, but the disease-relevant signal is not yet aligned with paired FA structure.

This is useful evidence for the paper narrative: sharpness alone is not sufficient. The proposed balanced frequency/flow route should be compared against GAN methods using both image metrics and downstream utility.

## Next Step

Run the full fair Pix2Pix baseline only if we need a publishable current-split GAN row:

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m scripts.train_pix2pix_current_split `
  --run_name pix2pix_current_split_e100 `
  --epochs 100 `
  --batch_size 16 `
  --lr 2e-4 `
  --lambda_l1 100 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 10
```

Expected runtime from the 10-epoch probe is roughly 3.5 hours for 100 epochs on the current GPU.
