# Metric-Restore Fidelity Flow and Modern Baseline Plan

## Goal

The next method update should raise PSNR and SSIM while keeping the Stage 1 LPIPS+GAN visual detail. The working hypothesis is:

- Stage 1 LPIPS+GAN provides sharp FA-like high-frequency structure.
- Stage 2 Fidelity Flow should only correct low/mid-low frequency medical bias.
- Checkpoint selection must reject updates that noticeably reduce sharpness.

This is implemented by the `metric_restore` preset in:

```text
pmrf_t1fa/train_pmrf_t1fa_stage2_fidelity_corrector.py
```

The preset increases paired-fidelity losses and keeps a sharpness-retention gate:

- `frequency_cutoff >= 0.16`
- `final_l1_weight >= 0.80`
- `final_mse_weight >= 0.80`
- `final_ssim_weight >= 0.50`
- `wm_l1_weight >= 1.00`
- `roi_weight >= 0.30`
- `hf_preserve_weight >= 2.50`
- `best_min_sharp_retention >= 0.97`
- `best_min_delta_psnr >= 0`
- `best_min_delta_ssim >= 0`

Smoke result: training from scratch was unstable for this objective, but fine-tuning from the existing full Fidelity Flow checkpoint improved PSNR and SSIM while retaining sharpness. Therefore full runs should use `--init_ckpt`.

## Private Dataset Full Run

```powershell
conda activate dinov3test

python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --training_preset metric_restore `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --init_ckpt outputs/pmrf_t1fa_stage2_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt `
  --run_name pm_dirf_fidelity_flow_v2_metric_restore_full `
  --corrector_mode flow `
  --epochs 20 `
  --batch_size 2 `
  --lr 3e-5 `
  --width 48 `
  --num_blocks 8 `
  --preview_every 512 `
  --mixed_precision bf16
```

Export and evaluate:

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/pm_dirf_fidelity_flow_v2_metric_restore_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --method PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --visualize_count 8
```

## ADNI Full Run

```powershell
conda activate dinov3test

python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --training_preset metric_restore `
  --stage1_ckpt outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt `
  --init_ckpt outputs/adni_pmrf_stage2_fidelity_flow_full_e40/checkpoints/best_fidelity_corrector.pt `
  --run_name adni_pmrf_stage2_fidelity_flow_v2_metric_restore_full `
  --corrector_mode flow `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --epochs 20 `
  --batch_size 2 `
  --lr 3e-5 `
  --width 48 `
  --num_blocks 8 `
  --preview_every 512 `
  --mixed_precision bf16
```

Export and evaluate:

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/adni_pmrf_stage2_fidelity_flow_v2_metric_restore_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --method ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count 8
```

## Modern Baseline Groups

Run every method through the same export, image-metric, and downstream pipeline.

### Strong Supervised Baselines

- U-Net / Stack U-Net
- Pix2Pix
- CycleGAN
- Restormer
- NAFNet-style MeanFlow
- Swin-style MeanFlow
- U-Mamba-style MeanFlow

These methods establish whether the proposed two-stage design beats strong paired image-to-image restoration baselines on PSNR, SSIM, MAE, MSE, sharpness, WM-MAE, ROI-CCC, and downstream ACC/AUC.

### Heavy Recent Generative Baselines

The ADNI heavy baseline runner already covers:

- DDIM
- DBM / diffusion bridge
- MOTFM image-to-image flow

```powershell
conda activate dinov3test

python scripts/run_heavy_baseline_suite.py `
  --device cuda `
  --ddim_steps 50 `
  --dbm_steps 40 `
  --motfm_steps 10 `
  --visualize_count 8
```

### Reporting Rule

Do not claim the method is best from a hand-picked metric. Report:

- Image fidelity: PSNR, SSIM, MSE, MAE
- Detail: Sharpness ratio and visual comparison
- Medical consistency: WM-MAE, ROI-CCC
- Downstream utility: ACC, AUC, Macro-F1
- Runtime/sampling cost when comparing with DDIM, DBM, and MOTFM

The target result is not merely highest sharpness. The target is higher PSNR/SSIM than the current sharp method while preserving sharpness within about 3 percent and improving WM/ROI consistency.
