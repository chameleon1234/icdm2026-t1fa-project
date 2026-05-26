# Blur Diagnosis and Next Training Plan

## Diagnosis

The blur is not only caused by an early Stage 2 stop. The `pm_dirf_sharp_vgg` log reached 49 epochs, yet validation sharpness stayed clearly below the real FA target. Full-folder evaluation shows:

- `PM_STAGE1` sharpness ratio: `0.519`
- `PM_DIRF` default sharpness ratio: `0.449`
- `PM_DIRF_SHARP_VGG` sharpness ratio: `0.679`
- Real FA target sharpness ratio: `1.000`

On the shared 8 visual slices, `PM_STAGE1` keeps about `72.8%` of simple local high-frequency energy, and the current `PM_DIRF` export raises it to about `80.3%`. The mean absolute change from Stage 1 to Stage 2 is only `0.004`, so Stage 2 is mostly doing conservative residual correction.

Root cause:

- Stage 1 is a 2D posterior-mean predictor, so it averages away part of FA detail before Stage 2 starts.
- Stage 2 is correctly coarse-aware and one-step oriented, but the current objective still rewards endpoint PSNR/SSIM enough that it changes Stage 1 only lightly.
- VGG/LPIPS helps somewhat, but it is generic natural-image perceptual pressure and should not dominate FA fidelity.

## Code Changes

- Added `T1FAStackDataset` for odd 1/3/5-slice T1 context input.
- Fixed residual Stage 1 prediction so multi-slice input adds residuals to the center T1 slice only.
- Stage 1 now supports `--context_slices` and `--posterior_mean_preset`.
- Stage 2 now auto-detects Stage 1 input channels from the checkpoint.
- Stage 2 now includes brain-masked L1, WM-masked L1, WM gradient loss, and lightweight ROI consistency loss.
- Prediction export now supports both old single-slice and new 2.5D Stage 1 checkpoints.

## Recommended Next Run

Stage 1 should be retrained first, because the current blur originates upstream.

```powershell
conda activate dinov3test
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage1.py `
  --run_name pmrf_t1fa_stage1_3slice_pm `
  --context_slices 3 `
  --posterior_mean_preset `
  --batch_size 4 `
  --epochs 200 `
  --early_stop_patience 20 `
  --no_auto_resume
```

Then train Stage 2 on that stronger coarse predictor:

```powershell
conda activate dinov3test
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_wm_residual_3slice `
  --batch_size 4 `
  --epochs 150 `
  --source_noise_std 0.01 `
  --eval_steps 1 `
  --condition_on_coarse `
  --best_metric balanced `
  --early_stop_patience 25 `
  --degrade_patience 10 `
  --no_auto_resume
```

If PowerShell has not been initialized for conda, use:

```powershell
conda run -n dinov3test accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage1.py --run_name pmrf_t1fa_stage1_3slice_pm --context_slices 3 --posterior_mean_preset --batch_size 4 --epochs 200 --early_stop_patience 20 --no_auto_resume
conda run -n dinov3test accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --run_name pm_dirf_wm_residual_3slice --batch_size 4 --epochs 150 --source_noise_std 0.01 --eval_steps 1 --condition_on_coarse --best_metric balanced --early_stop_patience 25 --degrade_patience 10 --no_auto_resume
```

## After Training

Export and evaluate with distinct method folders:

```powershell
python scripts/export_pm_dirf_predictions.py --stage stage1 --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --output_dir outputs/icdm2026/predictions/PM_STAGE1_3SLICE --device cuda
python scripts/export_pm_dirf_predictions.py --stage stage2 --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_wm_residual_3slice/checkpoints/best_stage2.pt --output_dir outputs/icdm2026/predictions/PM_DIRF_WM_3SLICE --device cuda --auto_condition_from_ckpt
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/PM_STAGE1_3SLICE --method PM_STAGE1_3SLICE --visualize_count 8
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/PM_DIRF_WM_3SLICE --method PM_DIRF_WM_3SLICE --visualize_count 8
```

Decision rule: keep the 3-slice version only if it improves WM-masked MAE, gradient error, sharpness ratio, and visual lesion/WM tract readability without a meaningful PSNR/SSIM collapse.
