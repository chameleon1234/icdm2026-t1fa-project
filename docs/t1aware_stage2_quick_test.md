# T1-Aware PM-DIRF Stage 2 Quick Test

This test checks whether the new Stage 2 can reduce visible blur by conditioning on both coarse FA and the 3-slice T1 stack.

## Train

For a fast code-path smoke test, first run:

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_t1aware_3slice_smoke `
  --condition_mode coarse_t1 `
  --epochs 1 `
  --batch_size 1 `
  --train_limit 16 `
  --val_limit 8 `
  --fid_eval_every 999 `
  --preview_every 999 `
  --save_every 999 `
  --no_auto_resume
```

For the actual quick detail test, run:

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_t1aware_endpoint_3slice_detail `
  --condition_mode coarse_t1 `
  --t_sampling endpoint `
  --epochs 40 `
  --batch_size 2 `
  --lr 8e-5 `
  --source_noise_std 0.01 `
  --eval_steps 1 `
  --best_metric detail_paired `
  --hf_weight 0.08 `
  --grad_weight 0.10 `
  --wm_grad_weight 0.08 `
  --detail_weight 0.12 `
  --residual_hf_weight 0.20 `
  --paired_sharp_weight 4.0 `
  --detail_sharp_weight 10.0 `
  --detail_coarse_penalty_weight 12.0 `
  --no_auto_resume
```

## Export

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_t1aware_endpoint_3slice_detail/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --device cuda `
  --condition_mode auto
```

## Evaluate and Visualize

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --method PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --visualize_count 8
```

The new visual panels will be written to:

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL
```

Use this run only if `Sharpness_Ratio` and `Gradient_Error` improve without a large PSNR/MAE drop.
