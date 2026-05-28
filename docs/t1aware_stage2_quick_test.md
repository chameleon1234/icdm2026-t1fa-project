# Stage1-Guided Multi-Step PM-DIRF Quick Test

This test targets the blur issue directly. The Stage 2 refiner is trained with DIRF-style multi-step rollout loss and the `coarse_t1_edge` condition, which keeps coarse FA, the full T1 stack, T1 edge cues, coarse edge cues, and the Stage1 coarse-minus-T1 residual in the conditioning path.

## Smoke Test

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_stage1guided_multistep_smoke `
  --condition_mode coarse_t1_edge `
  --t_sampling endpoint `
  --rollout_train_steps 2,4 `
  --eval_steps 4 `
  --epochs 1 `
  --batch_size 1 `
  --train_limit 8 `
  --val_limit 4 `
  --fid_eval_every 999 `
  --preview_every 999 `
  --save_every 999 `
  --disable_lpips `
  --no_auto_resume
```

## Quick Visual Test

If your GPU is already heavily occupied and Stage1 inference causes OOM, add `--stage1_device cpu`. Stage2 still trains on the accelerator device, but Stage1 coarse prediction is offloaded.

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_stage1guided_multistep_3slice `
  --condition_mode coarse_t1_edge `
  --t_sampling endpoint `
  --rollout_train_steps 4,8,10 `
  --epochs 40 `
  --batch_size 1 `
  --lr 8e-5 `
  --source_noise_std 0.01 `
  --eval_steps 10 `
  --best_metric detail_paired `
  --hf_weight 0.10 `
  --residual_hf_weight 0.25 `
  --rollout_hf_weight 0.30 `
  --rollout_residual_hf_weight 0.35 `
  --rollout_l1_weight 0.30 `
  --rollout_ssim_weight 0.30 `
  --rollout_wm_l1_weight 0.12 `
  --rollout_wm_grad_weight 0.08 `
  --detail_weight 0.12 `
  --grad_weight 0.10 `
  --wm_grad_weight 0.08 `
  --paired_sharp_weight 5.0 `
  --detail_sharp_weight 12.0 `
  --detail_coarse_penalty_weight 8.0 `
  --disable_lpips `
  --no_auto_resume
```

## Export K10

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_stage1guided_multistep_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_STAGE1GUIDED_MULTISTEP_K10 `
  --device cuda `
  --condition_mode auto `
  --eval_steps 10
```

## Optional Export K25

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_stage1guided_multistep_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_STAGE1GUIDED_MULTISTEP_K25 `
  --device cuda `
  --condition_mode auto `
  --eval_steps 25
```

## Evaluate and Visualize

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_STAGE1GUIDED_MULTISTEP_K10 `
  --method PM_DIRF_STAGE1GUIDED_MULTISTEP_K10 `
  --visualize_count 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_STAGE1GUIDED_MULTISTEP_K25 `
  --method PM_DIRF_STAGE1GUIDED_MULTISTEP_K25 `
  --visualize_count 8
```

The new visual panels are written to:

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_STAGE1GUIDED_MULTISTEP_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_STAGE1GUIDED_MULTISTEP_K25
```

Keep this line only if K10 or K25 visibly improves white-matter detail without obvious false texture or major PSNR/SSIM collapse.
