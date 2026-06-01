# 5-Slice WMROI Detail PM-DIRF v5 Quick Test

This quick-test path is now aligned with the current anti-blur route: use the 5-slice WM/ROI detail Stage1 checkpoint as the coarse predictor, then train a Stage2 detail teacher with dynamic condition rollout and split noisy-source rollout losses. Use `--no_auto_resume` for this v5 run unless you intentionally resume an exact compatible checkpoint.

## 1. Confirm Stage1 Export

```powershell
conda activate dinov3test
python scripts/export_pm_dirf_predictions.py `
  --stage stage1 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --output_dir outputs/icdm2026/predictions/PM_STAGE1_WMROI_DETAIL_5SLICE `
  --device cuda

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_STAGE1_WMROI_DETAIL_5SLICE `
  --method PM_STAGE1_WMROI_DETAIL_5SLICE `
  --visualize_count 8
```

Inspect:

```text
outputs/icdm2026/figures/method_slices/PM_STAGE1_WMROI_DETAIL_5SLICE
```

This is the coarse prior for Stage2. Do not judge Stage2 before confirming Stage1 is at least structurally usable.

## 2. Smoke Test Stage2 v5

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detail_teacher_v5_smoke `
  --stage2_training_preset detail_teacher `
  --epochs 1 `
  --batch_size 1 `
  --train_limit 8 `
  --val_limit 4 `
  --lr 8e-5 `
  --mixed_precision no `
  --disable_lpips `
  --fid_eval_every 999 `
  --preview_every 999 `
  --save_every 999 `
  --no_auto_resume
```

The preset now sets `rollout_noisy_l1_weight=0.20`, `rollout_noisy_detail_weight=0.10`, and `rollout_noisy_hf_weight=0.08`. These are split so that noisy endpoint fidelity and noisy detail/high-frequency supervision can be tuned independently.

## 3. Train Stage2 v5

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detail_teacher_v5_5slice `
  --stage2_training_preset detail_teacher `
  --epochs 60 `
  --batch_size 1 `
  --lr 8e-5 `
  --source_noise_std 0.05 `
  --eval_steps 10 `
  --rollout_train_steps 4,8,10,25 `
  --rollout_noisy_l1_weight 0.20 `
  --rollout_noisy_detail_weight 0.10 `
  --rollout_noisy_hf_weight 0.08 `
  --detail_refine_ratio_weight 2.0 `
  --detail_under_refine_penalty_weight 2.0 `
  --detail_delta_psnr_penalty_weight 0.5 `
  --detail_delta_ssim_penalty_weight 20.0 `
  --detail_delta_wm_penalty_weight 20.0 `
  --disable_lpips `
  --fid_eval_every 999 `
  --no_auto_resume
```

Watch these log fields together: `refine_ratio`, `delta_sharp`, `delta_psnr`, `delta_ssim`, and `delta_wm_l1`. If `refine_ratio` rises but `delta_psnr` stays negative and `delta_wm_l1` does not improve, the model is moving more without adding paired-correct detail.

## 4. Export K10 and K25

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detail_teacher_v5_5slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V5_K10 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 10

python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detail_teacher_v5_5slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V5_K25 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 25
```

## 5. Evaluate and Visualize

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V5_K10 `
  --method PM_DIRF_DETAIL_TEACHER_V5_K10 `
  --visualize_count 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V5_K25 `
  --method PM_DIRF_DETAIL_TEACHER_V5_K25 `
  --visualize_count 8
```

New visual folders:

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V5_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V5_K25
```
