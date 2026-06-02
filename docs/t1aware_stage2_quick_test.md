# 5-Slice WMROI Detail PM-DIRF v6 Quick Test

This quick-test path targets the current anti-blur Stage2 design: use the 5-slice WM/ROI detail Stage1 checkpoint as the coarse predictor, then train a v6 Stage2 detail teacher with dynamic condition rollout, delta-from-initial conditioning, decaying velocity schedule, split noisy-source rollout losses, and state-aware remaining-detail auxiliary losses.

Old v5 checkpoints can still be exported for comparison, but they were trained without the new delta condition channel and should not be treated as v6 results. Use a new `--run_name` and `--no_auto_resume` for v6.

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

## 2. Smoke Test Stage2 v6

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detail_teacher_v6_smoke `
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

The log should show `condition_channels=13`, `dynamic_condition_rollout=True`, `velocity_schedule=decaying`, and `remaining_detail=0.08/0.1`.

## 3. Train Stage2 v6

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detail_teacher_v6_5slice `
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
  --detail_delta_psnr_penalty_weight 5.0 `
  --detail_delta_ssim_penalty_weight 80.0 `
  --detail_delta_wm_penalty_weight 20.0 `
  --detail_fidelity_gate_penalty 0.30 `
  --detail_min_delta_psnr -0.005 `
  --detail_min_delta_ssim -0.0003 `
  --remaining_detail_weight 0.08 `
  --rollout_remaining_detail_weight 0.10 `
  --velocity_schedule decaying `
  --disable_lpips `
  --fid_eval_every 999 `
  --no_auto_resume
```

Watch these fields together: `refine_ratio`, `delta_sharp`, `delta_psnr`, `delta_ssim`, and `delta_wm_l1`. If `refine_ratio` rises but `delta_psnr` and `delta_wm_l1` stay negative, the model is moving more without adding paired-correct detail.

## 4. Export K10 and K25

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detail_teacher_v6_5slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V6_K10 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 10

python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detail_teacher_v6_5slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V6_K25 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 25
```

## 5. Evaluate and Visualize

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V6_K10 `
  --method PM_DIRF_DETAIL_TEACHER_V6_K10 `
  --visualize_count 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAIL_TEACHER_V6_K25 `
  --method PM_DIRF_DETAIL_TEACHER_V6_K25 `
  --visualize_count 8
```

New visual folders:

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V6_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V6_K25
```
