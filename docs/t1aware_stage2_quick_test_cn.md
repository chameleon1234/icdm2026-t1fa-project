# 5-slice WMROI Detail PM-DIRF v6 快速测试

这条 quick-test 对应当前的抗模糊 Stage2 设计：先用 5-slice WM/ROI detail Stage1 checkpoint 作为 coarse predictor，再训练 v6 Stage2 detail teacher。v6 包含动态 condition rollout、delta-from-initial 条件通道、decaying velocity schedule、拆分 noisy-source rollout loss，以及 state-aware remaining-detail 辅助损失。

旧 v5 checkpoint 仍然可以导出做对比，但它们训练时没有新的 delta 条件通道，不应该当成 v6 结果。v6 请使用新的 `--run_name`，并加 `--no_auto_resume`。

## 1. 先确认 Stage1 导出

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

重点检查：

```text
outputs/icdm2026/figures/method_slices/PM_STAGE1_WMROI_DETAIL_5SLICE
```

## 2. Stage2 v6 烟雾测试

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

日志里应该看到 `condition_channels=13`、`dynamic_condition_rollout=True`、`velocity_schedule=decaying`、`remaining_detail=0.08/0.1`。

## 3. 训练 Stage2 v6

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

训练时一起看这些字段：`refine_ratio`、`delta_sharp`、`delta_psnr`、`delta_ssim`、`delta_wm_l1`。如果 `refine_ratio` 变大，但 `delta_psnr` 和 `delta_wm_l1` 长期为负，说明模型只是“动得更多”，还没有“补对细节”。

## 4. 导出 K10 和 K25

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

## 5. 评估和可视化

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

新的可视化目录：

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V6_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V6_K25
```
