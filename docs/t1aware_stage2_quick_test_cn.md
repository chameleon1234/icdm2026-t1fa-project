# 5-slice WMROI Detail PM-DIRF v5 快速测试

这条 quick-test 现在统一到新的抗模糊路线：先使用 5-slice WM/ROI detail Stage1 checkpoint 作为 coarse predictor，再训练带动态 condition rollout 和拆分 noisy-source rollout loss 的 Stage2 detail teacher。v5 实验建议加 `--no_auto_resume`，除非你明确要续训一个完全兼容的 checkpoint。

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

这是 Stage2 的 coarse 先验。先确认 Stage1 结构可用，再判断 Stage2 是否真的补对细节。

## 2. Stage2 v5 烟雾测试

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

现在 preset 默认使用拆分权重：`rollout_noisy_l1_weight=0.20`、`rollout_noisy_detail_weight=0.10`、`rollout_noisy_hf_weight=0.08`。这样后面可以分别调 noisy endpoint 保真、detail 监督和高频监督，不会再被一个总权重一起放大。

## 3. 训练 Stage2 v5

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

训练时一起看这些字段：`refine_ratio`、`delta_sharp`、`delta_psnr`、`delta_ssim`、`delta_wm_l1`。如果 `refine_ratio` 变大，但 `delta_psnr` 长期为负、`delta_wm_l1` 也没有改善，那说明模型只是“动得更多”，还没有“补对细节”。

## 4. 导出 K10 和 K25

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

## 5. 评估和可视化

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

新的可视化目录：

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V5_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAIL_TEACHER_V5_K25
```
