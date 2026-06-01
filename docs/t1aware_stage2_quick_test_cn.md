# Detail-Stage1 PM-DIRF 快速测试

这次 K10/K25 的 Stage2-only detail-head 版本仍然模糊，核心原因是它们还在细化一个已经很平滑的 Stage1 posterior-mean checkpoint。这个快速测试把“抗模糊”的动作前移到 Stage1：先训练 3-slice base+detail 双头 Stage1，用 `detail_paired` 选择更清晰的 checkpoint，再让 Stage2 从这个更清晰的 coarse FA 上做细化。这里的 `detail_paired` 使用有上限的 sharpness bonus，目标是奖励缺失高频结构，而不是把明显过锐的噪声 checkpoint 选出来。

断点续训默认开启。重复运行同一个 `--run_name` 会自动从 `healthy_latest_*.pt` 或 `latest_*.pt` 继续；如果要指定某个 epoch checkpoint，用 `--resume_from path\to\checkpoint.pt`。注意 `--epochs` 表示最终训练到第几代，不是额外再训练几代。

## 1. Stage1 Detail 烟雾测试

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --run_name pmrf_t1fa_stage1_detail_smoke `
  --context_slices 3 `
  --stage1_model_variant detail `
  --stage1_prediction_mode residual `
  --stage1_detail_scale 0.60 `
  --best_metric detail_paired `
  --epochs 1 `
  --batch_size 1 `
  --train_limit 8 `
  --val_limit 4 `
  --fid_eval_every 999 `
  --preview_every 999 `
  --save_every 999 `
  --disable_lpips `
  --detail_hf_weight 0.35 `
  --detail_lap_weight 0.20 `
  --brain_l1_weight 0.08 `
  --wm_l1_weight 0.16 `
  --wm_grad_weight 0.08 `
  --roi_consistency_weight 0.04 `
  --paired_sharp_weight 8.0 `
  --detail_target_sharp_ratio 0.90 `
  --detail_max_sharp_ratio 1.20 `
  --detail_oversharp_penalty_weight 12.0
```

## 2. 训练 Stage1 Detail 候选模型

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --run_name pmrf_t1fa_stage1_detail_3slice `
  --context_slices 3 `
  --stage1_model_variant detail `
  --stage1_prediction_mode residual `
  --stage1_detail_scale 0.60 `
  --best_metric detail_paired `
  --epochs 80 `
  --batch_size 2 `
  --lr 8e-5 `
  --mse_weight 0.35 `
  --l1_start_weight 0.80 `
  --l1_end_weight 0.45 `
  --ssim_start_weight 0.65 `
  --ssim_end_weight 0.35 `
  --grad_weight 0.12 `
  --hf_weight 0.12 `
  --detail_hf_weight 0.35 `
  --detail_lap_weight 0.20 `
  --brain_l1_weight 0.08 `
  --wm_l1_weight 0.16 `
  --wm_grad_weight 0.08 `
  --roi_consistency_weight 0.04 `
  --paired_sharp_weight 8.0 `
  --detail_target_sharp_ratio 0.90 `
  --detail_max_sharp_ratio 1.20 `
  --detail_oversharp_penalty_weight 12.0 `
  --disable_lpips `
  --fid_eval_every 999
```

## 3. 先导出并检查 Stage1

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage1 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_detail_3slice/checkpoints/best_stage1.pt `
  --output_dir outputs/icdm2026/predictions/PM_STAGE1_DETAIL_3SLICE `
  --device cuda

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_STAGE1_DETAIL_3SLICE `
  --method PM_STAGE1_DETAIL_3SLICE `
  --visualize_count 8
```

重点看这个目录：

```text
outputs/icdm2026/figures/method_slices/PM_STAGE1_DETAIL_3SLICE
```

只有当 Stage1 本身比 `PM_STAGE1_3SLICE` 肉眼更清晰时，才继续做 Stage2。

## 4. 用 Detail Stage1 训练 Stage2

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_detail_3slice/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detailstage1_stage2_3slice `
  --stage2_training_preset detail_teacher `
  --stage2_model_variant detail `
  --condition_mode coarse_t1_edge `
  --detail_boost 0.90 `
  --t_sampling endpoint `
  --rollout_train_steps 4,8,10 `
  --epochs 40 `
  --batch_size 1 `
  --lr 8e-5 `
  --source_noise_std 0.05 `
  --eval_steps 10 `
  --best_metric detail_paired `
  --rollout_source_mode both `
  --rollout_noisy_weight 0.35 `
  --detail_weight 0.25 `
  --hf_weight 0.10 `
  --residual_hf_weight 0.25 `
  --detail_velocity_weight 0.18 `
  --rollout_detail_weight 0.35 `
  --rollout_hf_weight 0.30 `
  --rollout_residual_hf_weight 0.35 `
  --rollout_l1_weight 0.30 `
  --rollout_ssim_weight 0.30 `
  --rollout_wm_l1_weight 0.12 `
  --rollout_wm_grad_weight 0.08 `
  --dynamic_condition_rollout `
  --grad_weight 0.10 `
  --wm_grad_weight 0.08 `
  --paired_sharp_weight 5.0 `
  --detail_sharp_weight 12.0 `
  --detail_coarse_penalty_weight 8.0 `
  --detail_refine_ratio_weight 4.0 `
  --detail_refine_target_ratio 0.50 `
  --detail_under_refine_penalty_weight 4.0 `
  --detail_delta_psnr_penalty_weight 0.5 `
  --detail_delta_ssim_penalty_weight 20.0 `
  --detail_delta_wm_penalty_weight 20.0 `
  --detail_target_sharp_ratio 0.90 `
  --detail_max_sharp_ratio 1.20 `
  --detail_oversharp_penalty_weight 12.0 `
  --disable_lpips `
  --degrade_check_mode teacher `
  --disable_rollback_on_degrade
```

## 5. 导出并可视化 K10/K25

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_detail_3slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detailstage1_stage2_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAILSTAGE1_K10 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 10

python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_detail_3slice/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detailstage1_stage2_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAILSTAGE1_K25 `
  --device cuda `
  --condition_mode auto `
  --dynamic_condition_rollout `
  --eval_steps 25

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAILSTAGE1_K10 `
  --method PM_DIRF_DETAILSTAGE1_K10 `
  --visualize_count 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAILSTAGE1_K25 `
  --method PM_DIRF_DETAILSTAGE1_K25 `
  --visualize_count 8
```

新的可视化目录：

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAILSTAGE1_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAILSTAGE1_K25
```
