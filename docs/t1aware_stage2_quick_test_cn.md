# T1-aware PM-DIRF Stage 2 快速测试

这个测试用来判断新的 Stage 2 是否能缓解 FA 生成图像模糊：新的 refinement flow 不再只看 coarse FA，而是同时条件化 coarse FA 和 3-slice T1 stack。

## 训练

先用下面这个命令做代码路径 smoke test：

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

真正用于判断细节是否改善的快速实验再跑：

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

## 导出

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_t1aware_endpoint_3slice_detail/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --device cuda `
  --condition_mode auto
```

## 评估和可视化

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --method PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL `
  --visualize_count 8
```

新的可视化结果会写到：

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_T1AWARE_ENDPOINT_3SLICE_DETAIL
```

只有在 `Sharpness_Ratio` 和 `Gradient_Error` 改善、同时 PSNR/MAE 没有明显下降时，才把这条线作为下一版主实验。
