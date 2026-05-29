# Detail-head Stage1 引导多步 PM-DIRF 快速测试

这次测试直接针对“生成 FA 图模糊”的问题。之前的 `PM_DIRF_STAGE1GUIDED_MULTISTEP_K10` 仍然是单头 velocity 模型，所以 Stage 2 还是容易被 L1/SSIM/PSNR 拉回平滑的 posterior mean。这一版使用 `--stage2_model_variant detail`，把 Stage 2 拆成 average-velocity head 和 detail-velocity head，再在多步 rollout 里用 `detail_boost` 放大细节分支。

## Smoke Test

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detailhead_stage1guided_smoke `
  --stage2_model_variant detail `
  --condition_mode coarse_t1_edge `
  --detail_boost 0.90 `
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

## 快速可视化测试

如果当前 GPU 已经被桌面、浏览器或其他程序占用很多显存，Stage1 推理可能会 OOM。这种情况下可以额外加 `--stage1_device cpu`，Stage2 仍然在加速设备上训练，只是 Stage1 coarse 预测临时 offload 到 CPU。

```powershell
conda activate dinov3test
python -m pmrf_t1fa.train_pmrf_t1fa_stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --run_name pm_dirf_detailhead_stage1guided_3slice `
  --stage2_model_variant detail `
  --condition_mode coarse_t1_edge `
  --detail_boost 0.90 `
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
  --detail_velocity_weight 0.18 `
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

## 导出 K10

```powershell
      python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detailhead_stage1guided_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K10 `
        --device cuda `
        --condition_mode auto `
        --eval_steps 10
```

## 可选导出 K25

```powershell
python scripts/export_pm_dirf_predictions.py `
  --stage stage2 `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt `
  --stage2_ckpt outputs/pm_dirf_detailhead_stage1guided_3slice/checkpoints/best_stage2.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K25 `
  --device cuda `
  --condition_mode auto `
  --eval_steps 25
```

## 评估和可视化

```powershell
python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K10 `
  --method PM_DIRF_DETAILHEAD_STAGE1GUIDED_K10 `
  --visualize_count 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K25 `
  --method PM_DIRF_DETAILHEAD_STAGE1GUIDED_K25 `
  --visualize_count 8
```

新的可视化结果会写入：

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K10
outputs/icdm2026/figures/method_slices/PM_DIRF_DETAILHEAD_STAGE1GUIDED_K25
```

只有在 K10 或 K25 肉眼白质细节明显变清晰，并且没有明显假纹理、PSNR/SSIM 没有大幅崩掉时，才把这条线作为下一版主实验候选。
