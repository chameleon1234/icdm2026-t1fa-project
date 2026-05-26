# 模糊问题诊断与下一轮训练计划

## 诊断结论

现在的模糊不只是 Stage 2 “7 代早停”造成的。`pm_dirf_sharp_vgg` 日志实际已经训练到 49 个 epoch，但验证集 sharpness 仍明显低于真实 FA。当前整套测试集评估结果显示：

- `PM_STAGE1` sharpness ratio: `0.519`
- `PM_DIRF` default sharpness ratio: `0.449`
- `PM_DIRF_SHARP_VGG` sharpness ratio: `0.679`
- 真实 FA target sharpness ratio: `1.000`

在固定的 8 张可视化切片上，`PM_STAGE1` 只保留了真实 FA 约 `72.8%` 的简单局部高频能量；当前 `PM_DIRF` 导出结果提高到约 `80.3%`。但是 Stage 2 相比 Stage 1 的平均像素改变量只有 `0.004`，说明 Stage 2 主要是在做保守的小幅 residual correction。

根因判断：

- Stage 1 目前是 2D posterior-mean predictor，会在 Stage 2 之前先平均掉一部分 FA 细节。
- Stage 2 的 coarse-aware、one-step 方向是对的，但当前目标函数仍强烈偏向 endpoint PSNR/SSIM，所以它只会轻微修正 Stage 1。
- VGG/LPIPS 有一定帮助，但它是自然图像感知约束，不应该成为 FA 任务的主导锐化来源。

## 本次代码修改

- 新增 `T1FAStackDataset`，支持 1/3/5 张相邻 T1 切片作为输入通道。
- 修复 Stage 1 residual 预测：多切片输入时 residual 只加回中心 T1 切片，避免广播成多通道输出。
- Stage 1 新增 `--context_slices` 和 `--posterior_mean_preset`。
- Stage 2 自动从 Stage 1 checkpoint 推断输入通道数。
- Stage 2 新增 brain-masked L1、WM-masked L1、WM gradient loss、轻量 ROI consistency loss。
- 预测导出脚本同时兼容旧的单切片 Stage 1 和新的 2.5D Stage 1 checkpoint。

## 推荐下一轮训练

建议先重训 Stage 1，因为当前模糊的源头更靠前。

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

然后基于新的 Stage 1 训练 Stage 2：

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

如果 PowerShell 还没有初始化 conda，可以用备用写法：

```powershell
conda run -n dinov3test accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage1.py --run_name pmrf_t1fa_stage1_3slice_pm --context_slices 3 --posterior_mean_preset --batch_size 4 --epochs 200 --early_stop_patience 20 --no_auto_resume
conda run -n dinov3test accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --run_name pm_dirf_wm_residual_3slice --batch_size 4 --epochs 150 --source_noise_std 0.01 --eval_steps 1 --condition_on_coarse --best_metric balanced --early_stop_patience 25 --degrade_patience 10 --no_auto_resume
```

## 训练后要做什么

使用独立方法名导出和评估，避免覆盖旧结果：

```powershell
python scripts/export_pm_dirf_predictions.py --stage stage1 --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --output_dir outputs/icdm2026/predictions/PM_STAGE1_3SLICE --device cuda
python scripts/export_pm_dirf_predictions.py --stage stage2 --stage1_ckpt outputs/pmrf_t1fa_stage1_3slice_pm/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_wm_residual_3slice/checkpoints/best_stage2.pt --output_dir outputs/icdm2026/predictions/PM_DIRF_WM_3SLICE --device cuda --auto_condition_from_ckpt
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/PM_STAGE1_3SLICE --method PM_STAGE1_3SLICE --visualize_count 8
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/PM_DIRF_WM_3SLICE --method PM_DIRF_WM_3SLICE --visualize_count 8
```

判断标准：只有当 3-slice 版本能提升 WM-masked MAE、gradient error、sharpness ratio 和肉眼可见的白质/病灶可读性，同时 PSNR/SSIM 不明显崩掉时，才把它作为论文主实验路线。
