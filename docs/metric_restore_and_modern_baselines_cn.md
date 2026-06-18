# Metric-Restore Fidelity Flow 与近年对比实验计划

## 目标

下一步不是继续盲目追求更锐，而是在 Sharpness 不明显降低的前提下，把 PSNR 和 SSIM 拉上去。当前最合理的假设是：

- Stage 1 LPIPS+GAN 负责生成清晰的 FA 高频结构。
- Stage 2 Fidelity Flow 只负责低频/中低频医学指标校正。
- checkpoint 选择必须拒绝明显损失清晰度的结果。

这个逻辑已经落到：

```text
pmrf_t1fa/train_pmrf_t1fa_stage2_fidelity_corrector.py
```

新增的 `metric_restore` preset 会提高配对保真损失，同时保留清晰度门槛：

- `frequency_cutoff >= 0.16`
- `final_l1_weight >= 0.80`
- `final_mse_weight >= 0.80`
- `final_ssim_weight >= 0.50`
- `wm_l1_weight >= 1.00`
- `roi_weight >= 0.30`
- `hf_preserve_weight >= 2.50`
- `best_min_sharp_retention >= 0.97`
- `best_min_delta_psnr >= 0`
- `best_min_delta_ssim >= 0`

小规模验证结论：从零训练不稳定，但从已有 full Fidelity Flow checkpoint 微调，可以在 Sharpness 保持 0.97 以上的同时提升 PSNR 和 SSIM。因此正式训练应该使用 `--init_ckpt`。

## 私有数据集全量训练

```powershell
conda activate dinov3test

python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --training_preset metric_restore `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --init_ckpt outputs/pmrf_t1fa_stage2_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt `
  --run_name pm_dirf_fidelity_flow_v2_metric_restore_full `
  --corrector_mode flow `
  --epochs 20 `
  --batch_size 2 `
  --lr 3e-5 `
  --width 48 `
  --num_blocks 8 `
  --preview_every 512 `
  --mixed_precision bf16
```

导出和评估：

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/pm_dirf_fidelity_flow_v2_metric_restore_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --method PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --visualize_count 8
```

## ADNI 全量训练

```powershell
conda activate dinov3test

python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --training_preset metric_restore `
  --stage1_ckpt outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt `
  --init_ckpt outputs/adni_pmrf_stage2_fidelity_flow_full_e40/checkpoints/best_fidelity_corrector.pt `
  --run_name adni_pmrf_stage2_fidelity_flow_v2_metric_restore_full `
  --corrector_mode flow `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --epochs 20 `
  --batch_size 2 `
  --lr 3e-5 `
  --width 48 `
  --num_blocks 8 `
  --preview_every 512 `
  --mixed_precision bf16
```

导出和评估：

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/adni_pmrf_stage2_fidelity_flow_v2_metric_restore_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --method ADNI_PM_DIRF_FIDELITY_FLOW_V2_METRIC_RESTORE_FULL `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count 8
```

## 近年对比实验分组

所有方法必须经过同一套导出、图像指标、下游任务评估流程。

### 强监督图像到图像 baseline

- U-Net / Stack U-Net
- Pix2Pix
- CycleGAN
- Restormer
- NAFNet-style MeanFlow
- Swin-style MeanFlow
- U-Mamba-style MeanFlow

这组用于回答：我们的双阶段方法是否能在 PSNR、SSIM、MSE、MAE、Sharpness、WM-MAE、ROI-CCC、ACC/AUC 上超过强配对生成 baseline。

### 近年重生成 baseline

ADNI 的重方法 runner 已经覆盖：

- DDIM
- DBM / Diffusion Bridge
- MOTFM image-to-image flow

```powershell
conda activate dinov3test

python scripts/run_heavy_baseline_suite.py `
  --device cuda `
  --ddim_steps 50 `
  --dbm_steps 40 `
  --motfm_steps 10 `
  --visualize_count 8
```

## 汇报原则

不要再只靠某一个筛选指标说“最好”。最终表格必须同时报告：

- 图像保真：PSNR、SSIM、MSE、MAE
- 细节清晰度：Sharpness ratio 和固定切片可视化
- 医学一致性：WM-MAE、ROI-CCC
- 下游价值：ACC、AUC、Macro-F1
- 采样成本：DDIM、DBM、MOTFM 这类多步方法必须报告运行时间或步数

这轮目标不是单纯最高 Sharpness，而是在当前锐图基础上提升 PSNR/SSIM，同时 Sharpness 保持在约 97% 以上，并且 WM/ROI 不退化。
