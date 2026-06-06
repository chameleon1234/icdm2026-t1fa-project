# 对比方法实验设计

## 目标

所有对比方法必须先转换成同一测试集 PNG 格式，再用同一套脚本评估：

1. 导出预测到 `outputs/icdm2026/predictions/<METHOD>`。
2. 使用 `scripts/evaluate_method_folder.py` 计算 PSNR、SSIM、MSE、MAE、清晰度、WM-MAE、ROI-CCC，并生成固定切片可视化。
3. 使用 `scripts/evaluate_downstream_classification.py` 做 repeated subject-level 下游疾病分类。

这样可以避免把自然图像 FID 类评价和医学配对图像评价混在一起。

## 已实现对比方法

| 类别 | 方法 | checkpoint | 作用 |
|---|---|---|---|
| CNN | U-Net | `outputs/unet_training/checkpoints/unet_epoch_0099.pth` | 确定性监督回归 baseline。 |
| GAN | Pix2Pix | `checkpoints_pix2pix/epoch_100.pt` | paired adversarial image-to-image translation。 |
| GAN | CycleGAN | `checkpoints_cyclegan/epoch_100.pt` | unpaired adversarial translation baseline。 |
| Diffusion | DDIM | `checkpoints_baseline_ddim_steps1/epoch_100.pt` | 条件扩散迭代采样，使用 50 DDIM steps 导出。 |
| Flow | DIRF V5 | 已有预测文件夹 | 早期 flow baseline。 |
| PMRF | PM Stage1 / PM-DIRF 系列 | 已有预测文件夹 | 对应 `papers/PMRF.pdf` 的 posterior-mean + rectified-flow 思路。 |
| Ours | Frequency-preserving fidelity 系列 | 已有预测文件夹 | 当前平衡清晰度、医学保真和下游任务的主线方法。 |

## 当前执行命令模板

```powershell
conda activate dinov3test

python scripts/export_legacy_baseline_predictions.py --model_type cyclegan --ckpt checkpoints_cyclegan/epoch_100.pt --output_dir outputs/icdm2026/predictions/CycleGAN_E100 --method CycleGAN_E100 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type pix2pix --ckpt checkpoints_pix2pix/epoch_100.pt --output_dir outputs/icdm2026/predictions/Pix2Pix_E100 --method Pix2Pix_E100 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type unet --ckpt outputs/unet_training/checkpoints/unet_epoch_0099.pth --output_dir outputs/icdm2026/predictions/UNet_E99 --method UNet_E99 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type ddim --ckpt checkpoints_baseline_ddim_steps1/epoch_100.pt --output_dir outputs/icdm2026/predictions/DDIM_E100_K50 --method DDIM_E100_K50 --device cuda --batch_size 8 --sample_steps 50
```

导出后，统一运行图像指标和下游分类脚本。

## 第二批对比方法

`outputs/ldm_bridge_training_vgg15/checkpoints` 下存在 LDM/Diffusion Bridge checkpoint，但当前仓库没有稳定的导出脚本。它应该作为第二批 baseline 单独实现 VAE+LDM sampler，验证能导出完整 1900 张测试 PNG 后，再进入论文表格。不要把未经统一导出的结果混入当前主表。
