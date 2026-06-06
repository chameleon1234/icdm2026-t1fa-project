# U-Net 结果审计

## 结论

当前 `UNet_E99` 结果不能作为 ICDM 当前划分下的公平对比结果。

原因是 `outputs/unet_training/checkpoints/unet_epoch_0099.pth` 是用旧版 `train_unet.py` 的 subject split 训练出来的，不是现在 `data/processed/dataset_splits.json` 里的 train/val/test 划分。当前 ICDM test set 的 38 个受试者里，有 34 个出现在旧 U-Net 的训练受试者列表中。

## 证据

| 当前 test 内部子集 | 受试者数 | 切片数 | PSNR | SSIM | MAE | WM-MAE | SharpRatio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 U-Net 训练时见过 | 34 | 1700 | 38.03 | 0.9786 | 0.00558 | 0.01732 | 0.6826 |
| 旧 U-Net 训练时没见过 | 4 | 200 | 28.10 | 0.9055 | 0.01767 | 0.05906 | 0.5924 |

旧的 U-Net 自带验证文件也显示正常水平：PSNR 27.8819、SSIM 0.9035、MAE 0.0177。这和未见过受试者子集的结果一致，也说明全 test 的 36.99 PSNR 是被泄漏抬高的。

## 解释

`UNet_E99` 在完整 current test 上的 PSNR 36.99、SSIM 0.9709 不是正常泛化能力，而是 subject-level 数据泄漏导致的虚高结果。它不能作为公平 SOTA baseline。

## 后续动作

需要用当前 `data/processed/train` 重新训练 U-Net，用 `data/processed/val` 验证，再导出测试集 PNG，并重新跑图像指标和下游分类。在此之前，当前 U-Net 只能作为“泄漏检查案例”，不能写进公平主表当强 baseline。
