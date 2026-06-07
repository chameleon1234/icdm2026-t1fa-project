# ADNI 训练命令

这些命令假设 ADNI 已经预处理到以下目录：

- `data/adni_processed/train/t1_slices`
- `data/adni_processed/train/fa_slices`
- `data/adni_processed/val/t1_slices`
- `data/adni_processed/val/fa_slices`
- `data/adni_processed/test/t1_slices`
- `data/adni_processed/test/fa_slices`

训练、导出、评估都使用 GPU 环境：

```powershell
conda activate dinov3test
```

## Baseline 顺序

公开数据集 baseline 按这个顺序跑：

1. U-Net：最快的 sanity baseline。
2. Pix2Pix：paired GAN baseline。
3. CycleGAN：unpaired/translation GAN baseline。
4. Stage1 sharp model 和 frequency/flow fusion 变体。

ADNI 第一阶段目标不是立刻追最终最优 checkpoint，而是确认公开数据集能在同一 split、同一测试指标下放大方法差异。

## U-Net ADNI

```powershell
python -m scripts.train_unet_current_split `
  --run_name unet_adni_split_e50 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --epochs 50 `
  --batch_size 32 `
  --lr 1e-4 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 10
```

导出和评估：

```powershell
python scripts/export_legacy_baseline_predictions.py `
  --method unet `
  --checkpoint outputs/unet_adni_split_e50/checkpoints/best_unet.pt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir outputs/icdm2026/predictions/ADNI_UNET_E50 `
  --device cuda `
  --batch_size 32

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/ADNI_UNET_E50 `
  --method ADNI_UNET_E50 `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --visualize_count 8
```

## Pix2Pix ADNI

```powershell
python -m scripts.train_pix2pix_current_split `
  --run_name pix2pix_adni_split_e50 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --epochs 50 `
  --batch_size 8 `
  --lr 2e-4 `
  --lambda_l1 100 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 10
```

导出评估时使用 `--method pix2pix`，checkpoint 是 `outputs/pix2pix_adni_split_e50/checkpoints/best_pix2pix.pt`。

## CycleGAN ADNI

```powershell
python -m scripts.train_cyclegan_current_split `
  --run_name cyclegan_adni_split_e50 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --epochs 50 `
  --batch_size 4 `
  --lr 2e-4 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 10
```

导出评估时使用 `--method cyclegan`，checkpoint 是 `outputs/cyclegan_adni_split_e50/checkpoints/best_cyclegan.pt`。

## ADNI 下游任务

ADNI 下游只使用 `data/adni_processed/adni_slice_manifest.csv` 中有标签的 test subject。公开数据集主任务：

- CN vs MCI_spectrum+AD
- CN vs MCI_spectrum
- CN vs AD
- MCI_spectrum vs AD 作为辅助任务，因为 AD 总数只有 25 个 subject

示例命令：

```powershell
python scripts/evaluate_downstream_classification.py `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --split test `
  --include_t1 `
  --include_fa_gt `
  --method ADNI_UNET_E50=outputs/icdm2026/predictions/ADNI_UNET_E50 `
  --tasks cn_vs_mci_spectrum_ad,cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad,adni_three_class `
  --n_splits 5 `
  --repeat_seeds 1,2,3,4,5 `
  --output_root outputs/icdm2026/downstream_adni_public
```
