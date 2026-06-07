# ADNI Training Commands

These commands assume ADNI preprocessing has produced:

- `data/adni_processed/train/t1_slices`
- `data/adni_processed/train/fa_slices`
- `data/adni_processed/val/t1_slices`
- `data/adni_processed/val/fa_slices`
- `data/adni_processed/test/t1_slices`
- `data/adni_processed/test/fa_slices`

Use the GPU environment for all training/export/evaluation commands:

```powershell
conda activate dinov3test
```

## Baseline Order

Run public-dataset baselines in this order:

1. U-Net: fast sanity baseline.
2. Pix2Pix: paired GAN baseline.
3. CycleGAN: unpaired/translation GAN baseline.
4. Stage1 sharp model and frequency/flow fusion variants.

The first ADNI goal is not to chase the final best checkpoint immediately. The first goal is to confirm that the public dataset enlarges method differences under the same split and same test metrics.

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

Export and evaluate:

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

Export/evaluate with `--method pix2pix` and checkpoint `outputs/pix2pix_adni_split_e50/checkpoints/best_pix2pix.pt`.

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

Export/evaluate with `--method cyclegan` and checkpoint `outputs/cyclegan_adni_split_e50/checkpoints/best_cyclegan.pt`.

## Downstream ADNI

Use only labeled ADNI test subjects from `data/adni_processed/adni_slice_manifest.csv`. Main public downstream tasks:

- CN vs MCI_spectrum+AD
- CN vs MCI_spectrum
- CN vs AD
- MCI_spectrum vs AD as auxiliary because AD has only 25 total subjects

Example:

```powershell
python scripts/evaluate_downstream_classification.py `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --split test `
  --method T1_ONLY=data/adni_processed/test/t1_slices `
  --method FA_GT=data/adni_processed/test/fa_slices `
  --method ADNI_UNET_E50=outputs/icdm2026/predictions/ADNI_UNET_E50 `
  --tasks cn_vs_mci_spectrum_ad,cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad,adni_three_class `
  --n_splits 5 `
  --repeat_seeds 1,2,3,4,5 `
  --output_root outputs/icdm2026/downstream_adni_public
```
