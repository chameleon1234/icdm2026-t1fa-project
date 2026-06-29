# 训练数据使用审计

日期：2026-06-29

## 结论

本轮 `outputs/icdm2026/final_validation` 做的是**全量测试集统一评估**，不是把所有候选方法重新全量训练一遍。

需要区分三件事：

- **全量训练**：训练命令或 checkpoint args 中 `train_limit=0`、`val_limit=0`，实际使用完整 train/val split。
- **限量训练**：训练命令或 export summary 中存在 `train_limit=4096`、`val_limit=1024` 等限制。
- **全量评估/导出**：预测目录覆盖完整 test split，ADNI 为 5616 张 test slices，私有集为 1900 张 test slices。

## 数据集规模

| dataset | split | slices |
| --- | --- | ---: |
| ADNI | train | 19552 |
| ADNI | val | 2808 |
| ADNI | test | 5616 |
| Private | train | 8650 |
| Private | val | 1850 |
| Private | test | 1900 |

## ADNI 训练状态

| method | training status | evidence |
| --- | --- | --- |
| `ADNI_PM_DIRF_FIDELITY_FLOW_FULL` | 全量训练 | Stage1 checkpoint args: `train_limit=0`, `val_limit=0`, `epochs=80`; Stage2 checkpoint args: `train_limit=0`, `val_limit=0`, `epochs=40`; log: `19552 train / 2808 val slices` |
| `ADNI_PM_STAGE1_LPIPS_GAN_FULL` | 全量训练 | Stage1 checkpoint args: `train_limit=0`, `val_limit=0`, `context_slices=5`, `epochs=80` |
| `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6` | 限量训练 | export summary / checkpoint args: `train_limit=4096`, `val_limit=1024`, `epochs=6` |
| `ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080` | 派生融合结果 | 由已有预测做频域融合，不是独立全量训练模型 |
| `ADNI_PRIOR_FLOW_SAFE_PROBE_FULL` | probe / 非最终全量训练候选 | 预测已全量导出，但该路线用于消融，不建议写成最终主训练 |
| `ADNI_UNet_E50` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `19552 train / 2808 val slices` |
| `ADNI_StackUNet7_E50` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `19552 train / 2808 val slices` |
| `ADNI_Pix2Pix_E50` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `19552 train / 2808 val slices` |
| `ADNI_CycleGAN_E50` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `19552 train / 2808 val slices` |
| `ADNI_DDIM_E100_K50_PRETRAINED` | 预训练/已有结果全量评估 | 当前审计确认 test 预测全量；训练日志未在本次核查中证明为全量 |
| `ADNI_DBM_E100_K40_PRETRAINED` | 预训练/已有结果全量评估 | 当前审计确认 test 预测全量；训练日志未在本次核查中证明为全量 |
| `ADNI_MOTFM_I2I_K10_PRETRAINED` | 预训练/已有结果全量评估 | 当前审计确认 test 预测全量；训练日志未在本次核查中证明为全量 |

## 私有集训练状态

| method | training status | evidence |
| --- | --- | --- |
| `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` | 全量训练 | export summary / checkpoint args: Stage2 `train_limit=0`, `val_limit=0`, `epochs=30`; uses `data/processed/train` and `data/processed/val` |
| `PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030` | 全量训练 | checkpoint args: Stage1 `train_limit=0`, `val_limit=0`, `context_slices=1`, `epochs=60` |
| `PM_DIRF_FIDELITY_FLOW_FULL` | 全量训练 | Stage1 checkpoint args: `train_limit=0`, `val_limit=0`, `epochs=80`; uses private train/val splits |
| `UNet_CurrentSplit_E100` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `8650 train / 1850 val slices` |
| `StackUNet5_CurrentSplit_E50` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `8650 train / 1850 val slices` |
| `Pix2Pix_CurrentSplit_E100` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `8650 train / 1850 val slices` |
| `CycleGAN_CurrentSplit_E100` | 全量训练 | train summary: `train_limit=0`, `val_limit=0`, `8650 train / 1850 val slices` |

## 最终评估是否全量

最终统一评估使用了完整 test split：

| dataset | method group | predicted png count |
| --- | --- | ---: |
| ADNI | final candidates and baselines | 5616 |
| Private | final candidates and baselines | 1900 |

也就是说，**最终图像指标是全量测试集指标**。

## 需要修正的论文表述

不能笼统写“所有方法均全量训练”。更准确的写法是：

> All final image evaluations were conducted on the full held-out test split. For training, the main Fidelity Flow and conventional baselines were trained on the full training split, whereas the A080 disease-sensitive corrector was trained with a 4096-slice training subset and evaluated on the full test split.

中文：

> 所有最终图像指标均在完整独立测试集上评估。训练阶段中，主 Fidelity Flow 与常规对比方法使用完整训练集；A080 disease-sensitive corrector 分支使用 4096 张训练切片和 1024 张验证切片训练，并在完整测试集上评估。

## 建议

如果要把 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6` 写成最终 ADNI 主方法，最好补一版全量训练：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --coarse_pred_dir outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN4096 `
  --val_coarse_pred_dir outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_VAL1024 `
  --run_name adni_ds_corrector_from_a080_disease_roi_hfpreserve_full_e12 `
  --variant multihead `
  --epochs 12 `
  --batch_size 2 `
  --num_workers 0 `
  --lr 6e-5 `
  --width 48 `
  --num_blocks 8 `
  --mixed_precision bf16 `
  --disease_roi_csv outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv `
  --hf_preserve_weight 4.0 `
  --sharp_retention_weight 3.0
```

但注意：这条命令是否真正全量，还取决于 `coarse_pred_dir` 是否覆盖完整 train split。目前可见的 coarse train 目录名是 `TRAIN4096`，所以如果要严格全量训练 A080+DS，还需要先补齐 A080 base 的完整 train split 预测。
