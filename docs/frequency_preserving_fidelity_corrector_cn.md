# 高频保留医学定量校正器

## 目的

项目不再要求 Stage2 从已经丢失高频信息的 posterior-mean Stage1 输出中重新生成 FA 纹理。

新的职责分工为：

```text
T1 5-slice
  -> 清晰的感知-对抗 Stage1
  -> 高频保留医学定量校正器
  -> 最终 FA 与下游效用分析
```

Stage1 被冻结，负责提供肉眼可见的解剖结构和纹理。Stage2 只允许修改指定低频带，用于校正 FA 强度、WM 误差和区域一致性，不能重新生成或抹平 Stage1 的高频纹理。

## 模型结构

校正器输入条件为：

```text
[T1 5-slice, Stage1 FA, lowpass(Stage1 FA), highpass(Stage1 FA)]
```

两种校正器共享相同的 NAF-style backbone 和频率合成方式：

- `flow`：在低频校正空间学习条件残差 Flow 轨迹，是论文主实验候选。
- `direct`：直接预测低频校正量，是判断 Flow 是否真正有贡献的必要消融。

最终输出为：

```text
final = Stage1 FA + FFT_lowpass(raw_correction)
```

因此禁止修改的高频带直接继承自 Stage1。日志中的 `HFLeak` 仅测量 Stage2 写入禁止高频带的能量。

## 损失与选优

训练损失包括 Flow velocity、低频校正 L1、最终图像 L1/MSE/SSIM、target-derived proxy WM L1、网格区域一致性、校正幅度、背景校正和高频保持。

训练中的 ROI loss 是网格区域代理，不是真实解剖 ROI 标注。论文最终区域指标仍然是测试集 `ROI-CCC`，不能把训练代理写成解剖 ROI 真值。

只有同时通过以下门槛，checkpoint 才会保存为 `best_fidelity_corrector.pt`：

- 清晰度保留率不低于 `0.97`。
- WM L1 不退化。
- ROI 代理误差不退化。
- PSNR 下降不超过 `0.03 dB`。
- SSIM 下降不超过 `0.002`。

## 已验证的小规模结果

256 张切片、3 epoch probe：

| 模型 | 清晰度保留率 | 最佳 Delta WM L1 | 最佳 Delta ROI | 最佳 Delta PSNR | 禁止高频泄漏 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flow | 1.0004 | +0.001674 | +0.004872 | -0.0177 | 0.00000002 |
| Direct | 0.9998 | +0.000786 | +0.002036 | -0.0005 | 0.00000166 |

该结果验证了结构假设：Stage2 能在不破坏 Stage1 高频的前提下改善医学区域误差。但在完成全量训练和测试集 ROI-CCC 评估前，不能宣称最终模型已经优于 Stage1。

## 正式训练

首先进入 GPU 环境：

```powershell
conda activate dinov3test
```

Flow 主实验：

```powershell
python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --run_name pmrf_t1fa_stage2_fidelity_flow_full `
  --corrector_mode flow `
  --epochs 40 `
  --batch_size 2 `
  --width 48 `
  --num_blocks 8 `
  --eval_steps 4 `
  --mixed_precision bf16
```

Direct corrector 消融：

```powershell
python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --run_name pmrf_t1fa_stage2_fidelity_direct_full `
  --corrector_mode direct `
  --epochs 40 `
  --batch_size 2 `
  --width 48 `
  --num_blocks 8 `
  --mixed_precision bf16
```

断点续训时重复原命令，并增加：

```powershell
--resume outputs/<run_name>/checkpoints/latest_fidelity_corrector.pt
```

断点续训采用严格校验。架构、频率、损失、mask 或选优参数缺失或变化时，会拒绝续训，避免实验轨迹混淆。

## 导出与评估

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/pmrf_t1fa_stage2_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL `
  --method PM_DIRF_FIDELITY_FLOW_FULL `
  --visualize_count 8
```

如果正式训练没有生成 best checkpoint，说明该实验没有通过医学定量门槛，不能作为论文主方法。
