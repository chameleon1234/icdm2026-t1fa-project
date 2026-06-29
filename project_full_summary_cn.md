# T1-to-FA 生成论文项目完整汇总记录

更新时间：2026-06-29  
项目目录：`E:\桌面\T1_to_FA_Generation`  
历史聊天记录文件：`chat_record_cn.md`  
本文件用途：整合当前论文项目的目标、技术路线、关键对话决策、代码改动、实验结果、失败方案、保留方案、当前问题和下一步计划。

---

## 1. 当前项目目标

本项目目标是从 T1 MRI 生成 FA 图像，并证明生成结果不仅在图像指标上合理，而且具有医学/下游任务价值。

最初目标会议为 ICDM 2026 Applied Track，因此早期强调数据挖掘下游任务。后续与老师讨论后，目标扩展为医学影像/医学期刊也可以投稿，因此当前核心目标改为：

1. 生成清晰、准确、医学一致的 T1-to-FA 图像。
2. 在双数据集上验证：私有阿尔兹海默症数据集 + ADNI 公开数据集。
3. 指标同时覆盖传统图像质量、白质/病灶相关医学指标、清晰度指标和下游任务。
4. 论文逻辑必须能解释：为什么需要从 T1 生成 FA，以及生成 FA 如何帮助疾病分析或脑白质结构理解。

数据标签约定已经明确：

```text
Excel 标签以 1=CN, 2=SCD, 3=MCI, 4=AD 为准
```

当前主线任务：

```text
single-slice T1
    -> sharp / texture-aware FA prior
    -> Fidelity Flow or disease-sensitive corrector
    -> clear + ROI-consistent + downstream-useful FA
```

---

## 2. 数据集与目录

### 2.1 私有数据集

原始说明：

- 数据表：`data\data_information.xlsx`
- 248 例阿尔兹海默症相关数据
- 标签：`1=CN, 2=SCD, 3=MCI, 4=AD`
- 预处理代码：`preprocessing\preprocess_new.py`
- 常用切片目录：
  - `data/processed/train/t1_slices`
  - `data/processed/train/fa_slices`
  - `data/processed/val/t1_slices`
  - `data/processed/val/fa_slices`
  - `data/processed/test/t1_slices`
  - `data/processed/test/fa_slices`

### 2.2 ADNI 公开数据集

数据压缩包：

```text
data\ADNI_data.7z
```

预处理后目录：

```text
data/adni_processed/train/t1_slices
data/adni_processed/train/fa_slices
data/adni_processed/val/t1_slices
data/adni_processed/val/fa_slices
data/adni_processed/test/t1_slices
data/adni_processed/test/fa_slices
data/adni_processed/adni_slice_manifest.csv
```

ADNI 当前切片数量：

```text
train: 19552 slices
val:    2808 slices
test:   5616 slices
```

ADNI 评估必须带上：

```powershell
--adni_slice_manifest data\adni_processed\adni_slice_manifest.csv
```

否则 `evaluate_method_folder.py` 会出现：

```text
KeyError: Subject sub-002_S_4213 missing from subject index
```

---

## 3. 项目演化与关键对话决策

### 3.1 最初 PMRF 路线

参考：

```text
papers\PMRF.pdf
```

最初设计：

```text
Stage1: posterior mean predictor
Stage2: PMRF / rectified flow refinement
```

初始想法是学习 PMRF：

- Stage1 预测稳定的 posterior mean FA。
- Stage2 用 flow 注入纹理、高频、白质细节。

后来关键结论：

```text
这条路线作为最终“清晰生成”路线走不通。
```

原因：

1. Stage1 posterior mean 本质上倾向条件均值。
2. FA 细纹理并不是单张 T1 确定的普通纹理，Stage1 平滑后很多高频已被抹掉。
3. Stage2 从平滑 coarse 出发，只能做小修补，很难无中生有恢复真实细纹理。
4. 多次实验表明 Stage2 的 K10/K25 对视觉改善有限。

代表结果：

| 方法 | PSNR | SSIM | Sharp | WM-MAE | ROI |
|---|---:|---:|---:|---:|---:|
| PM_STAGE1_WMROI_DETAIL_5SLICE | 27.7409 | 0.8978 | 0.6810 | 0.0597 | 0.8548 |
| PM_DIRF_DETAIL_TEACHER_V5_K10 | 27.7573 | 0.8977 | 0.7590 | 0.0594 | 0.8603 |
| PM_DIRF_DETAIL_TEACHER_V5_K25 | 27.7599 | 0.8978 | 0.7605 | 0.0593 | 0.8610 |

结论：

```text
Stage2 能略微提高指标或锐度，但不能把平滑 posterior mean 变成真实清晰 FA。
```

### 3.2 2.5D / 5-slice 方向

尝试过：

```text
Stage1 输入从单张 T1 改为 3/5 张邻近切片，输出中心 FA
```

动机：

- FA 与邻近解剖上下文有关。
- 参考 Kwon 2026、Macro2Micro、Qiu 2026 等论文思路。

实际问题：

1. 私有数据量有限，5-slice 容易减少有效训练样本。
2. 2.5D 提升并不足以解决纹理真实性。
3. 老师后续明确指出“不用 5-slice，数据量不够”。

当前决策：

```text
论文主线改回 single-slice 训练。
```

### 3.3 LPIPS + GAN 路线

为解决模糊问题，尝试在 Stage1 使用：

```text
L1 + MSE + SSIM + WM + Grad + HF + LPIPS + GAN
```

实验结论：

- LPIPS-only 不提升锐度，甚至可能降低 Sharp。
- GAN-only 提升有限。
- LPIPS+GAN 组合能显著提升清晰度，但会引入局部白质过亮、亮条、亮点和假纹理。

私有数据消融：

| Method | PSNR | SSIM | Sharp | WM-MAE | ROI |
|---|---:|---:|---:|---:|---:|
| Baseline | 27.74 | 0.898 | 0.681 | 0.060 | 0.855 |
| LPIPS-only | 27.77 | 0.900 | 0.657 | 0.056 | 0.870 |
| GAN-only | 27.81 | 0.901 | 0.689 | 0.055 | 0.868 |
| LPIPS+GAN | 27.56 | 0.891 | 0.882 | 0.062 | 0.835 |
| + Fidelity Flow | 27.58 | 0.893 | 0.869 | 0.059 | 0.867 |

当前判断：

```text
LPIPS+GAN 是重要消融，证明清晰度能被拉起来，但不能直接作为最终可靠 Stage1，因为它会产生白质局部过亮伪纹理。
```

### 3.4 Fidelity Flow 路线

当前保留的核心结构：

```text
Stage1: sharp FA prior
Stage2: Fidelity Flow / disease-sensitive medical correction
```

它的论文逻辑：

1. Stage1 负责生成清晰 FA prior，保留结构与高频。
2. Stage2 不再承担“无中生有补纹理”，而是做医学一致性校正。
3. Stage2 重点监督：
   - WM-MAE
   - ROI consistency / ROI_CCC
   - disease-sensitive ROI
   - 亮度/局部过亮控制
   - 高频保留

老师反馈后的调整：

- Stage2 如果只小幅改变，审稿人会问“不知道它在干什么”。
- 因此 Stage2 需要有明确指标表达：ROI、WM、病灶区域、一致性、下游任务提升。

当前保留理由：

```text
Fidelity Flow 两阶段逻辑清楚，视觉效果较好，是目前最容易写成论文主线的结构。
```

### 3.5 Frequency Fusion / A080 路线

曾经尝试：

```text
low-frequency source: LowGuard / PM / FlowBase
high-frequency source: sharp teacher / Fidelity / LightGuard
```

代表候选：

```text
ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080
```

结果：

| Method | PSNR | SSIM | WM-MAE | ROI_CCC | Sharp |
|---|---:|---:|---:|---:|---:|
| A080 | 28.3714 | 0.9086 | 0.0613 | 0.7677 | 0.9961 |
| A080 + DS HFKeep | 28.6789 | 0.9128 | 0.0565 | 0.8285 | 1.0228 |

决策变化：

- 曾一度认为 Frequency Fusion 没必要，应回到 Fidelity Flow。
- 后续发现 A080 在 ADNI attention-MIL 下表现强，可作为候选/诊断分支。
- 但如果论文主线叫 Fidelity Flow，则 Frequency Fusion 不宜作为最终核心叙事，除非重新包装为“texture base / sharp prior construction”。

当前定位：

```text
A080 是强实验候选，不是最干净的最终主线；可作为 texture candidate 或 ablation。
```

### 3.6 Prior Flow / Hybrid Source 路线

为了避免 LPIPS+GAN 伪纹理，同时保留 texture，尝试：

```text
source = FA low-frequency prior + T1 high-frequency / template / teacher prior
flow: source -> FA
```

最近代码：

```text
pmrf_t1fa/train_pmrf_t1fa_stage1_prior_flow.py
scripts/export_prior_flow_stage1_predictions.py
```

关键新增：

- `TemplatePriorFlowNet`
- `build_prior_flow_state`
- `euler_sample_prior_flow`
- `--flow_source`
- `--flow_steps`
- `--pin_memory`
- `scripts/check_memory_safety.py`

最近安全 probe：

训练命令：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m pmrf_t1fa.train_pmrf_t1fa_stage1_prior_flow `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --run_name adni_stage1_prior_flow_safe_probe `
  --context_slices 1 `
  --flow_source template `
  --flow_steps 8 `
  --epochs 3 `
  --batch_size 1 `
  --num_workers 0 `
  --train_limit 1024 `
  --val_limit 512 `
  --lr 6e-5 `
  --width 48 `
  --num_blocks 8 `
  --mixed_precision bf16
```

完整 ADNI test 导出：

```text
outputs/icdm2026/predictions/ADNI_PRIOR_FLOW_SAFE_PROBE_FULL
```

完整 ADNI test 可视化：

```text
outputs/icdm2026/figures/method_slices/ADNI_PRIOR_FLOW_SAFE_PROBE_FULL
```

结果：

| Method | PSNR | SSIM | WM-MAE | ROI_CCC | Sharp |
|---|---:|---:|---:|---:|---:|
| ADNI_PRIOR_FLOW_SAFE_PROBE_FULL | 28.0684 | 0.8940 | 0.0570 | 0.8434 | 0.9798 |

判断：

```text
Prior Flow safe probe 是当前“稳态纹理候选”，Sharp 很理想，WM-MAE 好，ROI 也合理，但 SSIM 偏低，ROI 还低于 Old Fidelity Flow Full。
```

---

## 4. 当前代码状态

### 4.1 主要训练脚本

| 文件 | 用途 |
|---|---|
| `pmrf_t1fa/train_pmrf_t1fa_stage1.py` | 原始 Stage1 / sharp adversarial / detail Stage1 训练入口 |
| `pmrf_t1fa/train_pmrf_t1fa_stage2.py` | PMRF / DIRF 风格 Stage2 flow 训练入口 |
| `pmrf_t1fa/train_pmrf_t1fa_stage2_ds_corrector.py` | disease-sensitive / metric corrector |
| `pmrf_t1fa/train_pmrf_t1fa_stage2_fidelity_corrector.py` | Fidelity corrector |
| `pmrf_t1fa/train_pmrf_t1fa_stage2_hp_refiner.py` | high-pass residual refiner |
| `pmrf_t1fa/train_pmrf_t1fa_stage2_residual_flow.py` | residual flow corrector |
| `pmrf_t1fa/train_pmrf_t1fa_stage1_prior_flow.py` | 当前 Prior Flow / template-source / hybrid-source Stage1 |

### 4.2 主要模型文件

| 文件 | 用途 |
|---|---|
| `pmrf_t1fa/models/pmrf_t1fa.py` | Stage1Net、FlowUNet、TemplatePriorFlowNet 等核心模型 |
| `pmrf_t1fa/models/__init__.py` | 模型导出 |
| `pmrf_t1fa/__init__.py` | 包级导出 |

### 4.3 导出与评估脚本

| 文件 | 用途 |
|---|---|
| `scripts/export_pm_dirf_predictions.py` | Stage1/Stage2 PM-DIRF 导出 |
| `scripts/export_prior_flow_stage1_predictions.py` | Prior Flow Stage1 导出 |
| `scripts/export_ds_corrector_predictions.py` | DS corrector 导出 |
| `scripts/export_fidelity_corrector_predictions.py` | Fidelity corrector 导出 |
| `scripts/export_hp_refiner_predictions.py` | HP refiner 导出 |
| `scripts/evaluate_method_folder.py` | 统一评估 PSNR/SSIM/MSE/MAE/WM/ROI/Sharpness 等 |
| `scripts/evaluate_clarity_blur_metrics.py` | 清晰度/模糊诊断 |
| `scripts/build_adni_method_slice_panels.py` | ADNI 方法横向切片对比图 |
| `scripts/evaluate_slice_mil_roi.py` | slice-level / subject-level MIL 下游任务 |
| `scripts/check_memory_safety.py` | 新增，训练前 RAM/GPU 安全检查 |

### 4.4 最近 git 提交

```text
61ddef9 Add memory-safe prior flow training defaults
b4fbaf1 Add ADNI support to slice MIL evaluation
537f152 Add prior-flow hybrid T1 high-pass source
5837158 Add high-pass detail blending utility
4145ba5 Add FA intensity calibration utility
9f4e464 Add ADNI final metric comparison table
304c7aa Add ADNI discriminative metric strategy
e336a3a Add masked PSNR audit for ADNI comparisons
fa5f9f6 Add fair Restormer ADNI comparison
d013df3 Add final ADNI multihead panel entry
9e97a5e Add ADNI single-slice sharp stage2 panel preset
cb03e8a Add multihead disease-sensitive corrector
```

### 4.5 当前内存问题修复

用户遇到：

```text
Windows 任务管理器显示：
RAM 30.8/31.1 GB, 99%
磁盘 100%
GPU 0%
```

诊断：

```text
不是 NVIDIA 显存炸，而是系统 RAM / pagefile 被打满。
```

已修改：

1. `train_pmrf_t1fa_stage1_prior_flow.py`
   - 默认关闭 `pin_memory`
   - 新增 `--pin_memory`
   - 日志打印 `num_workers` 和 `pin_memory`
2. 新增 `scripts/check_memory_safety.py`
   - 检查 RAM、Python 进程、NVIDIA 显存
   - RAM 不足时提前报警

训练前建议：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe scripts\check_memory_safety.py --min_free_ram_gb 6
```

安全训练参数：

```text
batch_size=1
num_workers=0
pin_memory=False
mixed_precision=bf16
先 train_limit=1024/4096，再全量
```

---

## 5. 当前主要实验结果

### 5.1 ADNI 图像指标核心表

| Method | PSNR | SSIM | MSE | MAE | WM-MAE | ROI_CCC | Sharp | GradErr | WM-SkelErr | SliceConsErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADNI_PM_DIRF_FIDELITY_FLOW_FULL | 28.0915 | 0.9053 | 0.0017 | 0.0177 | 0.0570 | 0.8670 | 0.8892 | 0.1232 |  |  |
| ADNI_PM_DIRF_FIDELITY_FLOW_4096_E12 | 27.4185 | 0.8919 | 0.0019 | 0.0192 | 0.0637 | 0.8185 | 0.6931 | 0.1323 |  |  |
| ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080 | 28.3714 | 0.9086 | 0.0016 | 0.0174 | 0.0613 | 0.7677 | 0.9961 | 0.1174 | 0.0827 | 0.0329 |
| ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6 | 28.6789 | 0.9128 | 0.0015 | 0.0168 | 0.0565 | 0.8285 | 1.0228 | 0.1149 | 0.0759 | 0.0323 |
| ADNI_PRIOR_FLOW_SAFE_PROBE_FULL | 28.0684 | 0.8940 | 0.0017 | 0.0184 | 0.0570 | 0.8434 | 0.9798 | 0.1186 | 0.0784 | 0.0298 |
| ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_4096_E8_BEST_K8 | 28.0664 | 0.9022 | 0.0017 | 0.0182 | 0.0582 | 0.8194 | 1.1291 | 0.1209 | 0.0808 | 0.0302 |
| ADNI_STAGE1_TEMPLATE_WMMSGAN_4096_E8_FULL | 28.0801 | 0.9066 | 0.0017 | 0.0180 | 0.0633 | 0.7833 | 0.7110 | 0.1178 | 0.0854 | 0.0302 |
| ADNI_STAGE1_MBNAF_TEMPLATE_WMGAN_1024_E5 | 28.1523 | 0.9070 | 0.0017 | 0.0179 | 0.0640 | 0.7467 | 0.6897 | 0.1161 | 0.0891 | 0.0291 |
| ADNI_UNet_E50 | 28.4444 | 0.9073 | 0.0015 | 0.0170 | 0.0584 | 0.8122 | 0.5186 | 0.1183 |  |  |
| ADNI_Pix2Pix_E50 | 28.0113 | 0.9014 | 0.0017 | 0.0177 | 0.0606 | 0.8270 | 0.9126 | 0.1272 |  |  |
| ADNI_CycleGAN_E50 | 26.2564 | 0.8705 | 0.0025 | 0.0220 | 0.0784 | 0.6907 | 0.8769 | 0.1497 |  |  |
| ADNI_DDIM_E100_K50_PRETRAINED | 25.7831 | 0.8617 | 0.0028 | 0.0236 | 0.0880 | 0.5803 | 0.5694 | 0.1548 |  |  |
| ADNI_DBM_E100_K40_PRETRAINED | 25.8801 | 0.8765 | 0.0028 | 0.0232 | 0.0928 | 0.5057 | 0.4889 | 0.1408 |  |  |
| ADNI_MOTFM_I2I_K10_PRETRAINED | 17.2323 | 0.2378 | 0.0204 | 0.0876 | 0.1803 | 0.1392 | 2.1839 | 0.2681 |  |  |

说明：

- `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6` 图像指标很强，但论文叙事要谨慎，不能让 Frequency Fusion 变成解释不清的最终方法。
- `ADNI_PM_DIRF_FIDELITY_FLOW_FULL` ROI_CCC 最强，视觉上较好，是当前最容易讲清楚的两阶段主线。
- `ADNI_PRIOR_FLOW_SAFE_PROBE_FULL` 是最新安全 prior-flow 候选，Sharp/WM/ROI 较平衡，但 SSIM 低于 Fidelity Flow。

### 5.2 私有数据集图像指标核心表

| Method | PSNR | SSIM | MSE | MAE | WM-MAE | ROI_CCC | Sharp | GradErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_FULL | 27.4912 | 0.8923 | 0.0019 | 0.0190 | 0.0621 | 0.8348 | 0.8205 | 0.1241 |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 27.2797 | 0.8896 | 0.0020 | 0.0194 | 0.0625 | 0.8607 | 0.9374 | 0.1286 |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL | 28.0136 | 0.9023 | 0.0017 | 0.0176 | 0.0525 | 0.8916 | 1.0099 | 0.1182 |
| PRIVATE_DS_ATLAS_SINGLE_2048_E5 | 27.0712 | 0.8857 | 0.0021 | 0.0205 | 0.0607 | 0.8282 | 0.6548 | 0.1264 |
| PRIVATE_DS_HYBRID_SINGLE_2048_E5 | 27.0890 | 0.8851 | 0.0021 | 0.0204 | 0.0607 | 0.8297 | 0.7222 | 0.1267 |
| PM_STAGE1_WMROI_DETAIL_5SLICE | 27.7409 | 0.8978 | 0.0018 | 0.0184 | 0.0597 | 0.8548 | 0.6810 | 0.1191 |
| PM_DIRF_FIDELITY_FLOW_FULL | 27.5812 | 0.8925 | 0.0018 | 0.0189 | 0.0592 | 0.8665 | 0.8692 | 0.1237 |
| PM_DIRF_FIDELITY_DIRECT_FULL | 27.5874 | 0.8923 | 0.0018 | 0.0189 | 0.0593 | 0.8630 | 0.8742 | 0.1239 |
| DIRF_V5_3SLICE_K6 | 28.3995 | 0.9094 | 0.0015 | 0.0172 | 0.0545 | 0.8839 | 0.4551 | 0.1102 |
| UNet_CurrentSplit_E100 | 28.2361 | 0.8973 | 0.0016 | 0.0178 | 0.0575 | 0.8361 | 0.4561 | 0.1126 |
| Pix2Pix_CurrentSplit_E100 | 28.1136 | 0.9035 | 0.0016 | 0.0178 | 0.0578 | 0.8715 | 0.6415 | 0.1139 |
| CycleGAN_CurrentSplit_E100 | 26.1662 | 0.8711 | 0.0026 | 0.0225 | 0.0681 | 0.8115 | 0.8643 | 0.1394 |
| DDIM_E100_K50 | 27.4790 | 0.8876 | 0.0019 | 0.0192 | 0.0629 | 0.8272 | 0.6669 | 0.1280 |

说明：

- 私有集上 `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 指标非常强，尤其 WM-MAE、ROI_CCC 和 SharpRatio。
- `DIRF_V5_3SLICE_K6` / U-Net PSNR 高，但 Sharp 很低，肉眼模糊问题明显。
- Pix2Pix 在部分 ROI 指标上接近甚至强，需要论文里解释差异点：清晰度、WM/ROI 平衡、下游 utility、伪影控制。

---

## 6. 下游任务结果

### 6.1 ADNI attention-MIL 汇总

来源：

```text
outputs/icdm2026/reports/downstream_dual_dataset_attention_mil_cn.md
outputs/icdm2026/downstream_dualdataset_mil_summary.csv
```

| Method | Accuracy | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|
| ADNI_A080_DS_Fair | 0.799 | 0.690 | 0.523 |
| ADNI_A080_DS_HFKeep | 0.799 | 0.686 | 0.526 |
| ADNI_A080 | 0.794 | 0.678 | 0.521 |
| ADNI_DDIM | 0.776 | 0.537 | 0.498 |
| ADNI_Old_Fidelity | 0.771 | 0.620 | 0.576 |
| ADNI_MOTFM | 0.752 | 0.613 | 0.551 |
| ADNI_DBM | 0.730 | 0.603 | 0.539 |
| T1_ONLY | 0.717 | 0.715 | 0.602 |
| FA_GT | 0.700 | 0.589 | 0.536 |

注意：

- ADNI 下游结果中，A080+DS 在 Accuracy 上很强。
- 但 AUC/F1 并不总是领先 T1_ONLY。
- 这说明不能单纯宣称“全面超越”，需要选择合理叙事：生成 FA 提供互补疾病表征，在部分协议和任务上提升。

### 6.2 私有数据 attention-MIL 汇总

| Method | Accuracy | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|
| T1_PLUS_GT | 0.755 | 0.637 | 0.658 |
| T1_PLUS_Ours | 0.734 | 0.676 | 0.689 |
| FidelityFlow | 0.698 | 0.644 | 0.630 |
| UNet | 0.696 | 0.731 | 0.646 |
| FA_GT | 0.685 | 0.661 | 0.591 |
| CycleGAN | 0.613 | 0.609 | 0.544 |
| Pix2Pix | 0.587 | 0.727 | 0.547 |
| T1_ONLY | 0.555 | 0.709 | 0.542 |

关键任务：

| Method | Task | Accuracy | Macro-AUC | Macro-F1 |
|---|---|---:|---:|---:|
| FidelityFlow | CN vs AD | 0.857 | 0.838 | 0.788 |
| T1_PLUS_Ours | CN vs AD | 0.857 | 0.824 | 0.788 |
| FA_GT | CN vs AD | 0.857 | 0.765 | 0.743 |
| UNet | CN vs AD | 0.810 | 0.912 | 0.767 |
| T1_PLUS_Ours | CN vs MCI | 0.679 | 0.706 | 0.675 |
| T1_ONLY | MCI vs AD | 0.867 | 0.750 | 0.830 |

解释：

- 私有数据上，`FidelityFlow` 和 `T1_PLUS_Ours` 在 CN vs AD 上很强。
- CN vs MCI / MCI vs AD 不一定全面领先，需要承认任务差异。
- 下游任务可作为 utility evidence，但不能替代图像指标。

### 6.3 为什么 final.pdf 的下游指标可能很高

对话中总结过，`papers\相同任务\final.pdf` 下游指标高，可能来自以下设计差异：

1. 任务设置可能更容易，例如 CN vs AD，而不是多类或 MCI 细分。
2. 可能使用 slice-level 大样本训练，而不是严格 subject-level 5-fold。
3. 可能采用固定 train/test split，且样本分布更有利。
4. 可能报告 ACC/AUC，而不是 Macro-F1。
5. 可能使用 T1+FA 融合、atlas ROI、AAL3 ROI 或手工特征，而不是只用生成 FA。
6. 如果切片级划分没有严格 subject-level 隔离，指标会虚高。

当前决策：

```text
可以参考 final.pdf 的展示方式，但不能做数据泄漏。可以报告 ACC/AUC，同时保留 subject-level 或 MIL 协议说明。
```

---

## 7. 方案状态：保留与否定

### 7.1 已否定或不作为主线

| 方案 | 状态 | 原因 |
|---|---|---|
| PMRF posterior mean Stage1 + conservative Stage2 | 否定为最终清晰生成主线 | Stage1 已抹掉高频，Stage2 难以无中生有恢复 |
| 继续调 Stage2 多步 K10/K25 | 基本否定 | K10/K25 与 K1 差异小，视觉提升有限 |
| VGG19/LPIPS 强行加到 Stage2 | 否定 | 对 FA 定量图不稳定，不能解决 coarse 过平滑 |
| HP refiner 单独补高频 | 否定为主线 | DeltaSharp 可转正，但 WM/视觉/稳定性不够 |
| CleanBase + 小 refiner | 暂不作为最终 | 高亮柔和，但纹理仍不够 |
| 纯 T1-start flow | 否定为直接最终 | 纹理强但低频/FA 强度空间不对 |
| 纯 GAN/LPIPS Stage1 | 否定为最终 | 清晰但局部白质过亮和假纹理 |
| 5-slice / 2.5D 作为主线 | 暂停 | 数据量不够，老师建议 single-slice |
| Frequency Fusion 作为最终独立主方法 | 不推荐 | 逻辑容易显得后处理/拼接，除非包装成 texture prior 构造 |

### 7.2 当前保留方案

| 方案 | 状态 | 作用 |
|---|---|---|
| Old Fidelity Flow Full | 强主线候选 | 结构清楚，ROI 强，视觉较好 |
| A080 + DS HFKeep | 强实验候选 | ADNI 图像指标和下游 Accuracy 强，但叙事需谨慎 |
| Prior Flow Safe Probe | 新候选 | Sharp/WM/ROI 平衡，适合继续放大 |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL | 私有集强结果 | 私有集图像指标强，可作为私有数据主结果候选 |
| Pix2Pix / UNet / CycleGAN / DDIM / DBM / MOTFM | 对比实验 | 用于展示方法差异、模糊/ROI/下游表现 |

---

## 8. 当前问题

### 8.1 核心科学问题

当前模型不是不会生成 FA，而是卡在：

```text
真实细纹理生成
vs
伪白质纹理 / 亮度失控
```

已观察到：

1. LPIPS+GAN 能清楚，但高亮/假纹理明显。
2. LowGuard/CleanBase 能压高亮，但纹理变平。
3. PriorFlow / Hybrid source 能恢复一定 texture，但仍需验证能否比 Old Fidelity Flow 更稳定。
4. T1 high-pass 不能直接当 FA high-pass；T1 高频只是结构线索，不是 FA 微纹理真值。

### 8.2 当前工程问题

1. 输出结果非常多，需要整理最终候选，避免论文路线混乱。
2. ADNI 评估必须确保完整导出，不能用半导出结果。
3. 32GB RAM 机器训练/导出容易内存压力大，必须使用安全参数。
4. PowerShell 每次提示：

```text
File D:\文档\WindowsPowerShell\profile.ps1 cannot be loaded because running scripts is disabled on this system
```

这个不影响 Python 命令执行，但会污染输出。

### 8.3 论文风险

1. Pix2Pix / UNet 在部分指标上接近甚至更强，不能硬说全面超越。
2. 需要选择能够体现方法优势的指标：
   - ROI_CCC
   - WM-MAE
   - Sharpness_Ratio
   - WM_Skeleton_Error
   - Slice_Consistency_Error
   - downstream ACC/AUC/F1
3. 需要解释为什么生成 FA 有价值：
   - FA 是 DTI 扩散信息，不是 T1 能直接替代。
   - 生成 FA 可作为缺失模态补全、疾病分析辅助、白质结构表征。
   - 下游任务不是唯一目的，而是 utility evidence。

---

## 9. 当前最该做什么

### P0：先固定最终候选集合

建议将最终候选限制为 3 个：

1. `ADNI_PM_DIRF_FIDELITY_FLOW_FULL`
2. `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6`
3. `ADNI_PRIOR_FLOW_SAFE_PROBE_FULL` 或其放大版 `adni_stage1_prior_flow_safe_4096_e8`

私有集候选：

1. `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL`
2. `PM_DIRF_FIDELITY_FLOW_FULL`
3. `PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030`

### P1：跑 PriorFlow 放大版

先检查内存：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe scripts\check_memory_safety.py --min_free_ram_gb 6
```

建议训练：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m pmrf_t1fa.train_pmrf_t1fa_stage1_prior_flow `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --run_name adni_stage1_prior_flow_safe_4096_e8 `
  --context_slices 1 `
  --flow_source template `
  --flow_steps 8 `
  --epochs 8 `
  --batch_size 1 `
  --num_workers 0 `
  --train_limit 4096 `
  --val_limit 1024 `
  --lr 6e-5 `
  --width 48 `
  --num_blocks 8 `
  --mixed_precision bf16
```

目标：

```text
PSNR >= 28.1
SSIM >= 0.90
WM-MAE <= 0.057
ROI_CCC >= 0.85
SharpRatio 0.90 - 1.05
无明显局部白质过亮
```

### P2：若 PriorFlow 放大版视觉仍不够

不要继续盲目调 loss。应转向：

```text
FA texture prior / teacher high-frequency distillation
```

候选：

1. 用视觉最好的 meanflow K10/Old Fidelity 作为 high-frequency teacher。
2. 只蒸馏高频结构，同时用 GT/WM/ROI 约束位置。
3. 避免 GAN 直接生成假纹理。

### P3：双数据集最终表格

最终论文表格应至少包括：

1. 图像指标：PSNR / SSIM / MSE / MAE
2. 医学指标：WM-MAE / ROI_CCC / ROI_Spearman
3. 清晰度：Sharpness_Ratio / Tenengrad / EdgeGrad / BlurDeficit
4. 伪影：NonWMOver / LocalSpike / AnatomyUnsupportedSpike
5. 下游任务：ACC / AUC / Macro-F1
6. 视觉图：固定 ADNI 与私有集切片横向对比

---

## 10. 关键可视化路径

### ADNI

```text
outputs/icdm2026/figures/method_slices/ADNI_PM_DIRF_FIDELITY_FLOW_FULL
outputs/icdm2026/figures/method_slices/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6
outputs/icdm2026/figures/method_slices/ADNI_PRIOR_FLOW_SAFE_PROBE_FULL
outputs/icdm2026/figures/method_slices/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080
outputs/icdm2026/figures/adni_method_slice_panels
```

### 私有集

```text
outputs/icdm2026/figures/method_slices/PM_DIRF_FIDELITY_FLOW_FULL
outputs/icdm2026/figures/method_slices/PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL
outputs/icdm2026/figures/method_slices/PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030
outputs/icdm2026/figures/method_slices/PM_STAGE1_WMROI_DETAIL_5SLICE
```

### 伪影诊断

```text
outputs/icdm2026/figures/artifact_diagnosis_fidelity_flow_gate_mild
outputs/icdm2026/figures/artifact_diagnosis_fidelity_flow_gate_mild_no_boxes
outputs/icdm2026/figures/artifact_diagnosis_lowguard_detailhp_sweep_no_boxes
outputs/icdm2026/figures/artifact_diagnosis_priorflow_lightguard_best_vs_current_no_boxes
```

---

## 11. 报错与排查记录

### 11.1 RAM 爆掉但不是显存 OOM

现象：

```text
RAM 99%
Disk 100%
GPU 0%
```

结论：

```text
系统内存/分页文件被打满，不是 NVIDIA 显存炸。
```

处理：

```text
batch_size=1
num_workers=0
pin_memory=False
使用 scripts/check_memory_safety.py
```

### 11.2 ADNI 评估缺 manifest

报错：

```text
KeyError: Subject sub-002_S_4213 missing from subject index
```

解决：

```powershell
--adni_slice_manifest data\adni_processed\adni_slice_manifest.csv
```

### 11.3 导出超时导致半成品评估

现象：

```text
ADNI_PRIOR_FLOW_SAFE_PROBE 只导出 2452/5616 张
missing_prediction_count=3164
```

解决：

重新导出到：

```text
outputs/icdm2026/predictions/ADNI_PRIOR_FLOW_SAFE_PROBE_FULL
```

结果：

```text
exported=5616
n_slices=5616
n_subjects=108
```

---

## 12. 论文创新点候选总结

当前最推荐的论文创新表达：

### 12.1 清晰 FA prior 与医学一致性校正解耦

不是让一个模型同时承担所有目标，而是：

```text
Stage1: clear FA prior
Stage2: disease-sensitive fidelity correction
```

### 12.2 针对 T1-to-FA 的指标体系

从自然图像 FID 改为：

```text
PSNR / SSIM / MSE / MAE
WM-MAE
ROI_CCC / ROI_Spearman
Sharpness_Ratio / Tenengrad / EdgeGrad
Slice Consistency
WM Skeleton Error
Downstream ACC/AUC/Macro-F1
```

### 12.3 负结果也能变成论文动机

PMRF posterior mean 阶段证明：

```text
低误差 posterior mean 不等于视觉清晰，也不一定保留疾病相关细节。
```

这可以作为为什么需要 sharp prior / fidelity correction 的动机。

### 12.4 双数据集验证

私有数据集 + ADNI：

- 私有集证明临床疾病任务价值。
- ADNI 证明公开数据可复现性。

---

## 13. 一句话当前结论

当前项目已经不再是“能不能 T1-to-FA”的阶段，而是在选择最终论文主线：

```text
Fidelity Flow 是最容易讲清楚的两阶段主线；
A080+DS 是 ADNI 指标强候选；
PriorFlow 是最新安全纹理候选；
下一步应固定候选、做双数据集统一评估和视觉筛选，而不是继续无限开新分支。
```

