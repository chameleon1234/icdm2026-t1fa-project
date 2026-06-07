# ADNI 双数据集实验设计

## 目标

私有 248 例数据集用于方法开发和初步验证，ADNI 公开数据集用于最终可复现对比、消融和下游数据挖掘验证。

## 数据体检结果

- 压缩包：`data/ADNI_data.7z`
- 内部目录：
  - `ADNI_data/T1/*.nii.gz`
  - `ADNI_data/FA/*.nii.gz`
- 影像空间：`MNI152NLin6Asym`，`res-02`
- 抽样体积 shape：`91 x 109 x 91`
- 分辨率：`2 mm`
- 文件数：
  - T1: 542
  - FA: 538
  - T1-FA 配对 subject: 538
  - 只有 T1、没有 FA 的 subject: 4
- 标签文件：`data/subject_group_cleaned_filtered.csv`
- 有标签 subject: 405
- 配对且有标签的组别：
  - CN: 214
  - MCI: 113
  - AD: 25
  - EMCI: 24
  - LMCI: 13
  - `_S_MC`: 16
- 配对但无标签 subject: 133

## 标签策略

生成任务使用所有配对 T1-FA subject，包括没有诊断标签的 ADNI subject。

下游分类任务：

- `CN` 保留为正常对照。
- `MCI`、`EMCI`、`LMCI` 合并为 `MCI_spectrum`。
- `AD` 保留。
- `_S_MC` 暂时从下游任务剔除，除非后续确认其含义。
- 无标签 subject 不进入下游分类，但保留在生成任务 train/val/test 中。

## 划分策略

所有划分都必须按 subject 进行，绝不能按 slice 随机。

第一版 split：

- train: 70%
- val: 10%
- test: 20%
- 分层：
  - 有标签 subject 按归一化后的诊断组分层
  - 无标签 paired subject 单独作为 `UNLABELED` 分层

这样可以最大化生成模型的 paired 训练样本，同时保留公开数据集的有标签测试集用于 disease utility。

## 评估策略

ADNI 公开数据集用于放大和稳定模型差异，避免只在私有 38 个测试 subject 上挤很小的 PSNR/F1 差距。

图像保真：

- PSNR
- SSIM
- MSE
- MAE

细节和解剖：

- SharpRatio
- Gradient error
- WM-MAE
- ROI-MAE / ROI-CCC

下游 utility：

- CN vs AD
- CN vs MCI_spectrum
- CN vs MCI_spectrum+AD
- MCI_spectrum vs AD
- CN / MCI_spectrum / AD 三分类作为辅助任务

增益实验：

- T1 only
- synthetic FA only
- T1 + synthetic FA
- real FA only
- T1 + real FA

## 预处理命令

先生成 subject 级 manifest 和 train/val/test 划分：

```powershell
D:\Anaconda3\python.exe scripts/build_adni_manifest.py `
  --archive data/ADNI_data.7z `
  --labels data/subject_group_cleaned_filtered.csv `
  --output_root outputs/icdm2026 `
  --seed 42 `
  --train_ratio 0.7 `
  --val_ratio 0.1
```

再把配对的 NIfTI 体数据切成和私有数据集一致的 PNG 目录结构：

```powershell
D:\Anaconda3\python.exe scripts/preprocess_adni_slices.py `
  --archive data/ADNI_data.7z `
  --manifest outputs/icdm2026/adni_subject_manifest.csv `
  --output_root data/adni_processed `
  --extract_root data/adni_raw_extracted `
  --splits train,val,test `
  --slice_start 20 `
  --slice_end 72 `
  --target_size 224 `
  --min_brain_fraction 0.01
```

这里预处理建议用 `D:\Anaconda3\python.exe`，因为当前 `dinov3test` 环境有 `nibabel` 但没有 `libarchive`，不能直接读取 `.7z`。后续训练、导出、评估仍然进入 `conda activate dinov3test` 使用 GPU。

## 论文主线

私有数据集作为方法开发集，ADNI 作为公开验证集。一个方法只有同时在以下方面平衡，才适合作为主方法：

- paired image fidelity
- 视觉/白质细节
- ROI 一致性
- 下游疾病 utility

这能避免只基于小私有测试集上的 PSNR/SSIM 小差距过度叙事，也更贴合 ICDM 的数据挖掘主题。
