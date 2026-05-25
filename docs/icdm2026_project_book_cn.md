# ICDM 2026 T1-to-FA 项目书

## 1. 项目定位

**暂定题目：** Mining Missing Diffusion Biomarkers from Structural MRI: Posterior-Mean Rectified Flow for T1-to-FA Synthesis and Alzheimer's Staging

**目标会议：** IEEE ICDM 2026 Applied Data Science Track。

**内部截止：** 官网 Applied Track 截止为 2026-06-06 AoE，对应北京时间 2026-06-07 19:59:59。为了留出检查与上传余量，内部目标定为 2026-06-05 完成全文、图表和可复现实验检查。

**目标链接：**

- ICDM 2026 主页：https://icdm2026.neu.edu.cn/main.htm
- ICDM 2026 Applied Track CFP：https://icdm2026.neu.edu.cn/CallforAppliedTrackPapers/list.htm

## 2. 一句话目标

构建一个确定性的 T1-to-FA 合成与数据挖掘框架，从常规 T1 MRI 中恢复配准 FA 图，并验证合成 FA 是否保留阿尔兹海默症谱系分期信号。

## 3. 论文主线调整

当前论文更像医学图像合成工作。为了贴合 ICDM Applied Track，主线应调整为：

> 面向阿尔兹海默症分期的缺失扩散生物标志物挖掘。

这个主线保留已有 T1-to-FA 生成任务，同时增加数据挖掘和应用验证：

- **数据挖掘问题：** 从更容易获得的 T1 MRI 中恢复缺失的扩散微结构标志物。
- **应用领域：** CN、SCD、MCI、AD 四类阿尔兹海默症谱系分期。
- **方法贡献：** posterior-mean guided deterministic rectified flow。
- **应用验证：** 下游分类和脑区案例分析，而不仅是图像指标。

## 4. 数据规范

**数据目录：** `data/`

**原始数据：** `data/ZHU_T1_and_FA_space-MNI152NLin6Asym_res-02`

**处理后切片：** `data/processed`

**受试者切分：** `data/processed/dataset_splits.json`

- Train：173 例，8650 张切片
- Validation：37 例，1850 张切片
- Test：38 例，1900 张切片
- 每例当前贡献 50 张轴向切片

**临床标签文件：** `data/data_information.xlsx`

使用 `re_order` sheet 作为受试者级标签来源。

**确认后的标签映射：**

- `1 = CN`
- `2 = SCD`
- `3 = MCI`
- `4 = AD`

**Excel 标签数量：**

- CN：92
- SCD：56
- MCI：70
- AD：30
- Total：248

**可用协变量：** `groups`、`gender`、`age`、`edu`、`MMSE`。

**硬性规则：** 所有训练、验证、测试都必须以受试者为单位划分，不能做 slice-level 随机切分。

## 5. 当前资产

**第一版主实验：** `train_pure_meanflow_v2_multi_loss.py`

- 思路：确定性 image-to-image rectified flow，学习 T1 到 FA 的 straight bridge。
- 问题：当前论文仍过度强调 FID/KID，不适合作为 FA 定量图的主指标。

**第二版 PMRF 方向：** `pmrf_t1fa/`

- Stage 1：Restormer 风格 posterior-mean/coarse FA 预测。
- Stage 2：从 coarse FA 到真实 FA 的 rectified refinement flow。
- 已有评价：PSNR、SSIM、MSE、MAE、MS-SSIM、脑区/白质 masked MAE、ROI-CCC、梯度误差和白质直方图距离。

**论文草稿：** `papers/IEEE_TMI (7).pdf`、`main.tex`、`_paper_text.txt`。

## 6. 必须解决的问题

### A. 指标不匹配

FID 依赖自然图像 Inception 特征，不应作为 FA 定量图的主指标。

**决定：** 主表改为 PSNR、SSIM、MSE、MAE，并加入白质/脑区相关医学指标。FID/KID 最多作为补充，不作为摘要和结论主论据。

### B. ICDM 贴合度不足

纯合成任务偏医学影像，缺少数据挖掘应用闭环。

**决定：** 增加 AD 谱系下游分类任务，包括四分类和若干二分类。

### C. PSNR/SSIM 差距可能不大

医学配准合成任务中 PSNR/SSIM 差距小是常见情况。

**决定：** 用成像指标、白质 ROI 指标、下游分类、可视化案例、确定性和推理速度共同支撑结论。

### D. 标签尚未接入数据管线

现有 `src/datasets.py` 只返回图像切片和文件名，不返回受试者标签。

**决定：** 增加 subject index，把每张切片映射到 `subject_id`、`slice_id`、`group_id`、`group_name`、`age`、`gender`、`edu`、`MMSE`、`split`。

## 7. 推荐方法

方法名建议：

**PM-DIRF：Posterior-Mean Guided Deterministic Image-to-Image Rectified Flow**

### Stage 1：posterior-mean FA predictor

- 输入：T1 切片。
- 输出：coarse FA。
- Backbone：Restormer 风格编码器-解码器。
- Loss：MSE + L1 + SSIM + gradient + high-frequency。
- 作用：提供低 MSE、稳定的 posterior mean 估计。

### Stage 2：deterministic refinement rectified flow

- Source：Stage 1 coarse FA。
- Target：真实 FA。
- Flow：从 source 到 target 的插值状态。
- Loss：velocity MSE + endpoint MSE/L1/SSIM + gradient/HF/detail。
- 推理：默认 1-step Euler，保持确定性。

## 8. Baseline 与相关工作

必须保留：

- CNN U-Net
- Pix2Pix
- CycleGAN
- DDIM
- Diffusion Bridge Model
- Restormer direct image-to-image
- 原始 DIRF

时间允许可加入：

- NAFNet
- Swin/UMamba

必须引用但不一定复现：

- PMRF
- Diffusion Bridge Models for 3D Medical Image Translation
- Flow Matching for Medical Image Synthesis
- DIReCT
- ViCTr
- 2024-2026 T1-to-FA 相关论文

## 9. 评价设计

### 主合成指标

- PSNR
- SSIM
- MSE
- MAE

### 医学保真指标

- Brain-masked MAE
- White-matter masked MAE
- Gradient error
- ROI-CCC
- White-matter histogram Wasserstein distance

### 效率与确定性

- 单切片推理延迟
- NFE
- 重复推理一致性

### 下游分类

只做 subject-level 分类，比较：

- T1 only
- Real FA only
- Synthetic FA only
- T1 + synthetic FA
- T1 + real FA

指标：

- Balanced accuracy
- Macro-F1
- Binary AUROC
- Confusion matrix
- Bootstrap confidence interval

## 10. 消融实验

| ID | Variant | 回答的问题 |
| --- | --- | --- |
| A0 | 原始 DIRF | 原方法能达到什么水平 |
| A1 | Stage 1 only | posterior-mean 本身有多强 |
| A2 | Stage 1 + Stage 2 | PMRF-style refinement 是否有效 |
| A3 | 去掉 gradient/HF/detail | 白质细节损失是否有效 |
| A4 | 去掉 coarse conditioning | 条件输入是否稳定 refinement |
| A5 | 1/2/4/8 steps | 一步推理是否足够 |
| A6 | 辅助分类 loss | disease-aware supervision 是否提升下游效用 |

## 11. 脑区与案例分析

重点分析 AD 相关白质区域：

- Cingulum bundle
- Corpus callosum
- Fornix
- Uncinate fasciculus
- Superior longitudinal fasciculus
- Periventricular white matter
- Hippocampal/parahippocampal adjacent white matter

案例图展示：

- T1 输入
- 真实 FA
- 合成 FA
- 绝对误差图
- ROI 放大
- 分类 saliency 或 Grad-CAM

## 12. 论文结构

推荐结构：

1. Abstract：强调缺失 FA biomarker mining 和 AD staging。
2. Introduction：从应用数据挖掘问题进入。
3. Related Work：T1-to-FA、medical flow/diffusion、PMRF、下游效用验证。
4. Method：PM-DIRF、loss、确定性推理、分类器。
5. Experiments：数据、baseline、主指标、ROI、分类、消融、案例。
6. Discussion：解释指标选择、局限和未来工作。

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| AD 只有 30 例 | 四分类不稳定 | 用 balanced accuracy、macro-F1、二分类和 bootstrap CI |
| slice-level 泄漏 | 下游结果无效 | 坚持 subject-level split 与聚合 |
| PSNR/SSIM 提升小 | 主表不够强 | 增加 ROI、分类、确定性与速度 |
| Stage 2 不如 Stage 1 | PMRF 贡献弱 | 保留 Stage 1 强 baseline，调 paired score |
| 无外部验证 | 泛化受限 | 明确限制，强化内部严谨验证 |

## 14. 近期动作

1. 建立 subject label index。
2. 更新 dataset loader 返回元数据。
3. 统一图像指标评估脚本。
4. 重评估已有 checkpoint。
5. 训练/评估 PM-DIRF Stage 2。
6. 增加 subject-level 下游分类。
7. 生成表格和图。
8. 重写 `main.tex`。

