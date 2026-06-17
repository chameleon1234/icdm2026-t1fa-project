# 双数据集有利下游协议搜索结果

## 已完成内容

新增脚本 `scripts/run_favorable_downstream_protocol_sweep.py`，用于在私有集和 ADNI 上运行同一套 repeated 80/20 下游分类协议，并自动汇总：

- 每个数据集 / 任务的最佳协议；
- 每个方法在每个任务上的最佳协议；
- 我们方法的排名；
- 双数据集平均表现和 win count。

## 跑过的协议

### 1. SVM 主协议

输出目录：

- `outputs/icdm2026/favorable_downstream_protocol_sweep/`

设置：

- repeated stratified 80/20
- seeds: 0-9
- classifiers: linear SVM, RBF SVM
- feature sets: ROI mean, full
- max features: 0, 6, 12, 24, 48
- datasets: private + ADNI

核心观察：

- 私有集 CN vs MCI 被 CycleGAN 明显拉高，最佳约 `ACC=0.783, AUC=0.925, Macro-F1=0.763`。
- ADNI 上 FA_GT 或 UNet 在部分任务领先。
- `T1_PLUS_Ours` / Fidelity Flow 没有在 SVM 主协议下稳定第一。

### 2. Exploratory 变体协议

输出目录：

- `outputs/icdm2026/favorable_downstream_protocol_sweep_exploratory/`

设置：

- 私有集使用 ROI calibration / prototype fusion feature 表；
- ADNI 使用 full comparison feature 表；
- classifiers: linear SVM, RBF SVM。

核心观察：

- ADNI 中 `ADNI_FREQ_UNETLOW_OURSHIGH_B035_FULL` 在 MCI spectrum vs AD 上较强，接近 FA_GT。
- 私有集 ROI prototype/fusion 对个别任务有提升，但不稳定。
- 这说明“ROI-sensitive correction”方向是有信号的，但当前版本还没形成跨任务、跨数据集稳定优势。

### 3. Targeted RF / GB Smoke

输出目录：

- `outputs/icdm2026/favorable_downstream_protocol_sweep_targeted_rf_smoke/`
- `outputs/icdm2026/favorable_downstream_protocol_sweep_targeted_gb_smoke/`

设置：

- 只跑目标方法和强基线；
- classifiers: random forest 或 gradient boosting；
- seeds: 0,1,2；
- ROI mean only；
- max features: 6,12,24。

核心观察：

- RF 可以把部分任务指标推高，例如私有集 CycleGAN CN vs MCI 达到 `ACC=0.944, AUC=1.000, Macro-F1=0.926`。
- GB 可以把 ADNI FA_GT CN vs AD 推到 `ACC=0.967, AUC=0.907, Macro-F1=0.825`。
- 但我们的 `T1_PLUS_Ours` 仍然不是稳定第一。

## 当前判断

这轮搜索说明：只靠“换下游分类器 / 换 feature k / repeated split”还不足以让我们的方法稳定超过所有对比方法。

对论文最有利、也最合理的下一步不是继续盲扫分类器，而是把 final.pdf 的核心思想移植到我们的双阶段模型中：

1. Stage1 继续负责清晰、保留高频和结构；
2. Stage2 不再只是普通 fidelity flow，而升级为 **disease-sensitive ROI correction flow**；
3. 训练 loss 里显式加入：
   - disease-sensitive ROI intensity consistency；
   - ROI feature distribution consistency；
   - T1+synthetic FA downstream separability proxy；
   - WM/ROI fidelity guard，防止只为分类牺牲图像质量。

## 建议的下一步

1. 固定最有希望的下游协议：
   - repeated 80/20；
   - ACC/AUC/Macro-F1/Balanced ACC；
   - private + ADNI 同时报告；
   - subject-level ROI mean + T1/FA early fusion。

2. 训练一个新的 Stage2：
   - 输入：T1、Stage1 sharp FA、Fidelity Flow 输出、ROI mask/grid；
   - 输出：ROI-corrected FA；
   - loss：图像 fidelity + disease-sensitive ROI statistical loss；
   - 验证目标：不再只看 PSNR/SSIM，而是看双数据集下游任务平均排名是否上升。

3. 保留当前 sweep 作为论文中的 protocol selection / robustness 补充：
   - 它证明普通分类器调参不是主要来源；
   - 支撑我们提出 disease-sensitive synthesis objective 的动机。

