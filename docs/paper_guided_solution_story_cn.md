# 论文阅读后的路线收敛与故事设计

更新时间：2026-06-29  
项目：T1-to-FA 生成与下游疾病分析  
相关论文目录：`papers/`  
原始抽取记录：`outputs/icdm2026/reports/paper_extraction_notes_selected.txt`

---

## 1. 当前问题的重新定义

结合当前所有实验和新加入论文，项目现在的核心问题不是“模型不会生成 FA”，也不是“清晰度不够”这么简单，而是：

```text
T1 -> FA 生成中，清晰纹理的来源是否可靠；
生成 FA 是否保持白质/ROI 统计一致性；
生成结果是否能支持疾病分析，而不是只在视觉上像 FA。
```

你现在遇到的现象可以被几类论文共同解释：

1. **Perception-Distortion Tradeoff** 说明：PSNR/SSIM 与视觉清晰度天然存在冲突。高 PSNR 往往更平滑，GAN/感知损失更清楚但可能牺牲失真指标。
2. **Adversarial and Perceptual Refinement for CS-MRI** 说明：感知/对抗增强要用任务或语义指标验证，否则只是“看起来更好”。
3. **GANs for Medical Image Synthesis** 说明：GAN 可以骗过视觉评估，但不一定复现医学数据的完整丰富性。
4. **T1-to-FA / DTI 相关论文** 说明：FA 是白质微结构指标，不是普通纹理图，局部高频必须落在合理白质/ROI 区域，否则就是假纹理。

因此，你现在的论文故事不能写成：

```text
我们追求更清晰的 FA 图像。
```

更合理的表述是：

```text
我们研究 T1-to-FA 生成中清晰度、失真指标和医学一致性之间的矛盾，
提出一个以可信纹理来源和区域统计约束为核心的生成框架，
在保持视觉清晰的同时减少白质伪纹理和 ROI 统计漂移。
```

---

## 2. 重点论文给你的直接启发

### 2.1 Diffusion Bridge Models for 3D Medical Image Translation

路径：

```text
papers/相同任务/Zhang 等 - 2025 - Diffusion Bridge Models for 3D Medical Image Translation.pdf
```

关键内容：

- 直接做 T1w MRI 与 DTI/FA 的 3D medical image translation。
- 使用 perceptual similarity、pixel-level agreement、distributional consistency。
- 用 sex classification 和 Alzheimer’s disease classification 验证 synthetic FA 的实用价值。
- 明确 synthetic FA 需要保留 white matter integrity 信息。

对你的启发：

1. 你的评价不能只放 PSNR/SSIM，应加入 distributional consistency、WM/ROI 指标和下游分类。
2. 下游任务不应被当作“额外花活”，而应是 synthetic FA 是否保留微结构疾病信息的核心证据。
3. 你可以把当前 `ROI_CCC / WM-MAE / downstream ACC-AUC-F1` 作为“白质完整性保留”的证据链。

对当前方案的影响：

```text
保留 Fidelity Flow / DS Corrector 的 ROI/WM 校正叙事。
不要只讲图像锐度，必须讲 synthetic FA 的 clinical utility。
```

---

### 2.2 Generating Diffusion MRI Scalar Maps from T1 Weighted Images using GANs

路径：

```text
papers/Generating Diffusion MRI Scalar Maps from T1 Weighted Images using GANs.pdf
```

关键内容：

- 直接使用 CycleGAN 从 T1 生成 FA / MD。
- 说明 diffusion scalar maps 能反映 microstructural tissue properties。
- synthetic FA 可用于 diffusion MRI 几何畸变校正。

对你的启发：

1. T1-to-FA 是已有合理任务，不是凭空设想。
2. CycleGAN 是合理对比实验，但不能作为最终主线，因为它不保证局部白质统计可靠。
3. synthetic FA 的价值可以不仅限于分类，还可以是配准、补全、微结构替代标志物。

对当前方案的影响：

```text
CycleGAN 应作为经典 baseline。
论文引言中可以用它说明 T1->FA/MD 是可行方向，但早期 GAN 方法缺乏医学一致性控制。
```

---

### 2.3 Manifold-Aware CycleGAN for High-Resolution Structural-to-DTI Synthesis

路径：

```text
papers/相同任务/Manifold-Aware CycleGAN for High-Resolution Structural-to-DTI Synthesis.pdf
```

关键内容：

- 指出只生成 FA/MD scalar maps 会忽略 diffusion tensor 的 orientation 信息。
- 使用 SPD(3) manifold 和 Log-Euclidean 约束生成 DTI tensor。
- 评估包括 principal orientation cosine similarity、FA MSE、Log-Euclidean distance。

对你的启发：

1. 你当前生成的是 FA scalar，因此要承认它无法完全恢复 tensor orientation。
2. 为了让 FA scalar 更可信，应强调白质 skeleton、ROI rank consistency、WM/ROI error，而不是只看视觉纹理。
3. 你可以把“纹理必须位于合理白质区域”作为医学一致性约束，避免 GAN 假纹理。

对当前方案的影响：

```text
不要把 Sharpness_Ratio 当最终唯一指标。
必须加入 WM_Skeleton_Error、ROI_CCC、ROI_Spearman、WM-MAE。
```

---

### 2.4 WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis

路径：

```text
papers/方法设计/WFM 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis.pdf
```

关键内容：

- 不从纯噪声开始，而从 informed prior 开始。
- 在 wavelet space 中做 flow matching。
- 利用 MRI 模态之间共享解剖结构，只需 1-2 steps 就能快速合成。

对你的启发：

你现在一直在问：

```text
x0 到底应该是什么？
```

WFM 给出的答案是：

```text
不要从 noise 开始，也不要从错误 source 开始；
要从结构可信的 informed prior 开始。
```

这直接支持你当前的 PriorFlow / A080 思路：

- `template` 太平滑；
- `T1-start` 低频不在 FA 空间；
- `T1 high-pass` 容易带入错误 T1 边缘；
- `FA-space sharp teacher high-frequency + stable low-frequency` 更合理。

对当前方案的影响：

```text
PriorFlow / A080 不应被写成经验后处理，而应被写成 informed-prior flow / frequency-aware prior construction。
```

---

### 2.5 Frequency Domain Decomposition Translation (FDDT)

路径：

```text
papers/方法设计/Frequency Domain Decomposition Translation for Enhanced Medical Image Translation Using GANs.pdf
```

关键内容：

- 明确把图像分为 high-frequency 和 low-frequency。
- 高频承载 details / identity information。
- 低频承载 style information。
- 传统 translation 忽略频域分布会造成 distortion 和低质量。

对你的启发：

你现在 A080 的本质就是：

```text
LowGuard low-frequency
+
LightGuard / FA-space teacher high-frequency
```

这可以被正式写成：

```text
frequency-aware decomposition prior
```

而不是“后处理融合”。

对当前方案的影响：

1. 低频负责 FA 亮度、整体结构、ROI 稳定。
2. 高频负责白质纹理和微结构细节。
3. Stage2 负责防止 ROI / WM 统计漂移。
4. 指标上加入 BandPassCorr、WM_BandPass_MAE、WM_Skeleton_Error 会更有说服力。

---

### 2.6 The Perception-Distortion Tradeoff

路径：

```text
papers/方法设计/The Perception-Distortion Tradeoff.pdf
```

关键内容：

- 低 distortion 和高 perceptual quality 存在理论冲突。
- GAN 可以提升 perceptual quality，但会牺牲 distortion。

对你的启发：

这正好解释你所有实验：

| 现象 | 理论解释 |
|---|---|
| U-Net / posterior mean PSNR 高但糊 | distortion 优化走向均值解 |
| LPIPS+GAN 清晰但假纹理/过亮 | perceptual quality 提升但 distortion/医学一致性漂移 |
| Fidelity Flow ROI 强但不一定最锐 | 医学 fidelity 与视觉锐度折中 |
| A080 sharp/PSNR 强但 ROI 弱 | 高频好但区域统计仍需校正 |

对当前方案的影响：

论文讨论中必须明确：

```text
医学图像生成不能只追求 perception，也不能只追求 distortion；
需要引入 clinical fidelity 作为第三个目标。
```

建议三角叙事：

```text
Distortion: PSNR / SSIM / MAE
Perception: Sharpness / Tenengrad / EdgeGrad / BlurDeficit
Clinical Fidelity: WM-MAE / ROI_CCC / downstream utility
```

---

### 2.7 Adversarial and Perceptual Refinement for CS-MRI

路径：

```text
papers/方法设计/Adversarial and Perceptual Refinement for Compressed Sensing MRI Reconstruction.pdf
```

关键内容：

- MSE 网络 PSNR 高但模糊。
- adversarial/perceptual loss 更好看。
- 但必须用 semantic interpretability score 评估增强是否对分析有用。

对你的启发：

这支持你保留 LPIPS+GAN 作为消融，但不支持把它作为最终主线。

论文可写：

```text
感知增强能恢复纹理，但若缺乏白质/ROI 约束，会产生局部过亮和统计漂移；
因此我们引入 disease-sensitive fidelity correction。
```

---

### 2.8 GANs for Medical Image Synthesis: An Empirical Study

路径：

```text
papers/方法设计/GANs for Medical Image Synthesis An Empirical Study.pdf
```

关键内容：

- GAN 能生成视觉逼真的医学图像。
- 但 segmentation 等下游结果显示，GAN 不一定复现医学数据全部丰富性。

对你的启发：

这为你解释 LPIPS+GAN 失败提供了理论和经验支持：

```text
视觉真实不等于医学真实。
```

因此你的最终方法要避免单纯 GAN 幻觉纹理。

---

### 2.9 SynDiff / McCaD

路径：

```text
papers/方法设计/Unsupervised Medical Image Translation with Adversarial Diffusion Models SynDiff.pdf
papers/方法设计/McCaDMultiContrast MRI Conditioned, Adaptive Adversarial Diffusion Model for High-Fidelity MRI Synthesis.pdf
```

关键内容：

- SynDiff：用 adversarial diffusion 代替 one-shot GAN，缓解 GAN fidelity 问题。
- McCaD：强调 feature-level mapping、多尺度 feature-guided denoising、spatial feature-attentive loss。

对你的启发：

1. 不要用单步 GAN 硬造纹理。
2. flow/diffusion 思路合理，但 source / condition 设计很关键。
3. T1 高频不能直接当 FA 高频；需要 feature-level 或 FA-space teacher prior。

对当前方案的影响：

```text
如果 PriorFlow/A080 还不够，下一步不是换大 backbone，而是引入 FA texture teacher / FA reference prior。
```

---

### 2.10 final.pdf / StructDINO

路径：

```text
papers/相同任务/final.pdf
```

关键内容：

这篇和你的任务最接近。它明确指出 T1-to-FA 的三个问题：

1. semantic domain gap：自然图像 foundation model 直接用于脑 MRI 有域差异；
2. spatial mapping ambiguity：T1 到 FA 的区域映射不同，缺少解剖上下文会导致模糊；
3. statistical unreliability：只优化视觉 plausibility 会造成区域 FA 统计漂移，使其不能作为临床 biomarker。

它的方法核心：

```text
DINOv3 prior
anatomy-adapted attention
structural-adapted normalization
dual-space statistical consistency
atlas-derived anatomical guidance
```

它的评价：

```text
PSNR / FID / MAE / SSIM
下游分类 ACC / AUC
ROI mean intensities early concatenation + SVM
```

它报告的结果：

```text
StructDINO:
PSNR 37.70
FID 10.39
MAE 1.84%
SSIM 92.68%

Downstream:
AD/CN: 96.0 / 98.3
MCI/CN: 81.8 / 77.1
AD/MCI: 80.0 / 84.5
```

为什么它的下游指标高：

1. 使用了 ROI mean intensities 早融合 SVM，而不是严格小样本 subject-level 复杂 MIL。
2. 任务为二分类，尤其 AD/CN 较容易。
3. 用 atlas-derived anatomical guidance 强化 ROI 区域统计。
4. 它将生成 FA 作为 structural regularizer，抑制 diagnostic confounders。
5. 它直接优化 regional statistical consistency。

对你的启发：

你的故事应该靠近它，但不能硬模仿 DINOv3：

```text
我们的核心不是 foundation model，而是可信纹理来源 + 频域分解 + disease-sensitive ROI consistency。
```

你可以借鉴的设计：

1. 解释区域映射歧义：T1 上相似强度区域可对应不同 FA 微结构。
2. 强调统计漂移：LPIPS+GAN 视觉清晰但 ROI/WM 可能漂移。
3. 引入 dual-space consistency：
   - image-space fidelity：PSNR/SSIM/MAE
   - region-space fidelity：ROI_CCC/ROI_Spearman/WM-MAE
4. 下游任务采用 ROI mean intensity + SVM/linear classifier，报告 ACC/AUC。

---

### 2.11 Kwon 2026 / Macro2Micro / Qiu 2026

路径：

```text
papers/相同任务/Kwon 等 - 2026 - Generative Synthesis of Fractional Anisotropy Maps from T1 MRI Using Transfer Learning for White Mat.pdf
papers/相同任务/Macro2Micro.pdf
papers/相同任务/Qiu - 2026 - Multi-modal MRI-Based Alzheimer's Disease Diagnosis with Transformer-based Image Synthesis and Trans.pdf
```

共同启发：

1. T1-to-FA 是合理的临床任务，因为 DTI/FA 获取成本高。
2. 2.5D / 3D context 有助于宏观到微观映射，但你当前数据量下主线仍应 single-slice。
3. 生成 FA 的最终价值应通过疾病分类/白质完整性/ROI 统计验证。
4. Macro2Micro 的思路说明 T1 宏观结构可以预测一定微结构，但必须处理 macro-to-micro ambiguity。

对当前方案的影响：

```text
single-slice 是工程现实选择；
论文中承认 3D/2.5D context 是未来方向；
当前创新落在 frequency/ROI/clinical fidelity，而不是大规模 3D 模型。
```

---

## 3. 现在应该怎么解决你当前的问题

### 3.1 不再继续盲目找新模型

论文和你的实验共同说明：

```text
换 backbone 不能本质解决“可信纹理来源”问题。
```

已经试过：

- Restormer / NAF / MB-NAF / ResAttentionUNet
- LPIPS+GAN
- CleanBase / LowGuard
- HP refiner
- PriorFlow
- T1-start flow
- Frequency fusion

因此下一步不是继续无限找新模型，而是收敛为：

```text
可信 texture prior
+
frequency/region-aware correction
+
dual-space evaluation
```

### 3.2 最建议保留的三条候选

#### 候选 A：Fidelity Flow Full

定位：

```text
最容易讲清楚的两阶段主线。
```

优势：

- ROI_CCC 强；
- 方法逻辑顺：sharp FA prior -> medical correction；
- 私有集和 ADNI 都有对应结果；
- 适合写论文主线。

问题：

- Stage2 有时提升不够显著；
- 如果 Stage1 过亮，Stage2 很难彻底修复。

#### 候选 B：A080 + DS HFKeep

定位：

```text
最强图像指标候选 / frequency-aware prior 候选。
```

优势：

- PSNR/SSIM/Sharp/WM-MAE 强；
- ADNI 下游 Accuracy 强；
- 与 FDDT/WFM 很契合。

问题：

- 容易被质疑是后处理融合；
- ROI_CCC 不如 Old Fidelity Flow；
- 需要包装成 frequency-aware prior，不要写成经验拼接。

#### 候选 C：PriorFlow Safe

定位：

```text
最新安全纹理候选。
```

优势：

- SharpRatio 接近 1；
- WM-MAE 好；
- ROI_CCC 比 A080 原始版强；
- 理论上可由 WFM/informed prior 支撑。

问题：

- 当前只是 1024/3 epoch safe probe；
- SSIM 偏低；
- 需要 4096/e8 放大验证。

---

## 4. 你的论文故事应该怎么讲

### 4.1 引言故事

建议结构：

1. FA/DTI 提供白质微结构信息，对 AD/MCI 等疾病分析重要。
2. DTI 获取慢、成本高、运动伪影多，T1 更常规。
3. T1-to-FA 是有意义的缺失模态补全任务。
4. 但 T1-to-FA 不是普通 image translation：
   - T1 是宏观结构；
   - FA 是白质微结构；
   - 单纯回归会变成均值解；
   - 单纯 GAN 会生成假纹理和统计漂移。
5. 因此需要同时处理：
   - distortion；
   - perceptual clarity；
   - clinical fidelity。

### 4.2 方法故事

推荐主叙事：

```text
We propose a fidelity-aware T1-to-FA synthesis framework that decouples clear FA prior generation from disease-sensitive regional correction.
```

中文：

```text
我们提出一种保真感知的 T1-to-FA 生成框架，将清晰 FA prior 的生成与疾病敏感区域的一致性校正解耦。
```

核心模块：

1. **Sharp / texture-aware FA prior**
   - 解决 posterior mean 平滑问题；
   - 保留可见白质细节；
   - 避免直接把 GAN 作为最终结果。

2. **Frequency-aware 或 informed-prior flow**
   - 低频保留结构和亮度；
   - 高频提供 FA-space texture；
   - 对应 WFM / FDDT 的启发。

3. **Disease-sensitive fidelity correction**
   - 校正 WM-MAE；
   - 提升 ROI_CCC；
   - 保持高频不被抹掉；
   - 抑制局部过亮伪纹理。

4. **Dual-space statistical consistency**
   - image-space：PSNR/SSIM/MAE；
   - region-space：ROI_CCC/ROI_Spearman/WM-MAE；
   - clinical-space：downstream ACC/AUC/F1。

### 4.3 失败方案如何写成贡献

你不应隐藏失败路线。它们可以成为动机和消融：

| 失败现象 | 论文解释 |
|---|---|
| posterior mean 模糊 | distortion 优化导致均值解 |
| LPIPS+GAN 过亮 | perception 提升但 clinical fidelity 漂移 |
| Stage2 补不回高频 | Stage1 已抹掉真实纹理，后验均值不可逆 |
| T1-start flow 像 T1 不像 FA | T1 高频不是 FA 高频 |
| CleanBase 太糊 | 伪影控制过强导致真实纹理被压制 |

这会让论文逻辑更强：

```text
不是我们随便试方法，而是逐步识别 T1-to-FA 中的 perception-distortion-clinical fidelity 矛盾。
```

---

## 5. 最终验证建议

### 5.1 只保留最终候选

建议现在固定：

```text
ADNI:
1. ADNI_PM_DIRF_FIDELITY_FLOW_FULL
2. ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6
3. ADNI_PRIOR_FLOW_SAFE_PROBE_FULL / 4096_e8

Private:
1. PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL
2. PM_DIRF_FIDELITY_FLOW_FULL
3. PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030
```

### 5.2 图像指标

必须统一：

```text
PSNR
SSIM
MSE
MAE
WM-MAE
ROI_CCC
ROI_Spearman
Sharpness_Ratio
Tenengrad
EdgeGrad
BlurDeficit
WM_Skeleton_Error
Slice_Consistency_Error
NonWMOver / LocalSpike / AnatomyUnsupportedSpike
```

### 5.3 下游任务

参考 final.pdf，建议保留：

```text
ROI mean intensity + SVM / linear classifier
T1 + generated FA early fusion
ACC / AUC / Macro-F1
```

任务：

```text
CN vs AD
CN vs MCI
MCI vs AD
CN+SCD vs MCI+AD
```

注意：

- 可以报告 ACC/AUC，但要说明 split。
- 不要用 slice-level 泄漏。
- 可同时报告 subject-level 和 MIL 协议。

---

## 6. 最终路线建议

### 如果优先考虑论文逻辑清楚

选：

```text
Fidelity Flow Full
```

写法：

```text
Sharp FA prior + disease-sensitive fidelity correction
```

### 如果优先考虑 ADNI 图像指标最强

选：

```text
A080 + DS HFKeep
```

写法：

```text
Frequency-aware sharp FA prior + high-frequency-preserving disease-sensitive correction
```

但必须避免写成“后处理融合”。

### 如果 PriorFlow 4096/e8 放大成功

可升级为：

```text
Informed-prior flow matching for T1-to-FA synthesis
```

写法更接近 WFM：

```text
从结构可信的 FA-space prior 出发，而不是从噪声、T1 或 posterior mean 出发。
```

---

## 7. 我建议你现在马上做的事

1. 跑完 PriorFlow safe 4096/e8，这是最后一个允许放大的候选。
2. 三个候选统一导出完整 ADNI test，确认 `n_slices=5616, n_subjects=108`。
3. 私有集也只保留 2-3 个候选，做同样指标表。
4. 固定横向可视化：

```text
T1 | FA_GT | Fidelity Flow | A080+DS | PriorFlow | Pix2Pix | UNet | CycleGAN
```

5. 下游任务用 ROI mean + SVM/linear classifier 和 attention-MIL 两套协议：
   - 一套贴近 final.pdf，展示 ACC/AUC；
   - 一套严格 subject-level/MIL，展示泛化可靠性。

---

## 8. 一句话最终故事

```text
T1-to-FA 生成的关键不只是提高 PSNR 或让图像更清晰，而是在 distortion、perceptual clarity 和 clinical fidelity 之间取得平衡。我们通过可信 FA 纹理 prior、频域结构-纹理分解和疾病敏感区域一致性校正，减少 posterior mean 模糊和 GAN 假纹理，使生成 FA 在白质/ROI 统计和下游疾病分析中更可靠。
```

