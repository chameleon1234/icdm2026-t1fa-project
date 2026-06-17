# final.pdf 下游分类差距诊断

## 结论

当前结果低，不是单纯因为数据集不够。更核心的是三个差异叠加：

1. **固定测试划分更难且测试样本极少**：当前 test 中 AD 只有 4 例，AD/CN 任务一个被试就会改变约 4.8% ACC，AD/MCI 一个被试会改变约 6.7% ACC。
2. **ROI 协议没有完全对齐 final.pdf**：final.pdf 使用 Neuromorphometrics 解剖模板；当前项目只有 AAL3 atlas mask。ROI 分区不同会直接改变 SVM 的输入特征。
3. **生成模型训练目标不同**：final.pdf 的 StructDINO 把 ROI anatomical prior、STAN 和 DSSC 放进生成模型训练，使生成 FA 直接服务 ROI 统计和下游分类；当前方法更多是在生成后用 ROI/SVM 做评估，训练阶段还没有同等强度的 anatomical ROI statistical alignment。

## 当前数据和划分

Excel 标签以 `1=CN, 2=SCD, 3=MCI, 4=AD` 为准。当前项目统计为：

| Group | Count |
|---|---:|
| CN | 92 |
| SCD | 56 |
| MCI | 70 |
| AD | 30 |

当前 subject split：

| Split | AD | CN | MCI | SCD |
|---|---:|---:|---:|---:|
| train | 21 | 60 | 49 | 43 |
| val | 5 | 15 | 10 | 7 |
| test | 4 | 17 | 11 | 6 |

final.pdf 描述为 80% / 20% subject-level train/test，并且同一 split 同时用于 FA synthesis training 和 downstream clinical validation。我们此前很多下游实验只用了 `train`，没有合并 `val`，这会比 final.pdf 少用 37 个被试训练下游分类器。

## train+val 诊断

我临时构建了 `outputs/icdm2026/subject_index_trainval.csv`，把 train 和 val 合并成 trainval，并创建了 `data/processed/trainval/{t1_slices,fa_slices}`。

使用 final.pdf 风格协议：

- subject-level ROI mean features
- AAL3 atlas ROI
- SVM
- `trainval -> test`
- 任务：CN vs AD、CN vs MCI、MCI vs AD

### RBF SVM, T1 / FA / T1+GT FA

| Method | Task | ACC | AUC |
|---|---|---:|---:|
| T1_ONLY | CN vs AD | 0.762 | 0.691 |
| FA_GT | CN vs AD | 0.714 | 0.838 |
| T1_PLUS_GT | CN vs AD | 0.714 | 0.868 |
| T1_ONLY | CN vs MCI | 0.536 | 0.631 |
| FA_GT | CN vs MCI | 0.607 | 0.588 |
| T1_PLUS_GT | CN vs MCI | 0.643 | 0.674 |
| T1_ONLY | MCI vs AD | 0.467 | 0.500 |
| FA_GT | MCI vs AD | 0.800 | 0.523 |
| T1_PLUS_GT | MCI vs AD | 0.667 | 0.591 |

### Linear SVM, T1 / FA / T1+GT FA

| Method | Task | ACC | AUC |
|---|---|---:|---:|
| T1_ONLY | CN vs AD | 0.667 | 0.529 |
| FA_GT | CN vs AD | 0.714 | 0.735 |
| T1_PLUS_GT | CN vs AD | 0.810 | 0.779 |
| T1_ONLY | CN vs MCI | 0.464 | 0.508 |
| FA_GT | CN vs MCI | 0.607 | 0.572 |
| T1_PLUS_GT | CN vs MCI | 0.571 | 0.599 |
| T1_ONLY | MCI vs AD | 0.600 | 0.591 |
| FA_GT | MCI vs AD | 0.733 | 0.523 |
| T1_PLUS_GT | MCI vs AD | 0.533 | 0.523 |

这说明仅合并 val 并不能复现 final.pdf 的高 ACC/AUC。更重要的是：连真实 FA + T1 在当前 AAL3 + 当前 test split 下也无法达到 final.pdf 的水平，所以问题不能简单归因于“合成 FA 不够好”。

## 随机 80/20 split 敏感性

为了判断是否是 split 难度导致，我用所有 248 被试的 AAL3 `T1+FA_GT` ROI 特征做 200 次随机 80/20 stratified split。

| Classifier | Task | ACC Mean | ACC Max | AUC Mean | AUC Max |
|---|---|---:|---:|---:|---:|
| Linear | CN vs AD | 0.881 | 1.000 | 0.900 | 1.000 |
| Linear | CN vs MCI | 0.650 | 0.818 | 0.689 | 0.887 |
| Linear | MCI vs AD | 0.687 | 0.900 | 0.695 | 0.929 |
| RBF | CN vs AD | 0.867 | 1.000 | 0.935 | 1.000 |
| RBF | CN vs MCI | 0.722 | 0.909 | 0.810 | 0.989 |
| RBF | MCI vs AD | 0.726 | 0.900 | 0.729 | 1.000 |

这说明：

- CN vs AD 在某些随机划分上确实可以非常高，final.pdf 的 96/98 并不不可思议。
- CN vs MCI 和 MCI vs AD 的随机 split 波动也很大；如果测试集更容易，ACC/AUC 会明显上升。
- 当前固定 split 可能偏难，尤其 AD 测试样本极少，任何一个异常被试都会显著改变结果。

## final.pdf 与当前方法的训练目标差异

final.pdf 的下游高指标不是单纯来自 SVM，而是生成模型训练就服务下游 ROI 统计：

- 使用 DINOv3 / LoRA 作为强结构编码器。
- 使用 atlas-derived anatomical attention bias，让网络在特征提取阶段关注解剖 ROI。
- 使用 STAN，把 anatomical fingerprint 注入 decoder。
- 使用 DSSC，直接约束 ROI intensity 和 ROI feature-space statistics。
- 论文中损失权重包括 `lambda_inten=5.0`、`lambda_struct=10.0`、`lambda_gan=0.5`。

当前 Fidelity Flow / Stage1 路线虽然已经改善了视觉和部分 ROI 指标，但没有等价的“训练阶段 anatomical ROI statistical alignment”。因此下游分类没有被生成模型主动优化到 final.pdf 那种程度。

## 下一步建议

1. **协议复现优先**：找到或构建 Neuromorphometrics atlas mask，复现 final.pdf 的 ROI mean SVM，不再只用 AAL3。
2. **重新定义正式 split**：如果目标是对齐 final.pdf，采用 80/20 train/test，并固定一个 stratified split；或者报告 repeated stratified 80/20 mean±std，避免单一 test split 偶然性。
3. **生成模型训练服务下游**：把 disease-sensitive anatomical ROI consistency 纳入 Stage1/Stage2 训练，而不是只在评估后做 SVM。
4. **不要只追 ACC**：由于 test 很小，论文应同时报告 ACC、AUC、Macro-F1、balanced ACC 和 repeated split 稳定性。
5. **如果要追 final.pdf 高 ACC/AUC**：最直接路线不是继续调普通图像 loss，而是实现一个轻量 StructDINO-style 模块：atlas ROI pooling + ROI intensity loss + ROI feature consistency loss。

