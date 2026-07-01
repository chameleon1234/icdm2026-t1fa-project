# Results 章节草稿（中文）

## 双数据集公平下游闭环

本项目已经在 ADNI 和私有数据集上完成 fair subject-level train/test downstream evaluation。下游分类器只使用训练集训练，并在独立测试集上评估；结果采用 subject-level aggregation 或 MIL-level 输出。测试集预测没有被用作训练预测，test-CV 结果也没有混入主结论。

## ADNI 结果

在 ADNI 全方法公平下游评估中，`A080+DS Full` 取得最高平均 Macro-AUC，为 0.7226。该结果支持最终 ADNI 主方法具有下游可用性。

需要谨慎的是，`A080+DS Full` 并不是所有下游单项指标第一：DDIM 的 Accuracy 更高，MOTFM 的 Macro-F1 更高。因此 ADNI 下游结论应表述为：

`A080+DS Full achieves the highest Macro-AUC and competitive Accuracy/F1, supporting downstream utility, not universal downstream superiority.`

从图像与医学指标看，`A080+DS Full` 同时在重建保真度、白质/ROI 医学一致性和清晰度保持之间取得了最稳定的综合平衡。Macro-AUC 结果是对图像层面证据的补充，而不是替代。

## 私有集结果

在私有集 full-heavy fair downstream 中，`PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 不是最强的单独下游分类方法。它的平均 Macro-AUC 为 0.5913，低于 UNet (0.7244)、FA_GT (0.7046)、Pix2Pix (0.6997) 和 Stage1_LPIPS_GAN (0.6950)。

私有集 per-task sanity check 显示，`PRIVATE_DS_HYBRID` 最弱的任务是 `CN vs MCI-spectrum`，Accuracy = 0.4706，Macro-AUC = 0.4406，Macro-F1 = 0.4112。`CN vs AD` 和 `MCI-spectrum vs AD` 能正常运行，但也不是全面领先。

同时，`PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 在 integrated image/medical/texture/downstream score 中排名第一。因此，私有集结论应强调综合平衡，而不是下游分类单项优势。

## 安全的双数据集解释

双数据集结果共同说明，synthetic FA 更适合作为 T1 的 complementary white-matter representation，而不是替代 T1 或真实 DTI/FA。最终方法选择依据应写成 reconstruction quality、white-matter/ROI medical consistency、texture preservation 和 downstream utility 之间的 integrated balance。

论文中不应声称该方法所有指标第一，也不应声称双数据集下游都第一，更不应声称生成 FA 可以完全替代真实 FA。
