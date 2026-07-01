# 论文主张与限制

## 可安全主张

1. 在 ADNI 上，`A080+DS Full` 在最新 fair full-heavy 下游审计中取得最高平均 Macro-AUC，但不是 Accuracy 或 Macro-F1 全部第一。
2. 在私有集上，`PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 的 integrated score 排名第一，但下游 Macro-AUC 不是第一。
3. 双数据集均完成了 subject-level train/test downstream evaluation，未使用 test predictions 作为 train predictions。
4. Synthetic FA 应被描述为 T1 的 complementary white-matter representation，而不是 T1 或真实 DTI/FA 的替代品。
5. 最终方法选择依据是 integrated balance：image reconstruction fidelity 25%、medical fidelity 30%、texture realism 20%、downstream utility 25%。

## 必须保留的限制

1. 下游分类结果受数据集规模、任务难度、标签分布、特征提取方式和分类协议影响。
2. 私有集上 T1_ONLY、UNet、FA_GT、Pix2Pix 在部分下游指标上强于 `PRIVATE_DS_HYBRID`。
3. `PRIVATE_DS_HYBRID` 不是私有集下游 Macro-AUC 单项第一；其价值主要体现为图像质量、医学一致性和纹理真实性的综合平衡。
4. 生成 FA 的主要价值是补充白质结构表征和医学一致性，而不是保证所有分类任务最优。
5. Synthetic FA cannot fully replace real DTI/FA；它只能作为真实 DTI/FA 不可用时的补充性表征。
6. 当前私有集 subject-level test subjects 平均约 25.33，分类结论需要谨慎解释。
