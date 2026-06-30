# 最终双数据集下游与综合评分修复报告（Clean）

## 1. 总体结论

本次修复解决了 ADNI Pix2Pix / U-Net / CycleGAN 在 integrated comparison table 中 downstream 指标为空的问题。根因是图像质量表使用 `Pix2Pix / U-Net / CycleGAN`，而公平下游表使用 `ADNI_PIX2PIX / ADNI_UNET / ADNI_CYCLEGAN`，旧版 alias mapping 没有正确匹配。

修复后，ADNI Pix2Pix、U-Net、CycleGAN 的 `downstream_ACC`、`downstream_AUC`、`downstream_F1` 已成功合并到 `final_integrated_comparison_table_fixed.csv`。

ADNI 中，A080+DS Full 按 Primary Integrated Composite Score 排名第 1，分数为 `0.9642`。私有集中，`PRIVATE_DS_HYBRID` 按 Primary Integrated Composite Score 排名第 1，分数为 `0.7784`。

安全结论：A080+DS Full 被选为最终方法，是因为它在重建保真度、白质/ROI 医学一致性、纹理真实性和下游可用性之间取得最佳综合平衡，而不是因为它在每一个单项指标上都第一。

## 2. ADNI 公平下游结果

| 方法 | Accuracy | Macro-AUC | Macro-F1 | 备注 |
|---|---:|---:|---:|---|
| A080+DS Full | 0.7100 | 0.7226 | 0.5931 | ADNI Macro-AUC 最高 |
| Old Fidelity Flow | 0.6776 | 0.7133 | 0.5726 | 强两阶段基线 |
| T1_ONLY | 0.6898 | 0.7090 | 0.5815 | 原始 T1 表征 |
| A080 Base | 0.7364 | 0.7016 | 0.6004 | Accuracy / F1 有竞争力 |
| Pix2Pix | 0.7120 | 0.6892 | 0.5989 | downstream merge 已修复 |
| U-Net | 0.6072 | 0.6885 | 0.5134 | downstream merge 已修复 |
| FA_GT | 0.7241 | 0.6757 | 0.5727 | 真实 FA 参考 |
| CycleGAN | 0.6900 | 0.6726 | 0.5664 | downstream merge 已修复 |
| LightGuard PriorFlow | 0.5959 | 0.6533 | 0.4263 | 纹理探索分支 |

## 3. 私有集公平下游结果

| 方法 | Accuracy | Macro-AUC | Macro-F1 | 备注 |
|---|---:|---:|---:|---|
| T1_ONLY | 0.6659 | 0.7285 | 0.6267 | 私有集 Macro-AUC 最高 |
| Pix2Pix | 0.6483 | 0.7097 | 0.6049 | 有竞争力 |
| U-Net | 0.6730 | 0.7072 | 0.6229 | 有竞争力 |
| CycleGAN | 0.6992 | 0.6864 | 0.6509 | Macro-F1 最高 |
| FidelityFlow | 0.7050 | 0.6815 | 0.6362 | Accuracy 最高 |
| FA_GT | 0.6964 | 0.6639 | 0.6039 | 真实 FA 参考 |
| PRIVATE_DS_HYBRID | 0.6902 | 0.6493 | 0.6042 | 最终私有候选，但下游不是全第一 |

## 4. 协议覆盖情况

严格公平下游只使用 train split 训练分类器，并在 test split 做 subject-level 聚合评估。没有使用 test-as-train，也没有混用 test-CV 与 fair MIL。ADNI 使用 `data/adni_processed/adni_slice_manifest.csv`；私有集使用 `data/processed/train` 和 `data/processed/test`。

## 5. 未纳入公平下游的方法及原因

- ADNI DDIM / DBM / MOTFM：缺少 train-full prediction folder，因此未进入严格公平下游主表。
- 私有集 `PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030` 和 `Stage1_LPIPS_GAN`：缺少 train-full prediction folder，因此未进入私有集严格公平下游主表。
- 这些方法没有被删除，仍保留在图像质量表、coverage 表或 fixed integrated table 中；只是没有进入严格下游排名。

## 6. 综合评分结果

Primary Integrated Composite Score 使用固定权重：图像重建保真度 25%，医学一致性 30%，纹理真实性 20%，下游可用性 25%。权重没有为了让 A080+DS Full 第一而修改。

| 数据集 | 方法 | Primary score | Rank | 说明 |
|---|---|---:|---:|---|
| ADNI | A080+DS Full | 0.9642 | 1 | 最终 ADNI 主方法 |
| ADNI | Old Fidelity Flow | 0.8199 | 2 | 强两阶段基线 |
| ADNI | Pix2Pix | 0.7504 | 3 | downstream merge 已修复 |
| ADNI | A080 Base | 0.7442 | 4 | 清晰基底 |
| ADNI | U-Net | 0.6255 | 5 | downstream merge 已修复 |
| ADNI | LightGuard PriorFlow | 0.4891 | 6 | 探索分支 |
| ADNI | CycleGAN | 0.2211 | 7 | downstream merge 已修复 |
| Private | PRIVATE_DS_HYBRID | 0.7784 | 1 | 私有最终候选 |
| Private | FidelityFlow | 0.6828 | 2 | 两阶段基线 |
| Private | Pix2Pix | 0.6667 | 3 | 对比方法 |
| Private | U-Net | 0.6279 | 4 | 对比方法 |
| Private | CycleGAN | 0.2858 | 5 | 对比方法 |

完整 raw metrics 保存在 `final_integrated_comparison_table_fixed.csv`。

## 7. 论文安全表述

- 可以写：ADNI 中 A080+DS Full 的 Macro-AUC 最高，并且按 primary integrated composite score 排名第一。
- 不能写：A080+DS Full 所有单项指标第一。
- 私有集应写成：synthetic FA 是 T1 的 complementary representation，而不是 T1 的完全替代。
- 私有集下游最强方法不一定是 PRIVATE_DS_HYBRID，这一点需要保留。

## 8. 局限性

下游公平评估依赖 train-full prediction folder。缺少 train-full prediction 的方法不能进入严格公平下游主表；这是一种协议完整性限制，不是有意删除不利结果。

## 9. 下一步论文写作建议

Results 部分建议按“图像质量 -> 医学一致性 -> 下游可用性 -> 综合选择”的顺序写。最终方法选择要强调综合平衡，而不是单项冠军。
