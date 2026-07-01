# Results 部分写作草稿

## ADNI 图像质量结果

在 ADNI 测试集上，A080+DS Full 在重建保真度、白质/ROI 医学一致性与纹理真实性之间取得较好的平衡。与 Old Fidelity Flow 和 A080 Base 相比，A080+DS Full 保持了较高的 PSNR/SSIM，同时在医学一致性和综合评分上更稳定。Pix2Pix、U-Net、CycleGAN 在部分单项指标上仍有竞争力，因此论文中不应声称 A080+DS Full 在所有单项指标上第一。

## ADNI 下游结果

在严格 subject-level train/test 下游评估中，A080+DS Full 的平均 Macro-AUC 为 `0.7226`，是 ADNI 公平下游表中最高的 Macro-AUC。A080 Base 在 Accuracy 和 Macro-F1 上仍有竞争力，说明纹理清晰度、医学一致性和下游任务之间存在任务相关 trade-off。

## 私有集图像质量结果

私有集上，PRIVATE_DS_HYBRID 在图像质量、白质误差、ROI 一致性和清晰度保持之间表现稳定。但这不意味着它在所有下游任务上都优于 T1 或其他生成方法。该结果更适合支持 synthetic FA 是 T1 的补充表征，而不是 T1 的完全替代。

## 私有集下游结果

私有集公平下游显示，T1_ONLY 的平均 Macro-AUC 最高，FidelityFlow 的平均 Accuracy 较强，PRIVATE_DS_HYBRID 保持可竞争表现但不是所有下游指标第一。这提示私有集的下游疾病信号可能更多依赖 T1 原始结构或低频统计。

## Integrated Composite 结果

Primary Integrated Composite Score 使用固定权重：图像重建保真度 25%，医学一致性 30%，纹理真实性 20%，下游可用性 25%。ADNI 中 A080+DS Full 按该综合评分排名第一。私有集中 PRIVATE_DS_HYBRID 按该综合评分排名第一。

最终安全表述应为：A080+DS Full 被选为最终方法，是因为它在重建保真度、白质/ROI 医学一致性、纹理真实性和下游可用性之间取得最佳综合平衡，而不是因为它在每一个单项指标上都第一。

## 消融解释

已有消融说明，仅提升 LPIPS 或 GAN 压力并不能稳定带来真实 FA 纹理；过强锐化容易引入伪白质纹理或局部过亮。A080+DS 的价值在于先用 A080 保留较稳定的 FA-space 高频，再用 disease-sensitive corrector 做医学一致性校正。

## 安全结论

最终结论建议写成：A080+DS Full 提供了稳定的双数据集重建证据和 ADNI 下游可用性证据；私有集下游结果提示 synthetic FA 应被视为 T1 的互补表征。不要写生成 FA 全面替代 T1，也不要写所有指标第一。


## 2026-07-01 ???? fair full-heavy ??????

???????????????????? train-full prediction folders??? subject-level train/test downstream ????????ADNI ???? `outputs/icdm2026/downstream_adni_fair_full_all_methods/method_average_summary.csv`?????? `outputs/icdm2026/downstream_private_fair_full_heavy_all_methods/method_average_summary.csv`?

ADNI ??`ADNI_A080_DS_FULL` ??? Macro-AUC ???????? ADNI ???????????`PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` ????? Macro-AUC ?????UNet?FA_GT?Pix2Pix ??????????????????????????? FA ? T1 ? complementary representation?????????????????????????????? integrated balance??????????????
