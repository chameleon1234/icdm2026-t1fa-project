# 最终双阶段方法

## 最终选定方法

项目现在收口到直接双阶段模型：

```text
T1 5-slice -> Stage 1 清晰 FA 生成器 -> Stage 2 fidelity flow 校正器 -> final FA
```

最终 ADNI 输出目录是：

```text
outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL
```

Frequency Fusion 相关变体不再作为最终方法保留，因为双阶段 fidelity-flow 的视觉效果更符合当前论文目标，方法叙事也更干净。

## Stage 1：清晰 FA 生成器

Stage 1 使用 5-slice T1 上下文，并采用 sharp adversarial 训练策略，目标是避免 posterior-mean 平滑：

```text
T1_{z-2:z+2} -> 清晰 FA 初始预测
```

Checkpoint：

```text
outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt
```

导出结果：

```text
outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL
```

## Stage 2：Fidelity Flow 校正器

Stage 2 是 flow-based medical-fidelity corrector。它在 Stage 1 清晰输出的基础上做医学一致性校正，不再额外接 Frequency Fusion 后处理分支。

Checkpoint：

```text
outputs/adni_pmrf_stage2_fidelity_flow_full_e40/checkpoints/best_fidelity_corrector.pt
```

导出结果：

```text
outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL
```

## 论文表述

最终论文方法应描述为一个双阶段 T1-to-FA 生成框架：

1. 高频保持 Stage 1 生成器用于避免 PMRF-style posterior mean 带来的模糊。
2. Fidelity Flow 校正器在保留 Stage 1 清晰结构的同时提升医学一致性。
3. 下游 AD 分类用于验证生成 FA 的疾病相关 utility，而不仅是 paired image fidelity。

不要再把 Frequency Balanced Fusion 写进最终方法。
