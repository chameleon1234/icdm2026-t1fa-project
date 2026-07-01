# 私有集 per-task sanity check

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: private_per_task_sanity_v1

## 1. PRIVATE_DS_HYBRID 每任务表现

| task | Accuracy | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|
| cn_vs_mci_spectrum | 0.4706 | 0.4406 | 0.4112 |
| cn_vs_ad | 0.7302 | 0.6373 | 0.6221 |
| mci_spectrum_vs_ad | 0.6984 | 0.6961 | 0.5938 |

最弱任务是 **CN vs MCI-spectrum**。该任务的 Macro-AUC 为 0.4406，是拉低 `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 平均 Macro-AUC 的主要原因。

## 2. 任务层面解释

- **CN vs AD**：任务能够正常运行，PRIVATE_DS_HYBRID 的 Macro-AUC 为 0.6373，但低于 FA_GT、UNet、Pix2Pix、T1_ONLY 等方法。
- **CN vs MCI-spectrum**：这是 PRIVATE_DS_HYBRID 最弱任务，Macro-AUC 为 0.4406，说明它对早期谱系区分不稳定。
- **MCI-spectrum vs AD**：PRIVATE_DS_HYBRID 的 Macro-AUC 为 0.6961，属于中等水平，但仍低于 T1_ONLY、FidelityFlow、UNet、Pix2Pix、Stage1_LPIPS_GAN 等方法。

## 3. 方法平均结果

| method | Accuracy | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|
| UNet | 0.6580 | 0.7244 | 0.5918 |
| FA_GT | 0.6556 | 0.7046 | 0.5908 |
| Pix2Pix | 0.6229 | 0.6997 | 0.5296 |
| Stage1_LPIPS_GAN | 0.6335 | 0.6950 | 0.5621 |
| T1_ONLY | 0.6343 | 0.6895 | 0.5761 |
| FidelityFlow | 0.6730 | 0.6862 | 0.5992 |
| CycleGAN | 0.6323 | 0.6650 | 0.5597 |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 0.6106 | 0.5975 | 0.5329 |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL | 0.6331 | 0.5913 | 0.5424 |

## 4. Subject 数一致性

同一 protocol/task 下，各方法的 train/test subject 数一致，没有发现某个方法额外使用 subject 的异常。不同任务 subject 数不同，是因为任务纳入的诊断组不同。

关键任务 subject 数如下：

| task | train subjects | test subjects |
|---|---:|---:|
| cn_vs_ad | 81 | 21 |
| cn_vs_mci_spectrum | 152 | 34 |
| mci_spectrum_vs_ad | 113 | 21 |

## 5. 标签和 MCI-spectrum 定义

项目标签映射为：

- 1 = CN
- 2 = SCD
- 3 = MCI
- 4 = AD

本轮已确认 `MCI-spectrum = SCD + MCI`。因此私有集的 `cn_vs_mci_spectrum`、`mci_spectrum_vs_ad` 与 ADNI 中 `MCI_spectrum` 任务语义对齐。

## 6. 泄漏和聚合检查

未发现 train/test subject 重叠。下游结果来自 subject-level aggregation / MIL-level 协议，未发现 slice-level 泄漏或 subject 聚合异常。
