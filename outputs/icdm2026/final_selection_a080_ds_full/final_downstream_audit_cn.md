## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-30
- Verification Status: ANALYZED
- Version Label: final_downstream_audit_v1

## 最终下游任务审计报告

- **审计对象**: `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **预测目录**: `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`
- **ADNI manifest**: `data/adni_processed/adni_slice_manifest.csv`
- **审计输出目录**: `outputs/icdm2026/final_selection_a080_ds_full`
- **总体置信度**: SOLID for ADNI fair train/test MIL; CAUTION for private downstream because it is an existing historical audit, not a fresh rerun in this pass.

## 核心结论

1. ADNI 最新主方法下游已经完成公平 train/test MIL 审计，最终方法在表中以 `ADNI_A080_DS_FULL` 为别名出现。
2. `ADNI_A080_DS_FULL` 确认对应 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`，并且同时具备 test 预测目录和 train-full 预测目录。
3. 公平 train/test MIL 主表只包含具备训练集预测目录的方法: `T1_ONLY`, `FA_GT`, `ADNI_OLD_FIDELITY_FLOW`, `ADNI_A080_BASE`, `ADNI_A080_DS_FULL`。
4. `ADNI_UNET`, `ADNI_PIX2PIX`, `ADNI_CYCLEGAN` 只有 test 预测目录，没有对应 train-full 预测目录，因此不能进入公平 train/test MIL 主表，只能作为 subject-level test-CV 补充结果。
5. 当前脚本输出 Accuracy, Macro-AUC, Macro-F1；per-class F1 没有由现有下游脚本直接导出，不能在论文中虚构。
6. 结果不支持“最终方法所有下游指标第一”的强表述。更稳妥的表述是: A080+DS Full 提供了下游可用性证据，并保持有竞争力的分类表现；其主要优势仍然体现在重建质量、白质误差和清晰度保持上。

## 文件完整性

| 文件 | 状态 | 说明 |
|---|---:|---|
| `outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_subject_summary.csv` | 存在 | 最新 ADNI fair train/test MIL subject-level 汇总 |
| `outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_repeated_summary.csv` | 存在 | ADNI subject-level test-CV 补充对比 |
| `outputs/icdm2026/final_selection_a080_ds_full/final_selection_metrics.csv` | 存在 | 最终图像指标选择表 |
| `outputs/icdm2026/final_selection_a080_ds_full/adni_final_main_table.csv` | 存在 | ADNI 主图像指标表 |
| `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table_corrected.csv` | 已生成 | 修正后的最终下游总表 |
| `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_protocol_coverage.csv` | 已生成 | 方法覆盖与协议资格表 |

## ADNI 公平 Train/Test MIL 结果

协议: `slice_svm_vote`, `attention_mil`, `multi_task_mil`  
任务: `cn_vs_mci_spectrum`, `cn_vs_ad`, `mci_spectrum_vs_ad`  
聚合粒度: subject-level / MIL-level  
训练数据: train split  
测试数据: test split  
泄漏风险: 未发现 slice-level train/test 混用证据。

| Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---:|---:|---:|---:|
| `ADNI_A080_BASE` | 9 | 0.736 | 0.702 | 0.600 |
| `ADNI_A080_DS_FULL` | 9 | 0.710 | 0.723 | 0.593 |
| `ADNI_OLD_FIDELITY_FLOW` | 9 | 0.678 | 0.713 | 0.573 |
| `FA_GT` | 9 | 0.724 | 0.676 | 0.573 |
| `T1_ONLY` | 9 | 0.690 | 0.709 | 0.582 |

解释: 在公平 train/test MIL 设置下，`ADNI_A080_DS_FULL` 的平均 Macro-AUC 最高，但平均 ACC 和 Macro-F1 不是第一。因此它可以作为“下游有竞争力”的证据，但不应写成“全面优于所有方法”。

## ADNI 补充 Subject-Level Test-CV 结果

该协议用于补充展示所有已生成 test 预测目录的方法，包括 `ADNI_UNET`, `ADNI_PIX2PIX`, `ADNI_CYCLEGAN`。由于这些方法缺少 train-full 预测目录，这组结果不能替代公平 train/test MIL 主结果。

| Method | Rows | Mean ACC | Mean Macro-AUC | Mean Macro-F1 |
|---|---:|---:|---:|---:|
| `FA_GT` | 3 | 0.715 | 0.704 | 0.607 |
| `ADNI_PIX2PIX` | 3 | 0.623 | 0.466 | 0.517 |
| `ADNI_A080_DS_FULL` | 3 | 0.687 | 0.533 | 0.515 |
| `T1_ONLY` | 3 | 0.689 | 0.577 | 0.504 |
| `ADNI_OLD_FIDELITY_FLOW` | 3 | 0.658 | 0.617 | 0.506 |
| `ADNI_UNET` | 3 | 0.631 | 0.485 | 0.493 |
| `ADNI_A080_BASE` | 3 | 0.629 | 0.439 | 0.478 |
| `ADNI_CYCLEGAN` | 3 | 0.583 | 0.423 | 0.453 |

解释: 这组结果用于“方法覆盖更全”的补充对比。论文中必须注明其协议不同，不能和公平 MIL 主表直接合并排名。

## 私有集下游审计

私有集下游已有历史结果，但本轮没有重新全量训练或重跑私有集下游。已有文件显示私有集包含:

- `T1_ONLY`
- `FA_GT`
- `FidelityFlow`
- `Stage1`
- `UNet`
- `Pix2Pix`
- `CycleGAN`
- `T1_PLUS_*` 融合版本

相关目录:

- `outputs/icdm2026/downstream_private_slice_mil_finalpdf_train_test_seed2026`
- `outputs/icdm2026/downstream_finalpdf_compatible_private`
- `outputs/icdm2026/downstream_private_final_audit/downstream_private_audit_cn.md`

判断: 私有集结果可作为 existing/private audit 引用，但若论文需要和 ADNI 最终主方法完全对称的“最新最终候选”私有集结果，还需要单独确认 `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` 的 train/test 预测目录是否完整，并按同一 subject-level 协议补跑。

## Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Protocol mixing | CAUTION | fair train/test MIL 与 test-CV 补充结果协议不同 | 主表和补充表分开展示 |
| Data leakage | NOTE | 最新 ADNI MIL 使用 train/test split 和 subject-level 聚合，未见 slice-level 泄漏证据 | 保留 manifest 和命令记录 |
| Overclaiming | CAUTION | A080+DS Full 不是所有下游指标第一 | 使用“competitive downstream utility”表述 |
| Missing per-class F1 | NOTE | 当前脚本未导出 per-class F1 | 不在论文中宣称已有 per-class F1 |
| Baseline eligibility | CAUTION | UNet/Pix2Pix/CycleGAN 缺少 train-full 预测目录 | 仅作为 test-CV supplement |
| Final-method alias | NOTE | `ADNI_A080_DS_FULL` 是最终方法别名 | 在表注中写清全称 |
| Private/ADNI asymmetry | CAUTION | 私有集为历史审计，ADNI 为最新补跑 | 双数据集表中标明结果来源 |
| Single metric ranking | CAUTION | AUC, ACC, F1 排名不一致 | 不用单一指标宣称全面最优 |
| Sample dependence | NOTE | MIL 以 subject 为单位聚合 | 报告 subject-level 协议 |
| Multiple tasks | NOTE | 三个任务难度不同 | 分任务报告并提供平均值 |
| External validity | NOTE | ADNI 与私有集分布不同 | 双数据集分别报告 |

## Reproducibility

- **Method**: environment-sensitive rerun / artifact audit
- **Verdict**: PARTIALLY_REPRODUCIBLE

| Item | Status | Evidence |
|---|---|---|
| Final method prediction directory | MATCH | test 与 train-full 目录均存在 |
| ADNI manifest | MATCH | `data/adni_processed/adni_slice_manifest.csv` |
| Fair MIL result files | MATCH | subject summary 和 repeated summary 均存在 |
| Corrected downstream table | MATCH | `final_downstream_table_corrected.csv` |
| Byte-level deterministic rerun | Not checked | 本轮审计确认结构和结果，不做逐字节复现 |

## Safe Manuscript Wording

A080+DS Full 提供了下游可用性证据，并保持有竞争力的分类表现；其主要优势仍然体现在重建质量、白质误差和清晰度保持上。

英文建议:

`A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation.`

