# ADNI ROI-level 下游分类补充评估

## 资源与协议

未发现可直接证明适配 ADNI 切片空间的 anatomical atlas mask。项目内存在 AAL3 原始文件和 private_aal3 mask，但这些不是本次 ADNI final prediction folders 的逐切片配准 mask。因此本次安全版本使用项目已有的 2×3 coarse grid brain/WM ROI 特征，不将其表述为解剖 atlas。

- 本实验没有训练新的 T1-to-FA 生成模型。
- 本实验没有改变最终主方法 A080+DS Full。
- 使用 ADNI manifest 的 subject-level train/test split；同一 subject 的切片不会跨 split。
- ROI 协议为项目已有 2×3 coarse grid + brain/WM mask 统计特征，不是 AAL/Neuromorphometrics 解剖 atlas。
- 该结果是 ROI-level supplementary utility evidence，不替代原有 image-level MIL downstream。

## 方法平均结果

| method            | rows | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects |
| ----------------- | ---- | ------------- | ---------------------- | -------------- | ------------- | --------------------- | -------------------- |
| Old Fidelity Flow | 6    | 0.7105        | 0.6006                 | 0.6651         | 0.5767        | 181.3333              | 52.0000              |
| A080 Base         | 6    | 0.7428        | 0.5864                 | 0.6630         | 0.5746        | 181.3333              | 52.0000              |
| FA_GT             | 6    | 0.7372        | 0.5750                 | 0.6580         | 0.5716        | 181.3333              | 52.0000              |
| A080+DS Full      | 6    | 0.7260        | 0.5787                 | 0.6497         | 0.5595        | 181.3333              | 52.0000              |
| T1_ONLY           | 6    | 0.7327        | 0.6161                 | 0.6227         | 0.6021        | 181.3333              | 52.0000              |

## 分任务结果

| method            | task               | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ------------------ | ------------- | ---------------------- | -------------- | ------------- |
| Old Fidelity Flow | cn_vs_ad           | 0.8750        | 0.7535                 | 0.8372         | 0.7160        |
| A080+DS Full      | cn_vs_ad           | 0.8854        | 0.7593                 | 0.8349         | 0.7327        |
| FA_GT             | cn_vs_ad           | 0.8438        | 0.6035                 | 0.8233         | 0.5950        |
| A080 Base         | cn_vs_ad           | 0.8958        | 0.7209                 | 0.8209         | 0.7184        |
| T1_ONLY           | cn_vs_ad           | 0.8438        | 0.6477                 | 0.7814         | 0.6298        |
| A080 Base         | cn_vs_mci_spectrum | 0.5753        | 0.5967                 | 0.6182         | 0.5749        |
| FA_GT             | cn_vs_mci_spectrum | 0.5822        | 0.5798                 | 0.6008         | 0.5762        |
| T1_ONLY           | cn_vs_mci_spectrum | 0.5685        | 0.5758                 | 0.5767         | 0.5664        |
| A080+DS Full      | cn_vs_mci_spectrum | 0.5068        | 0.5184                 | 0.5543         | 0.5061        |
| Old Fidelity Flow | cn_vs_mci_spectrum | 0.5137        | 0.5318                 | 0.5380         | 0.5137        |
| Old Fidelity Flow | mci_spectrum_vs_ad | 0.7429        | 0.5167                 | 0.6200         | 0.5006        |
| A080+DS Full      | mci_spectrum_vs_ad | 0.7857        | 0.4583                 | 0.5600         | 0.4397        |
| A080 Base         | mci_spectrum_vs_ad | 0.7571        | 0.4417                 | 0.5500         | 0.4306        |
| FA_GT             | mci_spectrum_vs_ad | 0.7857        | 0.5417                 | 0.5500         | 0.5435        |
| T1_ONLY           | mci_spectrum_vs_ad | 0.7857        | 0.6250                 | 0.5100         | 0.6101        |
