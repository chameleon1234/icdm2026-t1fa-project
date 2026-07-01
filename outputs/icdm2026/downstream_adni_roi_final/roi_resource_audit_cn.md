# ADNI ROI 资源审计

未发现可直接证明适配 ADNI 切片空间的 anatomical atlas mask。项目内存在 AAL3 原始文件和 private_aal3 mask，但这些不是本次 ADNI final prediction folders 的逐切片配准 mask。因此本次安全版本使用项目已有的 2×3 coarse grid brain/WM ROI 特征，不将其表述为解剖 atlas。

## 目录覆盖

| method            | train_dir                                                                                               | test_dir                                                                                     | runnable | train_png | train_split_counts | train_subjects | test_png | test_split_counts | test_subjects |
| ----------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------- | --------- | ------------------ | -------------- | -------- | ----------------- | ------------- |
| T1_ONLY           | data\adni_processed\train\t1_slices                                                                     | data\adni_processed\test\t1_slices                                                           | True     | 19552     | {"train": 19552}   | 376            | 5616     | {"test": 5616}    | 108           |
| FA_GT             | data\adni_processed\train\fa_slices                                                                     | data\adni_processed\test\fa_slices                                                           | True     | 19552     | {"train": 19552}   | 376            | 5616     | {"test": 5616}    | 108           |
| Old Fidelity Flow | outputs\icdm2026\predictions\ADNI_PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL                                 | outputs\icdm2026\predictions\ADNI_PM_DIRF_FIDELITY_FLOW_FULL                                 | True     | 19552     | {"train": 19552}   | 376            | 5616     | {"test": 5616}    | 108           |
| A080 Base         | outputs\icdm2026\predictions\ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL                      | outputs\icdm2026\predictions\ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080                      | True     | 19552     | {"train": 19552}   | 376            | 5616     | {"test": 5616}    | 108           |
| A080+DS Full      | outputs\icdm2026\predictions\ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_TRAIN_FULL | outputs\icdm2026\predictions\ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST | True     | 19552     | {"train": 19552}   | 376            | 5616     | {"test": 5616}    | 108           |
