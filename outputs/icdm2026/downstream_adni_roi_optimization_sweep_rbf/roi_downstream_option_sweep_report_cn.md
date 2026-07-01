# ADNI ROI 下游任务优化尝试

本 sweep 只使用已经提取好的 ROI feature tables，没有训练新的图像生成模型，也没有改变最终主方法。

尝试内容：训练集内 SelectKBest 特征选择、Logistic / Linear SVM / RBF SVM / RandomForest、以及 T1+FA ROI 特征融合。

## 每个方法的最佳配置

| method            | classifier | max_features | rows | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ---------- | ------------ | ---- | ------------- | ---------------------- | -------------- | ------------- |
| FA_GT             | rbf_svm    | 20           | 3    | 0.7398        | 0.6580                 | 0.7467         | 0.6230        |
| Old Fidelity Flow | rbf_svm    | 20           | 3    | 0.7190        | 0.6080                 | 0.7067         | 0.5882        |
| A080+DS Full      | rbf_svm    | 20           | 3    | 0.7543        | 0.5503                 | 0.6848         | 0.5470        |
| A080 Base         | rbf_svm    | 20           | 3    | 0.7823        | 0.5919                 | 0.6796         | 0.5972        |
| T1_PLUS_FA_GT     | rbf_svm    | 40           | 3    | 0.7356        | 0.5597                 | 0.6706         | 0.5569        |
| T1_PLUS_A080_DS   | rbf_svm    | 40           | 3    | 0.7408        | 0.5959                 | 0.6660         | 0.5866        |
| T1_PLUS_A080_BASE | rbf_svm    | 40           | 3    | 0.7664        | 0.5886                 | 0.6610         | 0.5699        |
| T1_PLUS_OLD_FLOW  | rbf_svm    | 40           | 3    | 0.7569        | 0.6058                 | 0.6464         | 0.6016        |
| T1_ONLY           | rbf_svm    | 40           | 3    | 0.7473        | 0.5725                 | 0.6205         | 0.5622        |

## 前 20 个配置

| method            | classifier | max_features | rows | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ---------- | ------------ | ---- | ------------- | ---------------------- | -------------- | ------------- |
| FA_GT             | rbf_svm    | 20           | 3    | 0.7398        | 0.6580                 | 0.7467         | 0.6230        |
| FA_GT             | rbf_svm    | 40           | 3    | 0.7138        | 0.6742                 | 0.7104         | 0.6186        |
| Old Fidelity Flow | rbf_svm    | 20           | 3    | 0.7190        | 0.6080                 | 0.7067         | 0.5882        |
| Old Fidelity Flow | rbf_svm    | 40           | 3    | 0.7285        | 0.6136                 | 0.6869         | 0.5953        |
| A080+DS Full      | rbf_svm    | 20           | 3    | 0.7543        | 0.5503                 | 0.6848         | 0.5470        |
| A080 Base         | rbf_svm    | 20           | 3    | 0.7823        | 0.5919                 | 0.6796         | 0.5972        |
| T1_PLUS_FA_GT     | rbf_svm    | 40           | 3    | 0.7356        | 0.5597                 | 0.6706         | 0.5569        |
| A080 Base         | rbf_svm    | 40           | 3    | 0.7708        | 0.5841                 | 0.6670         | 0.5818        |
| T1_PLUS_A080_DS   | rbf_svm    | 40           | 3    | 0.7408        | 0.5959                 | 0.6660         | 0.5866        |
| T1_PLUS_A080_BASE | rbf_svm    | 40           | 3    | 0.7664        | 0.5886                 | 0.6610         | 0.5699        |
| A080+DS Full      | rbf_svm    | 40           | 3    | 0.7543        | 0.5486                 | 0.6518         | 0.5464        |
| T1_PLUS_OLD_FLOW  | rbf_svm    | 40           | 3    | 0.7569        | 0.6058                 | 0.6464         | 0.6016        |
| T1_PLUS_A080_DS   | rbf_svm    | 20           | 3    | 0.7126        | 0.5736                 | 0.6396         | 0.5638        |
| T1_PLUS_A080_BASE | rbf_svm    | 20           | 3    | 0.7104        | 0.5698                 | 0.6240         | 0.5624        |
| T1_ONLY           | rbf_svm    | 40           | 3    | 0.7473        | 0.5725                 | 0.6205         | 0.5622        |
| T1_ONLY           | rbf_svm    | 20           | 3    | 0.7077        | 0.5736                 | 0.6033         | 0.5629        |
| T1_PLUS_OLD_FLOW  | rbf_svm    | 20           | 3    | 0.7356        | 0.5875                 | 0.6004         | 0.5864        |
| T1_PLUS_FA_GT     | rbf_svm    | 20           | 3    | 0.7398        | 0.5969                 | 0.5886         | 0.5951        |
