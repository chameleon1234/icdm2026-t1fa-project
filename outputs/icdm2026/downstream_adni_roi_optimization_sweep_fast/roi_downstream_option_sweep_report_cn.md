# ADNI ROI 下游任务优化尝试

本 sweep 只使用已经提取好的 ROI feature tables，没有训练新的图像生成模型，也没有改变最终主方法。

尝试内容：训练集内 SelectKBest 特征选择、Logistic / Linear SVM / RBF SVM / RandomForest、以及 T1+FA ROI 特征融合。

## 每个方法的最佳配置

| method            | classifier | max_features | rows | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ---------- | ------------ | ---- | ------------- | ---------------------- | -------------- | ------------- |
| FA_GT             | logistic   | 20           | 3    | 0.7519        | 0.7031                 | 0.7244         | 0.6481        |
| T1_PLUS_A080_BASE | logistic   | 40           | 3    | 0.6938        | 0.6037                 | 0.7038         | 0.5654        |
| A080 Base         | linear_svm | 20           | 3    | 0.7803        | 0.6191                 | 0.7001         | 0.6269        |
| T1_PLUS_FA_GT     | linear_svm | 0            | 3    | 0.6971        | 0.5492                 | 0.6956         | 0.5436        |
| T1_PLUS_A080_DS   | logistic   | 20           | 3    | 0.6680        | 0.5804                 | 0.6848         | 0.5455        |
| Old Fidelity Flow | logistic   | 20           | 3    | 0.7053        | 0.5930                 | 0.6835         | 0.5729        |
| T1_PLUS_OLD_FLOW  | logistic   | 40           | 3    | 0.7168        | 0.6125                 | 0.6750         | 0.5876        |
| A080+DS Full      | logistic   | 20           | 3    | 0.7450        | 0.5514                 | 0.6735         | 0.5397        |
| T1_ONLY           | logistic   | 20           | 3    | 0.6333        | 0.5888                 | 0.6466         | 0.5387        |

## 前 20 个配置

| method            | classifier | max_features | rows | mean_accuracy | mean_balanced_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ---------- | ------------ | ---- | ------------- | ---------------------- | -------------- | ------------- |
| FA_GT             | logistic   | 20           | 3    | 0.7519        | 0.7031                 | 0.7244         | 0.6481        |
| FA_GT             | linear_svm | 20           | 3    | 0.7263        | 0.6842                 | 0.7166         | 0.6247        |
| T1_PLUS_A080_BASE | logistic   | 40           | 3    | 0.6938        | 0.6037                 | 0.7038         | 0.5654        |
| A080 Base         | linear_svm | 20           | 3    | 0.7803        | 0.6191                 | 0.7001         | 0.6269        |
| FA_GT             | logistic   | 40           | 3    | 0.6999        | 0.5814                 | 0.6963         | 0.5662        |
| T1_PLUS_FA_GT     | linear_svm | 0            | 3    | 0.6971        | 0.5492                 | 0.6956         | 0.5436        |
| T1_PLUS_FA_GT     | linear_svm | 40           | 3    | 0.6777        | 0.6637                 | 0.6928         | 0.5911        |
| T1_PLUS_FA_GT     | logistic   | 0            | 3    | 0.7013        | 0.5864                 | 0.6887         | 0.5726        |
| T1_PLUS_A080_BASE | logistic   | 20           | 3    | 0.6775        | 0.5837                 | 0.6874         | 0.5530        |
| T1_PLUS_A080_DS   | logistic   | 20           | 3    | 0.6680        | 0.5804                 | 0.6848         | 0.5455        |
| T1_PLUS_A080_BASE | linear_svm | 40           | 3    | 0.7172        | 0.6364                 | 0.6844         | 0.6043        |
| Old Fidelity Flow | logistic   | 20           | 3    | 0.7053        | 0.5930                 | 0.6835         | 0.5729        |
| A080 Base         | logistic   | 20           | 3    | 0.7755        | 0.6048                 | 0.6830         | 0.5786        |
| T1_PLUS_A080_BASE | linear_svm | 20           | 3    | 0.6822        | 0.5787                 | 0.6817         | 0.5579        |
| T1_PLUS_A080_DS   | linear_svm | 20           | 3    | 0.6731        | 0.5992                 | 0.6803         | 0.5598        |
| T1_PLUS_OLD_FLOW  | logistic   | 40           | 3    | 0.7168        | 0.6125                 | 0.6750         | 0.5876        |
| T1_PLUS_FA_GT     | logistic   | 20           | 3    | 0.6495        | 0.5965                 | 0.6737         | 0.5488        |
| A080+DS Full      | logistic   | 20           | 3    | 0.7450        | 0.5514                 | 0.6735         | 0.5397        |
| A080 Base         | logistic   | 0            | 3    | 0.7331        | 0.5959                 | 0.6731         | 0.5830        |
| A080+DS Full      | logistic   | 40           | 3    | 0.7618        | 0.5503                 | 0.6727         | 0.5406        |
