# ADNI ROI ???? sweep ??

?????????????????????????????????

## ????????

| source                       | method            | classifier | max_features | accuracy | balanced_accuracy | macro_auc | macro_f1 |
| ---------------------------- | ----------------- | ---------- | ------------ | -------- | ----------------- | --------- | -------- |
| rbf_selectk                  | FA_GT             | rbf_svm    | 20           | 0.7398   | 0.6580            | 0.7467    | 0.6230   |
| rbf_selectk                  | Old Fidelity Flow | rbf_svm    | 20           | 0.7190   | 0.6080            | 0.7067    | 0.5882   |
| fast_logistic_linear_selectk | T1_PLUS_A080_BASE | logistic   | 40           | 0.6938   | 0.6037            | 0.7038    | 0.5654   |
| fast_logistic_linear_selectk | A080 Base         | linear_svm | 20           | 0.7803   | 0.6191            | 0.7001    | 0.6269   |
| fast_logistic_linear_selectk | T1_PLUS_FA_GT     | linear_svm | 0            | 0.6971   | 0.5492            | 0.6956    | 0.5436   |
| fast_logistic_linear_selectk | T1_PLUS_A080_DS   | logistic   | 20           | 0.6680   | 0.5804            | 0.6848    | 0.5455   |
| rbf_selectk                  | A080+DS Full      | rbf_svm    | 20           | 0.7543   | 0.5503            | 0.6848    | 0.5470   |
| fast_logistic_linear_selectk | T1_PLUS_OLD_FLOW  | logistic   | 40           | 0.7168   | 0.6125            | 0.6750    | 0.5876   |
| fast_logistic_linear_selectk | T1_ONLY           | logistic   | 20           | 0.6333   | 0.5888            | 0.6466    | 0.5387   |

## ??????

| source                                | method            | classifier     | max_features | accuracy | balanced_accuracy | macro_auc | macro_f1 |
| ------------------------------------- | ----------------- | -------------- | ------------ | -------- | ----------------- | --------- | -------- |
| rbf_selectk                           | FA_GT             | rbf_svm        | 20           | 0.7398   | 0.6580            | 0.7467    | 0.6230   |
| fast_logistic_linear_selectk          | FA_GT             | logistic       | 20           | 0.7519   | 0.7031            | 0.7244    | 0.6481   |
| rbf_selectk                           | Old Fidelity Flow | rbf_svm        | 20           | 0.7190   | 0.6080            | 0.7067    | 0.5882   |
| fast_logistic_linear_selectk          | T1_PLUS_A080_BASE | logistic       | 40           | 0.6938   | 0.6037            | 0.7038    | 0.5654   |
| fast_logistic_linear_selectk          | A080 Base         | linear_svm     | 20           | 0.7803   | 0.6191            | 0.7001    | 0.6269   |
| fast_logistic_linear_selectk          | T1_PLUS_FA_GT     | linear_svm     | 0            | 0.6971   | 0.5492            | 0.6956    | 0.5436   |
| rbf_selectk                           | A080+DS Full      | rbf_svm        | 20           | 0.7543   | 0.5503            | 0.6848    | 0.5470   |
| fast_logistic_linear_selectk          | T1_PLUS_A080_DS   | logistic       | 20           | 0.6680   | 0.5804            | 0.6848    | 0.5455   |
| fast_logistic_linear_selectk          | Old Fidelity Flow | logistic       | 20           | 0.7053   | 0.5930            | 0.6835    | 0.5729   |
| rbf_selectk                           | A080 Base         | rbf_svm        | 20           | 0.7823   | 0.5919            | 0.6796    | 0.5972   |
| fast_logistic_linear_selectk          | T1_PLUS_OLD_FLOW  | logistic       | 40           | 0.7168   | 0.6125            | 0.6750    | 0.5876   |
| fast_logistic_linear_selectk          | A080+DS Full      | logistic       | 20           | 0.7450   | 0.5514            | 0.6735    | 0.5397   |
| rbf_selectk                           | T1_PLUS_FA_GT     | rbf_svm        | 40           | 0.7356   | 0.5597            | 0.6706    | 0.5569   |
| rbf_selectk                           | T1_PLUS_A080_DS   | rbf_svm        | 40           | 0.7408   | 0.5959            | 0.6660    | 0.5866   |
| baseline_all_features_logistic_linear | Old Fidelity Flow | mixed_baseline | 0            | 0.7105   | 0.6006            | 0.6651    | 0.5767   |
| baseline_all_features_logistic_linear | A080 Base         | mixed_baseline | 0            | 0.7428   | 0.5864            | 0.6630    | 0.5746   |
| rbf_selectk                           | T1_PLUS_A080_BASE | rbf_svm        | 40           | 0.7664   | 0.5886            | 0.6610    | 0.5699   |
| baseline_all_features_logistic_linear | FA_GT             | mixed_baseline | 0            | 0.7372   | 0.5750            | 0.6580    | 0.5716   |
| baseline_all_features_logistic_linear | A080+DS Full      | mixed_baseline | 0            | 0.7260   | 0.5787            | 0.6497    | 0.5595   |
| fast_logistic_linear_selectk          | T1_ONLY           | logistic       | 20           | 0.6333   | 0.5888            | 0.6466    | 0.5387   |
| rbf_selectk                           | T1_PLUS_OLD_FLOW  | rbf_svm        | 40           | 0.7569   | 0.6058            | 0.6464    | 0.6016   |
| baseline_all_features_logistic_linear | T1_ONLY           | mixed_baseline | 0            | 0.7327   | 0.6161            | 0.6227    | 0.6021   |
| rbf_selectk                           | T1_ONLY           | rbf_svm        | 40           | 0.7473   | 0.5725            | 0.6205    | 0.5622   |
