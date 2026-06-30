## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-06-30
- Verification Status: ANALYZED
- Version Label: fair_downstream_cached_roi_v1

## Private Fair Train/Test MIL Downstream Report

- eligible methods: T1_ONLY, FA_GT, FIDELITY_FLOW, PRIVATE_DS_HYBRID, UNet, Pix2Pix, CycleGAN
- protocols: slice_svm_vote, attention_mil, multi_task_mil
- Train and test splits are separate; test predictions were not used as training predictions.
- Per-class F1 was recomputed from subject-level y_true/y_pred predictions.

## Method Average Summary

| method            | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects |
| ----------------- | ---- | ------------- | -------------- | ------------- | --------------------- | -------------------- |
| T1_ONLY           | 9    | 0.6659        | 0.7285         | 0.6267        | 129.6667              | 24.6667              |
| Pix2Pix           | 9    | 0.6483        | 0.7097         | 0.6049        | 129.6667              | 24.6667              |
| UNet              | 9    | 0.6730        | 0.7072         | 0.6229        | 129.6667              | 24.6667              |
| CycleGAN          | 9    | 0.6992        | 0.6864         | 0.6509        | 129.6667              | 24.6667              |
| FIDELITY_FLOW     | 9    | 0.7050        | 0.6815         | 0.6362        | 129.6667              | 24.6667              |
| FA_GT             | 9    | 0.6964        | 0.6639         | 0.6039        | 129.6667              | 24.6667              |
| PRIVATE_DS_HYBRID | 9    | 0.6902        | 0.6493         | 0.6042        | 129.6667              | 24.6667              |

## Per Task Summary

| method            | task             | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 |
| ----------------- | ---------------- | ---- | ------------- | -------------- | ------------- |
| T1_ONLY           | cn_scd_vs_mci_ad | 3    | 0.6579        | 0.7314         | 0.6499        |
| CycleGAN          | cn_scd_vs_mci_ad | 3    | 0.6754        | 0.7053         | 0.6691        |
| FIDELITY_FLOW     | cn_scd_vs_mci_ad | 3    | 0.6579        | 0.6773         | 0.6199        |
| Pix2Pix           | cn_scd_vs_mci_ad | 3    | 0.6053        | 0.6696         | 0.5653        |
| UNet              | cn_scd_vs_mci_ad | 3    | 0.6316        | 0.6676         | 0.5977        |
| PRIVATE_DS_HYBRID | cn_scd_vs_mci_ad | 3    | 0.6579        | 0.6502         | 0.6158        |
| FA_GT             | cn_scd_vs_mci_ad | 3    | 0.5877        | 0.5700         | 0.5349        |
| T1_ONLY           | cn_vs_ad         | 3    | 0.6508        | 0.7647         | 0.5995        |
| UNet              | cn_vs_ad         | 3    | 0.6984        | 0.7647         | 0.6354        |
| FA_GT             | cn_vs_ad         | 3    | 0.7460        | 0.7549         | 0.6492        |
| Pix2Pix           | cn_vs_ad         | 3    | 0.6508        | 0.7549         | 0.6014        |
| CycleGAN          | cn_vs_ad         | 3    | 0.7778        | 0.7402         | 0.7100        |
| FIDELITY_FLOW     | cn_vs_ad         | 3    | 0.7460        | 0.7157         | 0.6483        |
| PRIVATE_DS_HYBRID | cn_vs_ad         | 3    | 0.7460        | 0.6765         | 0.6346        |
| Pix2Pix           | mci_vs_ad        | 3    | 0.6889        | 0.7045         | 0.6481        |
| T1_ONLY           | mci_vs_ad        | 3    | 0.6889        | 0.6894         | 0.6306        |
| UNet              | mci_vs_ad        | 3    | 0.6889        | 0.6894         | 0.6357        |
| FA_GT             | mci_vs_ad        | 3    | 0.7556        | 0.6667         | 0.6275        |
| FIDELITY_FLOW     | mci_vs_ad        | 3    | 0.7111        | 0.6515         | 0.6405        |
| PRIVATE_DS_HYBRID | mci_vs_ad        | 3    | 0.6667        | 0.6212         | 0.5623        |
| CycleGAN          | mci_vs_ad        | 3    | 0.6444        | 0.6136         | 0.5736        |
