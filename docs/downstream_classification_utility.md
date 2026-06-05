# Downstream Disease Utility Evaluation

## Purpose

This experiment tests whether generated FA images preserve disease-relevant information for ICDM-style data mining, beyond image fidelity scores alone. It is intentionally subject-level: all slices from one subject are aggregated into one feature vector before classification or regression, so slice-level leakage is avoided.

## Current Methods

- `T1_ONLY`: input T1 reference baseline.
- `Stage1_Baseline`: conservative Stage1 PMRF-style prediction.
- `Stage1_LPIPS_GAN`: sharp Stage1 prediction with high-frequency preservation.
- `Fidelity_Flow`: frequency-preserving Stage2 flow corrector.
- `Fidelity_Direct`: direct low-frequency corrector ablation.
- `FA_GT`: real FA upper bound.
- `T1_PLUS_*`: feature-level fusion between T1 and a generated/real FA source.

## Tasks

- `four_class`: CN / SCD / MCI / AD.
- `cn_vs_ad`: CN vs AD.
- `cn_vs_mci_ad`: CN vs MCI+AD.
- `cn_scd_vs_mci_ad`: CN+SCD vs MCI+AD.
- `MMSE` regression: auxiliary cognitive-score regression.

For the current paper, the most useful positive findings are disease-stage classification tasks, not MMSE regression.

## Command

```powershell
conda activate dinov3test

python scripts/evaluate_downstream_classification.py `
  --include_t1 `
  --include_fa_gt `
  --method Stage1_Baseline=outputs\icdm2026\predictions\PM_STAGE1_WMROI_DETAIL_5SLICE `
  --method Stage1_LPIPS_GAN=outputs\icdm2026\predictions\PM_STAGE1_LPIPS_GAN_5SLICE_FINAL `
  --method Fidelity_Flow=outputs\icdm2026\predictions\PM_DIRF_FIDELITY_FLOW_FULL `
  --method Fidelity_Direct=outputs\icdm2026\predictions\PM_DIRF_FIDELITY_DIRECT_FULL `
  --fusion T1_PLUS_STAGE1_BASELINE=T1_ONLY+Stage1_Baseline `
  --fusion T1_PLUS_STAGE1_LPIPS_GAN=T1_ONLY+Stage1_LPIPS_GAN `
  --fusion T1_PLUS_FIDELITY_FLOW=T1_ONLY+Fidelity_Flow `
  --fusion T1_PLUS_FIDELITY_DIRECT=T1_ONLY+Fidelity_Direct `
  --fusion T1_PLUS_FA_GT=T1_ONLY+FA_GT `
  --tasks four_class,cn_vs_ad,cn_vs_mci_ad,cn_scd_vs_mci_ad `
  --regression_targets MMSE `
  --n_splits 5 `
  --max_features 12 `
  --output_root outputs\icdm2026\downstream_classification
```

`--max_features 12` performs fold-internal univariate feature selection. This is important because the test split has only 38 subjects, so direct high-dimensional feature concatenation overfits.

## Outputs

- `outputs/icdm2026/downstream_classification/subject_features.csv`
- `outputs/icdm2026/downstream_classification/classification_summary.csv`
- `outputs/icdm2026/downstream_classification/classification_predictions.csv`
- `outputs/icdm2026/downstream_classification/regression_summary.csv`
- `outputs/icdm2026/downstream_classification/regression_predictions.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.png`

## Current Reading

The generated FA results should not be claimed as a strong four-class classifier yet. The useful signal is disease-stage utility:

- `Fidelity_Flow` gives the best generated-FA result for `CN+SCD vs MCI+AD` after fold-internal feature selection (`Macro-F1=0.559`), above T1-only (`0.526`) and the blurry Stage1 baseline (`0.491`).
- `T1_PLUS_STAGE1_LPIPS_GAN` is useful for `CN vs MCI+AD` (`Macro-F1=0.611`), approaching the real FA upper bound (`FA_GT=0.683`, `T1_PLUS_FA_GT=0.624`).
- `MMSE` regression is currently not a positive generated-FA result. T1-only and real FA are stronger than generated FA, so MMSE should be treated as a negative/secondary analysis rather than a main claim.

This supports the current paper story:

1. Conservative posterior-mean FA gives stable fidelity but weak visual detail.
2. Sharp Stage1 restores high-frequency structure.
3. Frequency-preserving Stage2 improves medical consistency while keeping sharpness.
4. Downstream utility is evaluated at subject level to connect generation quality with disease-stage mining.
