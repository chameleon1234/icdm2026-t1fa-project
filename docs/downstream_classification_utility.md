# Downstream Disease Utility Evaluation

## Purpose

This experiment tests whether generated FA images preserve disease-relevant information for ICDM-style data mining, beyond image fidelity scores alone. It is intentionally subject-level: all slices from one subject are aggregated into one feature vector before classification, so slice-level leakage is avoided.

## Current Methods

- `T1_ONLY`: input T1 reference baseline.
- `Stage1_Baseline`: conservative Stage1 PMRF-style prediction.
- `Stage1_LPIPS_GAN`: sharp Stage1 prediction with high-frequency preservation.
- `Fidelity_Flow`: frequency-preserving Stage2 flow corrector.
- `Fidelity_Direct`: direct low-frequency corrector ablation.
- `FA_GT`: real FA upper bound.

## Tasks

- `four_class`: CN / SCD / MCI / AD.
- `cn_vs_ad`: CN vs AD.
- `cn_vs_mci_ad`: CN vs MCI+AD.
- `cn_scd_vs_mci_ad`: CN+SCD vs MCI+AD.

The last task is the most important disease-stage utility task in the current results, because it asks whether generated FA helps separate earlier/non-impaired subjects from impairment-stage subjects.

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
  --tasks four_class,cn_vs_ad,cn_vs_mci_ad,cn_scd_vs_mci_ad `
  --n_splits 5 `
  --output_root outputs\icdm2026\downstream_classification
```

## Outputs

- `outputs/icdm2026/downstream_classification/subject_features.csv`
- `outputs/icdm2026/downstream_classification/classification_summary.csv`
- `outputs/icdm2026/downstream_classification/classification_predictions.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.csv`
- `outputs/icdm2026/downstream_classification/confusion_matrices/*.png`

## Current Reading

The generated FA results should not be claimed as a strong four-class classifier yet. The useful signal is that sharp/fidelity-preserving generated FA improves the clinically meaningful `CN+SCD vs MCI+AD` utility over the conservative blurry Stage1 baseline, while real FA remains the upper bound.

This supports the current paper story:

1. Conservative posterior-mean FA gives stable fidelity but weak visual detail.
2. Sharp Stage1 restores high-frequency structure.
3. Frequency-preserving Stage2 improves medical consistency while keeping sharpness.
4. Downstream utility is evaluated at subject level to connect generation quality with disease-stage mining.
