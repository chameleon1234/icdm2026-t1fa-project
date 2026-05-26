# ICDM 2026 Progress Log

## Label Mapping

Confirmed on 2026-05-26:

- 1 = CN
- 2 = SCD
- 3 = MCI
- 4 = AD

## Git And Repository

- Remote repository: `chameleon1234/icdm2026-t1fa-project`
- Local working branch: `work/icdm-task1-3`
- Large data, papers, checkpoints, generated plots, and outputs are intentionally ignored.

## Completed Tasks

- 2026-05-26: Created project book and execution plan.
- 2026-05-26: Task 1 completed. Added `configs/icdm2026.yaml` and progress logs.
- 2026-05-26: Task 2 completed. Added subject index loader and `scripts/build_subject_index.py`; verified 248 subjects, CN=92, SCD=56, MCI=70, AD=30, train=173, val=37, test=38.
- 2026-05-26: Task 3 completed. Added `T1FASubjectSliceDataset`; verified test dataset length is 1900 and samples include image tensors plus subject metadata.
- 2026-05-26: Task 4 completed. Added unified image metrics and `scripts/evaluate_method_folder.py`; verified common test-set evaluation with GT-as-prediction sanity run.

## Verification Evidence

2026-05-26 Task 1-3:

```text
pytest tests/test_build_subject_index_cli.py tests/test_subject_index.py tests/test_t1fa_subject_dataset.py -q
4 passed
```

```text
python scripts/build_subject_index.py --config configs/icdm2026.yaml
Saved subject index to outputs/icdm2026/subject_index.csv
Subjects: 248
Splits: train=173, val=37, test=38
Groups: CN=92, SCD=56, MCI=70, AD=30
Mapping: 1=CN, 2=SCD, 3=MCI, 4=AD
```

2026-05-26 Task 4:

```text
pytest tests/test_build_subject_index_cli.py tests/test_subject_index.py tests/test_t1fa_subject_dataset.py tests/test_image_metrics.py tests/test_evaluate_method_folder_cli.py -q
7 passed
```

```text
python scripts/evaluate_method_folder.py --pred_dir data/processed/test/fa_slices --method GT_SANITY --config configs/icdm2026.yaml
GT_SANITY: n_slices=1900 n_subjects=38 PSNR=inf SSIM=1.0000 MSE=0.000000 MAE=0.000000
Saved slice metrics to: outputs\icdm2026\metrics\GT_SANITY_slice_metrics.csv
Saved subject metrics to: outputs\icdm2026\metrics\GT_SANITY_subject_metrics.csv
Saved summary to: outputs\icdm2026\metrics\GT_SANITY_summary.json
```

```text
GT_SANITY summary check:
n_slices=1900, n_subjects=38, PSNR_mean=inf, SSIM_mean=1.0, MSE_mean=0.0, MAE_mean=0.0, ROI_CCC=1.0
GT_SANITY subject table:
38 subjects, group counts {'AD': 4, 'CN': 17, 'MCI': 11, 'SCD': 6}
```
