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

