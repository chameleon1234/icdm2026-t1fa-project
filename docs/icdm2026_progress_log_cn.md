# ICDM 2026 进度日志

## 标签映射

已于 2026-05-26 确认：

- 1 = CN
- 2 = SCD
- 3 = MCI
- 4 = AD

## Git 与仓库

- 远端仓库：`chameleon1234/icdm2026-t1fa-project`
- 本地工作分支：`work/icdm-task1-3`
- 大体积数据、论文 PDF、模型权重、生成图表和输出目录均通过 `.gitignore` 排除。

## 已完成任务

- 2026-05-26：创建项目书和实施计划。
- 2026-05-26：Task 1 完成。新增 `configs/icdm2026.yaml` 和中英文进度日志。
- 2026-05-26：Task 2 完成。新增 subject index 加载器和 `scripts/build_subject_index.py`；已验证 248 例，CN=92，SCD=56，MCI=70，AD=30，train=173，val=37，test=38。
- 2026-05-26：Task 3 完成。新增 `T1FASubjectSliceDataset`；已验证测试集长度 1900，样本包含图像张量和受试者元数据。

## 验证记录

2026-05-26 Task 1-3：

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
