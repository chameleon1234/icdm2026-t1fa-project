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

- 2026-05-26：创建项目书和执行计划。
- 2026-05-26：Task 1 完成。新增 `configs/icdm2026.yaml` 和中英文进度日志。
- 2026-05-26：Task 2 完成。新增 subject index 加载器和 `scripts/build_subject_index.py`；已验证 248 例，CN=92，SCD=56，MCI=70，AD=30，train=173，val=37，test=38。
- 2026-05-26：Task 3 完成。新增 `T1FASubjectSliceDataset`；已验证测试集长度为 1900，样本包含图像张量和受试者元数据。
- 2026-05-26：Task 4 完成。新增统一图像指标模块和 `scripts/evaluate_method_folder.py`；已通过真实 FA 作为预测结果的 GT sanity run 验证公共测试集评估流程。
- 2026-05-26：Task 5 的 PM_STAGE1 部分完成。新增 `scripts/export_pm_dirf_predictions.py`；已按测试集文件名导出 1900 张 PM_STAGE1 PNG；评估脚本已扩展为生成固定顺序的可视化面板。PM_DIRF 导出功能已实现，但还需要可用的 Stage 2 checkpoint。

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

2026-05-26 Task 4：

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

2026-05-26 Task 5：

```text
python scripts/export_pm_dirf_predictions.py --stage stage1 --config configs/icdm2026.yaml --batch_size 8 --device cpu
PM_STAGE1: exported=1900 output_dir=outputs\icdm2026\predictions\PM_STAGE1
Saved manifest to: outputs\icdm2026\predictions\PM_STAGE1\export_manifest.csv
```

```text
python scripts/evaluate_method_folder.py --pred_dir data/processed/test/fa_slices --method GT_SANITY --config configs/icdm2026.yaml --visualize_count 8 --reset_visualize_manifest
GT_SANITY: n_slices=1900 n_subjects=38 PSNR=inf SSIM=1.0000 MSE=0.000000 MAE=0.000000
```

```text
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/PM_STAGE1 --method PM_STAGE1 --config configs/icdm2026.yaml --visualize_count 8
PM_STAGE1: n_slices=1900 n_subjects=38 PSNR=27.972 SSIM=0.9034 MSE=0.001696 MAE=0.018168
```

```text
PM_STAGE1 summary check:
n_slices=1900, n_subjects=38, PSNR_mean=27.972420835201195, SSIM_mean=0.9033873518517143, MSE_mean=0.0016956827229679268, MAE_mean=0.01816821260696375, ROI_CCC=0.799743333006903
PM_STAGE1 subject table:
38 subjects, group counts {'AD': 4, 'CN': 17, 'MCI': 11, 'SCD': 6}
Stage 2 checkpoint check:
outputs/pm_dirf_default/checkpoints/best_stage2.pt = missing
outputs/pmrf_t1fa_stage2_b/checkpoints/best_stage2.pt = missing
```

所有方法共用的可视化 manifest：

```text
sub-002_z020.png CN
sub-024_z041.png CN
sub-056_z063.png CN
sub-072_z034.png CN
sub-140_z055.png SCD
sub-167_z026.png MCI
sub-190_z048.png MCI
sub-246_z069.png AD
```
