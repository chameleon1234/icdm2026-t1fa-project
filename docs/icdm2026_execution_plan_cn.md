# ICDM 2026 T1-to-FA 论文实施计划

> **给 agentic worker：** 执行本计划时应使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，并按 checkbox 逐项推进。

**目标：** 建立一套可复现的 ICDM Applied Track 论文管线，用医学保真指标评价 T1-to-FA 合成，并用阿尔兹海默症分期验证合成 FA 的下游价值。

**架构：** 依次建立 subject-label 数据层、统一图像指标层、PM-DIRF 预测导出层、subject-level 下游分类层、论文表格和图生成层。保留已有训练脚本可用，同时把可复用逻辑沉淀到 `src/` 和 `scripts/`。

**技术栈：** Python、PyTorch、TorchMetrics、pandas、scikit-learn、NumPy、OpenCV、nibabel、matplotlib、seaborn、现有 `pmrf_t1fa` 模块和处理后的 PNG 切片。

---

## 文件结构

计划创建或修改：

- `docs/icdm2026_progress_log.md`
- `docs/icdm2026_progress_log_cn.md`
- `configs/icdm2026.yaml`
- `src/data/subject_index.py`
- `src/data/t1fa_subject_dataset.py`
- `src/eval/image_metrics.py`
- `src/eval/subject_aggregation.py`
- `src/eval/downstream_classification.py`
- `scripts/build_subject_index.py`
- `scripts/evaluate_method_folder.py`
- `scripts/export_pm_dirf_predictions.py`
- `scripts/run_downstream_classification.py`
- `scripts/make_icdm2026_tables.py`
- `scripts/make_icdm2026_figures.py`
- `src/datasets.py`
- `pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py`
- `main.tex`

## Task 1：项目配置与进度日志

创建 `configs/icdm2026.yaml`，固定数据路径、标签映射、评价指标和输出目录。

创建中英文进度日志，明确：

- 1 = CN
- 2 = SCD
- 3 = MCI
- 4 = AD

验证命令：

```powershell
python -c "import yaml; print(yaml.safe_load(open('configs/icdm2026.yaml', encoding='utf-8'))['project']['label_mapping'])"
```

期望包含：

```text
{1: 'CN', 2: 'SCD', 3: 'MCI', 4: 'AD'}
```

## Task 2：Subject Label Index

实现 `src/data/subject_index.py`：

- 读取 `data/data_information.xlsx` 的 `re_order` sheet。
- 将 `Sub001` 标准化为 `sub-001`。
- 与 `data/processed/dataset_splits.json` 合并。
- 输出每个受试者一行。
- 校验标签映射、总人数和 train/val/test 数量。

实现 CLI：

```powershell
python scripts/build_subject_index.py --config configs/icdm2026.yaml
```

期望输出：

```text
Saved subject index to outputs/icdm2026/subject_index.csv
Subjects: 248
Splits: train=173, val=37, test=38
Groups: CN=92, SCD=56, MCI=70, AD=30
```

## Task 3：Metadata-Aware Dataset

实现 `T1FASubjectSliceDataset`，每个样本返回：

- `t1_slice`
- `fa_slice`
- `fname`
- `subject_id`
- `slice_id`
- `group_id`
- `group_name`
- `age`
- `gender`
- `edu`
- `MMSE`
- `split`

图像读取和归一化保持与原 `T1FADataset` 一致，即 `[0,255]` PNG 转为 `[-1,1]` tensor。

验证命令：

```powershell
python -c "from src.data.subject_index import load_subject_index; from src.data.t1fa_subject_dataset import T1FASubjectSliceDataset; df=load_subject_index('data/data_information.xlsx','data/processed/dataset_splits.json'); ds=T1FASubjectSliceDataset('data/processed/test/t1_slices','data/processed/test/fa_slices',df); x=ds[0]; print(len(ds), x['t1_slice'].shape, x['subject_id'], x['group_name'], x['split'])"
```

期望包含：

```text
1900 torch.Size([3, 224, 224]) sub-
test
```

## Task 4：统一图像指标

后续实现 `src/eval/image_metrics.py` 和 `scripts/evaluate_method_folder.py`，统一计算：

- PSNR
- SSIM
- MSE
- MAE
- Brain-masked MAE
- WM-masked MAE
- Gradient error
- ROI-CCC
- WM histogram Wasserstein

## Task 5：PM-DIRF 预测导出

后续实现 `scripts/export_pm_dirf_predictions.py`，从 Stage 1/Stage 2 checkpoint 导出测试集合成 FA，并保持文件名与测试集 FA 一致。

## Task 6：已有方法重评估

将 DIRF、UNet、Restormer、DDIM、DBM、CycleGAN、Pix2Pix 等方法的预测结果统一导出到：

```text
outputs/icdm2026/predictions/<METHOD_NAME>/
```

然后用统一指标脚本评估。

## Task 7：下游分类

建立 subject-level 特征聚合和分类评估，比较：

- T1 only
- Real FA only
- Synthetic FA only
- T1 + synthetic FA
- T1 + real FA

任务：

- four_class
- cn_vs_ad
- cn_vs_mci_ad
- cn_scd_vs_mci_ad

## Task 8：消融实验

核心消融：

- Stage 1 posterior-mean only
- Stage 1 + Stage 2 PM-DIRF
- 去掉 detail/gradient/HF loss
- 去掉 coarse conditioning
- 1/2/4/8 step 推理
- 可选 disease-aware auxiliary classification loss

## Task 9：表格和图

生成：

- 主合成指标表
- 下游分类表
- 消融表
- pipeline 图
- qualitative comparison 图
- ROI case analysis 图
- downstream confusion matrix 图
- ablation curves 图

## Task 10：论文重写

重写 `main.tex`：

- 标题和摘要改为 missing diffusion biomarker mining + AD staging。
- Introduction 强调 ICDM 应用数据挖掘定位。
- 主结果顺序改为合成指标、医学 ROI、下游分类、消融、案例、效率。
- FID/KID 不再作为主结论。

## Task 11：最终可复现检查

记录环境、最终文件和验收标准。验收标准包括：

- subject index 有 248 行。
- test split 有 38 例和 1900 张切片。
- 所有主方法都有 PSNR/SSIM/MSE/MAE。
- PM-DIRF 至少完成 Stage 1 与一个 Stage 2 评估。
- 至少完成一个下游分类表。
- 至少完成一个 ROI/case-analysis 图。
- `main.tex` 不再把 FID 作为主论据。

