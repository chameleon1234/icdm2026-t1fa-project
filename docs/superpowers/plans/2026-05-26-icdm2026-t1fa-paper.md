# ICDM 2026 T1-to-FA Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible ICDM Applied Track paper pipeline that evaluates T1-to-FA synthesis with medical fidelity metrics and validates synthesized FA through Alzheimer's staging.

**Architecture:** Add a subject-label data layer, a unified metric/evaluation layer, a PM-DIRF training and export layer, a downstream subject-level classification layer, and a figure/table generation layer. Keep existing training scripts usable while migrating reusable pieces into focused modules under `src/` and `scripts/`.

**Tech Stack:** Python, PyTorch, TorchMetrics, pandas, scikit-learn, NumPy, OpenCV, nibabel, matplotlib, seaborn, existing `pmrf_t1fa` modules, existing processed PNG slices.

---

## File Structure

Create or modify these files:

- Create: `docs/icdm2026_progress_log.md`
- Create: `configs/icdm2026.yaml`
- Create: `src/data/subject_index.py`
- Create: `src/data/t1fa_subject_dataset.py`
- Create: `src/eval/image_metrics.py`
- Create: `src/eval/subject_aggregation.py`
- Create: `src/eval/downstream_classification.py`
- Create: `scripts/build_subject_index.py`
- Create: `scripts/evaluate_method_folder.py`
- Create: `scripts/export_pm_dirf_predictions.py`
- Create: `scripts/run_downstream_classification.py`
- Create: `scripts/make_icdm2026_tables.py`
- Create: `scripts/make_icdm2026_figures.py`
- Modify: `src/datasets.py`
- Modify: `pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py`
- Modify: `main.tex`

Because the current workspace is not a git repository, every task should append its completion evidence to `docs/icdm2026_progress_log.md` instead of committing. If a git repository is initialized during the project, convert each completed task into a normal commit.

## Task 1: Project Configuration And Progress Log

**Files:**

- Create: `configs/icdm2026.yaml`
- Create: `docs/icdm2026_progress_log.md`

- [ ] **Step 1: Create config file**

Add `configs/icdm2026.yaml` with this content:

```yaml
project:
  name: icdm2026_t1fa_ad_staging
  target_conference: ICDM 2026 Applied Data Science Track
  label_mapping:
    1: CN
    2: SCD
    3: MCI
    4: AD

data:
  excel_path: data/data_information.xlsx
  excel_sheet: re_order
  split_json: data/processed/dataset_splits.json
  processed_root: data/processed
  raw_root: data/ZHU_T1_and_FA_space-MNI152NLin6Asym_res-02
  image_size: 224
  slices_per_subject: 50

evaluation:
  main_metrics: [PSNR, SSIM, MSE, MAE]
  medical_metrics:
    - Brain_Masked_MAE
    - WM_Masked_MAE
    - Gradient_Error
    - ROI_CCC
    - WM_Hist_Wasserstein
  classification_tasks:
    - four_class
    - cn_vs_ad
    - cn_vs_mci_ad
    - cn_scd_vs_mci_ad

outputs:
  root: outputs/icdm2026
  subject_index: outputs/icdm2026/subject_index.csv
  metrics_root: outputs/icdm2026/metrics
  predictions_root: outputs/icdm2026/predictions
  tables_root: outputs/icdm2026/tables
  figures_root: outputs/icdm2026/figures
```

- [ ] **Step 2: Create progress log**

Create `docs/icdm2026_progress_log.md`:

```markdown
# ICDM 2026 Progress Log

## Label Mapping

Confirmed on 2026-05-26:

- 1 = CN
- 2 = SCD
- 3 = MCI
- 4 = AD

## Completed Tasks

No implementation tasks completed yet.
```

- [ ] **Step 3: Verify YAML parses**

Run:

```powershell
python -c "import yaml; print(yaml.safe_load(open('configs/icdm2026.yaml', encoding='utf-8'))['project']['label_mapping'])"
```

Expected output contains:

```text
{1: 'CN', 2: 'SCD', 3: 'MCI', 4: 'AD'}
```

## Task 2: Subject Label Index

**Files:**

- Create: `src/data/subject_index.py`
- Create: `scripts/build_subject_index.py`
- Test: command-line assertions below

- [ ] **Step 1: Implement `src/data/subject_index.py`**

Responsibilities:

- Read `data/data_information.xlsx`, sheet `re_order`.
- Normalize `Sub001` to `sub-001`.
- Merge with `data/processed/dataset_splits.json`.
- Emit one row per subject.
- Validate group mapping and subject count.

Core API:

```python
def load_subject_index(excel_path: str, split_json: str) -> pandas.DataFrame:
    ...
```

Returned columns:

```text
subject_id, group_id, group_name, gender, age, edu, MMSE, split
```

- [ ] **Step 2: Implement CLI `scripts/build_subject_index.py`**

Command:

```powershell
python scripts/build_subject_index.py --config configs/icdm2026.yaml
```

Expected output:

```text
Saved subject index to outputs/icdm2026/subject_index.csv
Subjects: 248
Splits: train=173, val=37, test=38
Groups: CN=92, SCD=56, MCI=70, AD=30
```

- [ ] **Step 3: Verification**

Run:

```powershell
python scripts/build_subject_index.py --config configs/icdm2026.yaml
python -c "import pandas as pd; df=pd.read_csv('outputs/icdm2026/subject_index.csv'); print(df.shape); print(df.groupby('group_name').size().to_dict()); print(df.groupby('split').size().to_dict())"
```

Expected output contains:

```text
(248, 8)
{'AD': 30, 'CN': 92, 'MCI': 70, 'SCD': 56}
{'test': 38, 'train': 173, 'val': 37}
```

## Task 3: Metadata-Aware Dataset

**Files:**

- Create: `src/data/t1fa_subject_dataset.py`
- Modify: `src/datasets.py`

- [ ] **Step 1: Add metadata-aware dataset**

Create `T1FASubjectSliceDataset` that returns:

```python
{
    "t1_slice": t1_tensor,
    "fa_slice": fa_tensor,
    "fname": filename,
    "subject_id": subject_id,
    "slice_id": slice_id,
    "group_id": group_id,
    "group_name": group_name,
    "age": age,
    "gender": gender,
    "edu": edu,
    "MMSE": mmse,
    "split": split,
}
```

Use the same image reading and `[-1, 1]` normalization as `src/datasets.py`.

- [ ] **Step 2: Keep backward compatibility**

Modify `src/datasets.py` only by importing or wrapping the new class. Existing scripts that use `T1FADataset` must still run unchanged.

- [ ] **Step 3: Verification**

Run:

```powershell
python -c "from src.data.subject_index import load_subject_index; from src.data.t1fa_subject_dataset import T1FASubjectSliceDataset; df=load_subject_index('data/data_information.xlsx','data/processed/dataset_splits.json'); ds=T1FASubjectSliceDataset('data/processed/test/t1_slices','data/processed/test/fa_slices',df); x=ds[0]; print(len(ds), x['t1_slice'].shape, x['subject_id'], x['group_name'], x['split'])"
```

Expected output contains:

```text
1900 torch.Size([3, 224, 224]) sub-
test
```

## Task 4: Unified Image Metrics

**Files:**

- Create: `src/eval/image_metrics.py`
- Create: `scripts/evaluate_method_folder.py`

- [ ] **Step 1: Implement metric functions**

Add functions:

```python
compute_psnr(pred_01, target_01) -> float
compute_ssim(pred_01, target_01) -> float
compute_mse(pred_01, target_01) -> float
compute_mae(pred_01, target_01) -> float
build_brain_mask(t1_01, fa_01) -> torch.Tensor
build_wm_mask(fa_01, brain_mask) -> torch.Tensor
masked_mae(pred_01, target_01, mask) -> float
gradient_error(pred_01, target_01, mask) -> float
roi_ccc(pred_roi_values, target_roi_values) -> float
```

Match the existing logic in `pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py` for masks, gradients, histogram distance, and ROI-CCC.

- [ ] **Step 2: Implement folder evaluator**

`scripts/evaluate_method_folder.py` should evaluate saved prediction PNGs with filenames matching test FA filenames:

```powershell
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/DIRF --method DIRF --config configs/icdm2026.yaml
```

Expected files:

```text
outputs/icdm2026/metrics/DIRF_slice_metrics.csv
outputs/icdm2026/metrics/DIRF_subject_metrics.csv
outputs/icdm2026/metrics/DIRF_summary.json
```

- [ ] **Step 3: Verification with ground truth as prediction**

Run:

```powershell
python scripts/evaluate_method_folder.py --pred_dir data/processed/test/fa_slices --method GT_SANITY --config configs/icdm2026.yaml
```

Expected:

```text
PSNR should be very high or infinite-handled
SSIM should be approximately 1.0
MSE should be approximately 0.0
MAE should be approximately 0.0
```

## Task 5: PM-DIRF Prediction Export

**Files:**

- Create: `scripts/export_pm_dirf_predictions.py`
- Modify: `pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py`

- [ ] **Step 1: Add prediction export script**

The script loads:

- Stage 1 checkpoint from `outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt`.
- Stage 2 checkpoint when available.
- Test T1 slices.

It writes generated FA PNGs to:

```text
outputs/icdm2026/predictions/PM_DIRF/
```

Filenames must match the original test filenames, for example:

```text
sub-002_z020.png
```

- [ ] **Step 2: Support Stage 1-only export**

Add argument:

```powershell
--stage stage1
```

This exports posterior-mean coarse predictions and writes to:

```text
outputs/icdm2026/predictions/PM_STAGE1/
```

- [ ] **Step 3: Verification**

Run:

```powershell
python scripts/export_pm_dirf_predictions.py --stage stage1 --config configs/icdm2026.yaml --limit 16
python -c "from pathlib import Path; files=list(Path('outputs/icdm2026/predictions/PM_STAGE1').glob('*.png')); print(len(files)); print(files[0].name if files else 'missing')"
```

Expected:

```text
16
sub-
```

## Task 6: Re-Evaluate Existing Methods

**Files:**

- Use existing scripts:
  - `evaluate_metrics.py`
  - `evaluate_cyclegan_metrics.py`
  - `evaluate_ddim.py`
  - `evaluate_pix2pix.py`
  - `evaluate_restormer.py`
  - `evaluate_baseline_restormer.py`
  - `evaluate_swin.py`
  - `eval_baseline_dbm.py`
- Create summary through `scripts/make_icdm2026_tables.py`

- [ ] **Step 1: Inventory checkpoints**

Create a table in `docs/icdm2026_progress_log.md` listing each method, checkpoint path, output prediction folder, and whether it is ready.

- [ ] **Step 2: Export predictions for every ready method**

Each method should produce a prediction folder under:

```text
outputs/icdm2026/predictions/<METHOD_NAME>/
```

- [ ] **Step 3: Evaluate every prediction folder**

For each method:

```powershell
python scripts/evaluate_method_folder.py --pred_dir outputs/icdm2026/predictions/<METHOD_NAME> --method <METHOD_NAME> --config configs/icdm2026.yaml
```

- [ ] **Step 4: Verification**

Run:

```powershell
python -c "from pathlib import Path; print([p.name for p in Path('outputs/icdm2026/metrics').glob('*_summary.json')])"
```

Expected output includes at minimum:

```text
DIRF_summary.json
PM_STAGE1_summary.json
UNET_summary.json
RESTORMER_summary.json
DDIM_summary.json
DBM_summary.json
```

## Task 7: Downstream Subject-Level Classification

**Files:**

- Create: `src/eval/subject_aggregation.py`
- Create: `src/eval/downstream_classification.py`
- Create: `scripts/run_downstream_classification.py`

- [ ] **Step 1: Subject feature aggregation**

Create subject-level features from FA images:

- Mean FA over brain mask.
- Mean FA over white-matter mask.
- Standard deviation over white-matter mask.
- ROI grid means using 2x3 grid.
- Slice-wise mean and standard deviation aggregated over 50 slices.
- Optional learned embedding from a small CNN only if time allows.

- [ ] **Step 2: Classification models**

Implement classical baselines first:

- Logistic regression.
- Linear SVM.
- Random forest.
- XGBoost only if already installed; otherwise skip.

Inputs:

- T1 only.
- Real FA only.
- Synthetic FA only.
- T1 + synthetic FA.
- T1 + real FA.

- [ ] **Step 3: Tasks**

Run:

```powershell
python scripts/run_downstream_classification.py --config configs/icdm2026.yaml --task four_class --synthetic_method PM_DIRF
python scripts/run_downstream_classification.py --config configs/icdm2026.yaml --task cn_vs_ad --synthetic_method PM_DIRF
python scripts/run_downstream_classification.py --config configs/icdm2026.yaml --task cn_vs_mci_ad --synthetic_method PM_DIRF
python scripts/run_downstream_classification.py --config configs/icdm2026.yaml --task cn_scd_vs_mci_ad --synthetic_method PM_DIRF
```

Expected files:

```text
outputs/icdm2026/metrics/downstream_four_class.csv
outputs/icdm2026/metrics/downstream_cn_vs_ad.csv
outputs/icdm2026/metrics/downstream_cn_vs_mci_ad.csv
outputs/icdm2026/metrics/downstream_cn_scd_vs_mci_ad.csv
```

- [ ] **Step 4: Verification**

Run:

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/icdm2026/metrics/downstream_four_class.csv'); print(df[['input_type','balanced_accuracy','macro_f1']].to_string(index=False))"
```

Expected output has rows for:

```text
T1
Real_FA
Synthetic_FA
T1_Synthetic_FA
T1_Real_FA
```

## Task 8: Ablation Experiments

**Files:**

- Use `pmrf_t1fa/train_pmrf_t1fa_stage1.py`
- Use `pmrf_t1fa/train_pmrf_t1fa_stage2.py`
- Create output tables via `scripts/make_icdm2026_tables.py`

- [ ] **Step 1: Stage 1 posterior-mean baseline**

Train or reuse:

```powershell
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage1.py --run_name pmrf_t1fa_stage1_paired_residual --stage1_prediction_mode residual
```

- [ ] **Step 2: Stage 2 default PM-DIRF**

Train:

```powershell
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --run_name pm_dirf_default --eval_steps 1 --condition_on_coarse
```

- [ ] **Step 3: No-detail-loss ablation**

Train:

```powershell
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --run_name pm_dirf_no_detail --detail_weight 0 --grad_weight 0 --hf_weight 0 --eval_steps 1 --condition_on_coarse
```

- [ ] **Step 4: No-coarse-condition ablation**

Train:

```powershell
accelerate launch pmrf_t1fa/train_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --run_name pm_dirf_no_condition --disable_condition_on_coarse --eval_steps 1
```

- [ ] **Step 5: Step-count ablation**

Evaluate trained default checkpoint with:

```powershell
python pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_default/checkpoints/best_stage2.pt --eval_steps 1 --output_dir outputs/icdm2026/metrics/pm_dirf_steps_1
python pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_default/checkpoints/best_stage2.pt --eval_steps 2 --output_dir outputs/icdm2026/metrics/pm_dirf_steps_2
python pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_default/checkpoints/best_stage2.pt --eval_steps 4 --output_dir outputs/icdm2026/metrics/pm_dirf_steps_4
python pmrf_t1fa/evaluate_pmrf_t1fa_stage2.py --stage1_ckpt outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt --stage2_ckpt outputs/pm_dirf_default/checkpoints/best_stage2.pt --eval_steps 8 --output_dir outputs/icdm2026/metrics/pm_dirf_steps_8
```

## Task 9: Tables And Figures

**Files:**

- Create: `scripts/make_icdm2026_tables.py`
- Create: `scripts/make_icdm2026_figures.py`

- [ ] **Step 1: Main synthesis table**

Generate:

```text
outputs/icdm2026/tables/table1_synthesis_metrics.csv
outputs/icdm2026/tables/table1_synthesis_metrics.tex
```

Columns:

```text
Method, NFE, PSNR, SSIM, MSE, MAE, WM-MAE, ROI-CCC, Latency
```

- [ ] **Step 2: Downstream classification table**

Generate:

```text
outputs/icdm2026/tables/table2_downstream.csv
outputs/icdm2026/tables/table2_downstream.tex
```

Columns:

```text
Task, Input, Balanced Accuracy, Macro-F1, AUROC
```

- [ ] **Step 3: Ablation table**

Generate:

```text
outputs/icdm2026/tables/table3_ablation.csv
outputs/icdm2026/tables/table3_ablation.tex
```

- [ ] **Step 4: Figures**

Generate:

```text
outputs/icdm2026/figures/fig1_pipeline.png
outputs/icdm2026/figures/fig2_qualitative_comparison.png
outputs/icdm2026/figures/fig3_roi_case_analysis.png
outputs/icdm2026/figures/fig4_downstream_confusion.png
outputs/icdm2026/figures/fig5_ablation_curves.png
```

## Task 10: Paper Rewrite

**Files:**

- Modify: `main.tex`

- [ ] **Step 1: Rewrite title and abstract**

Replace the current synthesis-only framing with missing biomarker mining and downstream AD staging.

- [ ] **Step 2: Rewrite introduction**

Make the ICDM motivation explicit:

- Missing modality mining.
- Quantitative biomarker synthesis.
- Disease staging utility.
- Deterministic reproducibility.

- [ ] **Step 3: Rewrite experiments**

Main results order:

1. Synthesis fidelity.
2. Medical ROI preservation.
3. Downstream classification.
4. Ablation.
5. Case analysis.
6. Efficiency and determinism.

- [ ] **Step 4: Remove FID from main claim**

FID/KID should not appear in abstract, primary table, or main conclusion. If retained, put them in a supplementary paragraph with a caveat.

- [ ] **Step 5: Verification**

Compile:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Expected:

```text
main.pdf generated without unresolved references in the final pass
```

## Task 11: Final Reproducibility Check

**Files:**

- Modify: `docs/icdm2026_progress_log.md`

- [ ] **Step 1: Record environment**

Run:

```powershell
python --version
python -c "import torch, pandas, sklearn; print(torch.__version__); print(pandas.__version__); print(sklearn.__version__)"
```

- [ ] **Step 2: Record final artifacts**

Append to `docs/icdm2026_progress_log.md`:

```markdown
## Final Artifacts

- Main paper: `main.pdf`
- Project book: `docs/icdm2026_project_book.md`
- Execution plan: `docs/icdm2026_execution_plan.md`
- Subject index: `outputs/icdm2026/subject_index.csv`
- Metrics: `outputs/icdm2026/metrics`
- Tables: `outputs/icdm2026/tables`
- Figures: `outputs/icdm2026/figures`
```

- [ ] **Step 3: Acceptance criteria**

The project is ready for paper writing freeze when:

- Subject index has 248 rows.
- Test split has 38 subjects and 1900 slices.
- All main methods have PSNR/SSIM/MSE/MAE.
- PM-DIRF has at least Stage 1 and one Stage 2 evaluation.
- At least one downstream classification table is complete.
- At least one ROI/case-analysis figure is complete.
- `main.tex` no longer uses FID as a main claim.
