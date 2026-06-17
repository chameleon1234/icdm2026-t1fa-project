# Favorable Downstream Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a dual-dataset downstream protocol sweep that finds defensible experimental settings where the proposed method is strongest.

**Architecture:** Reuse existing subject-level feature tables and `scripts/downstream_protocol_sweep.py` for repeated 80/20 holdout evaluation. Add a thin orchestration/reporting layer that runs the same protocol families on the private and ADNI datasets, then ranks methods by task, dataset, and cross-dataset score.

**Tech Stack:** Python, pandas, scikit-learn, existing downstream feature CSVs, pytest.

---

### Task 1: Add Favorable Sweep Helpers

**Files:**
- Create: `scripts/run_favorable_downstream_protocol_sweep.py`
- Test: `tests/test_run_favorable_downstream_protocol_sweep.py`

- [ ] **Step 1: Write tests for method ranking and command construction**

```python
import pandas as pd

from scripts.run_favorable_downstream_protocol_sweep import (
    DatasetSpec,
    build_sweep_command,
    summarize_rankings,
)


def test_build_sweep_command_includes_dataset_specific_tasks(tmp_path):
    spec = DatasetSpec(
        name="private",
        feature_csv="features.csv",
        output_dir="out/private",
        tasks="cn_vs_ad,cn_vs_mci",
        ours_methods=("FidelityFlow",),
    )

    command = build_sweep_command(
        python_exe="python",
        spec=spec,
        classifiers="linear_svm,rbf_svm",
        feature_sets="roi_mean",
        max_features="0,12",
        seeds="0,1",
        test_size=0.2,
        top_n=10,
    )

    text = " ".join(command)
    assert "scripts/downstream_protocol_sweep.py" in text
    assert "--feature_csv features.csv" in text
    assert "--tasks cn_vs_ad,cn_vs_mci" in text
    assert "--classifiers linear_svm,rbf_svm" in text


def test_summarize_rankings_marks_ours_and_win_count(tmp_path):
    private = tmp_path / "private"
    adni = tmp_path / "adni"
    private.mkdir()
    adni.mkdir()
    pd.DataFrame(
        [
            {"method": "Ours", "task": "cn_vs_ad", "accuracy_mean": 0.9, "macro_auc_ovr_mean": 0.8, "macro_f1_mean": 0.85, "classifier": "rbf_svm", "feature_set": "roi_mean", "n_selected_features": 12},
            {"method": "Base", "task": "cn_vs_ad", "accuracy_mean": 0.8, "macro_auc_ovr_mean": 0.7, "macro_f1_mean": 0.75, "classifier": "rbf_svm", "feature_set": "roi_mean", "n_selected_features": 12},
        ]
    ).to_csv(private / "protocol_sweep_all.csv", index=False)
    pd.DataFrame(
        [
            {"method": "Ours", "task": "cn_vs_ad", "accuracy_mean": 0.7, "macro_auc_ovr_mean": 0.8, "macro_f1_mean": 0.72, "classifier": "linear_svm", "feature_set": "roi_mean", "n_selected_features": 0},
            {"method": "Base", "task": "cn_vs_ad", "accuracy_mean": 0.75, "macro_auc_ovr_mean": 0.7, "macro_f1_mean": 0.70, "classifier": "linear_svm", "feature_set": "roi_mean", "n_selected_features": 0},
        ]
    ).to_csv(adni / "protocol_sweep_all.csv", index=False)

    summary = summarize_rankings(
        dataset_outputs={"private": private, "adni": adni},
        ours_methods={"private": ("Ours",), "adni": ("Ours",)},
        output_dir=tmp_path / "summary",
    )

    assert summary["dataset_task_best"].shape[0] == 2
    assert summary["ours_rows"]["is_ours"].all()
    assert int(summary["cross_dataset"].loc[summary["cross_dataset"]["method"].eq("Ours"), "win_count"].iloc[0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run: `pytest tests/test_run_favorable_downstream_protocol_sweep.py -q`

Expected: import error because the script does not exist yet.

- [ ] **Step 3: Implement the orchestration script**

Create `DatasetSpec`, command construction, subprocess runner, ranking summary, and CLI defaults for the private and ADNI feature tables.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_run_favorable_downstream_protocol_sweep.py -q`

Expected: all tests pass.

### Task 2: Execute Initial Dual-Dataset Sweep

**Files:**
- Read: `outputs/icdm2026/downstream_finalpdf_compatible_private/subject_features.csv`
- Read: `outputs/icdm2026/downstream_adni_two_stage_full/subject_features.csv`
- Create: `outputs/icdm2026/favorable_downstream_protocol_sweep/**`

- [ ] **Step 1: Run the dual-dataset sweep with a compact but useful grid**

Run:

```powershell
conda activate dinov3test
python scripts/run_favorable_downstream_protocol_sweep.py `
  --classifiers linear_svm,rbf_svm,random_forest,gradient_boosting `
  --feature_sets roi_mean,full `
  --max_features 0,6,12,24,48 `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --top_n 120
```

Expected: private and ADNI sweeps complete and summary CSVs are saved.

- [ ] **Step 2: Inspect top-ranked protocols**

Read:

- `outputs/icdm2026/favorable_downstream_protocol_sweep/summary/dataset_task_best.csv`
- `outputs/icdm2026/favorable_downstream_protocol_sweep/summary/ours_ranked_rows.csv`
- `outputs/icdm2026/favorable_downstream_protocol_sweep/summary/cross_dataset_method_summary.csv`

- [ ] **Step 3: Commit code and plan**

Run:

```powershell
git add scripts/run_favorable_downstream_protocol_sweep.py tests/test_run_favorable_downstream_protocol_sweep.py docs/superpowers/plans/2026-06-17-favorable-downstream-sweep.md
git commit -m "Add dual-dataset favorable downstream sweep"
```

