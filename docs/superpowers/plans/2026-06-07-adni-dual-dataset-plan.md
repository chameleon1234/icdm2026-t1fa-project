# ADNI Dual-Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable ADNI public-dataset preparation path for dual-dataset T1-to-FA generation and downstream disease-utility validation.

**Architecture:** Add a focused ADNI manifest/split script first, then use its outputs as the single source of truth for NIfTI-to-slice preprocessing and future training. Keep generation splits subject-level and keep downstream labels normalized separately from raw labels.

**Tech Stack:** Python, pandas, libarchive, scikit-learn, pytest, existing `configs/icdm2026.yaml` and evaluation scripts.

---

### Task 1: ADNI Manifest And Split

**Files:**
- Create: `scripts/build_adni_manifest.py`
- Create: `tests/test_build_adni_manifest.py`
- Output: `outputs/icdm2026/adni_subject_manifest.csv`
- Output: `outputs/icdm2026/adni_label_summary.csv`
- Output: `outputs/icdm2026/adni_split_subjects.json`

- [ ] **Step 1: Write tests**

Write tests for subject ID normalization, label mapping, paired-subject construction, and subject-level split leakage checks.

- [ ] **Step 2: Implement script**

Implement archive scanning, label merge, normalized groups, split assignment, and summary writing.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_build_adni_manifest.py -q`

- [ ] **Step 4: Generate ADNI outputs**

Run: `python scripts/build_adni_manifest.py --archive data/ADNI_data.7z --labels data/subject_group_cleaned_filtered.csv --output_root outputs/icdm2026`

- [ ] **Step 5: Commit**

Commit the script, tests, and design documents.

### Task 2: ADNI NIfTI-To-Slice Preprocessing

**Files:**
- Create: `scripts/preprocess_adni_slices.py`
- Create: `tests/test_preprocess_adni_slices.py`
- Output: `data/adni_processed/{train,val,test}/{t1_slices,fa_slices}`

- [ ] **Step 1: Write preprocessing tests**

Test slice selection, T1 normalization, FA clipping, fixed output names, and subject split preservation.

- [ ] **Step 2: Implement preprocessing**

Read paired NIfTI files from extracted ADNI data, generate axial 5-slice-compatible PNG slices, and write a slice manifest.

- [ ] **Step 3: Smoke preprocess**

Run a small subject-limited preprocessing smoke test before full conversion.

### Task 3: ADNI Baseline Evaluation

**Files:**
- Create: `configs/icdm2026_adni_methods.yaml`
- Reuse: existing training/export/evaluation scripts where possible

- [ ] **Step 1: Train or export primary methods**

Run Stage1, Fidelity Flow, FREQ, U-Net, Pix2Pix, CycleGAN, and DDIM on ADNI splits.

- [ ] **Step 2: Run image metrics**

Use the unified `evaluate_method_folder.py` metric path.

- [ ] **Step 3: Run downstream utility**

Evaluate T1 only, synthetic FA only, T1 + synthetic FA, real FA only, and T1 + real FA.

### Task 4: Dual-Dataset Paper Tables

**Files:**
- Create: `scripts/build_dual_dataset_tables.py`
- Output: `outputs/icdm2026/tables/dual_dataset_*`

- [ ] **Step 1: Build dual-dataset table builder**

Merge private and ADNI results into comparable method rows.

- [ ] **Step 2: Generate final tables**

Generate image metrics, downstream utility, and ablation tables for both datasets.

- [ ] **Step 3: Interpret selection**

Select the final method based on balanced performance across private and ADNI datasets.
