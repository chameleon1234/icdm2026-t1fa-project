# Multiobjective Method Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ranking tool that combines image synthesis metrics and downstream clinical utility for private and ADNI datasets.

**Architecture:** Add one standalone script that loads existing metric summaries and downstream CSVs, canonicalizes method names, normalizes metrics by direction, computes image/downstream/overall scores, and marks Pareto-front methods. Keep the ranking logic as importable functions with focused unit tests.

**Tech Stack:** Python, pandas, pytest, JSON/CSV artifacts already generated under `outputs/icdm2026`.

---

### Task 1: Ranking Logic Tests

**Files:**
- Create: `tests/test_build_multiobjective_method_ranking.py`

- [ ] **Step 1: Add tests for metric normalization, Pareto selection, and scoring**

The tests should create tiny in-memory data frames and assert:
- higher-is-better and lower-is-better normalization both map best to `1.0`
- dominated methods are excluded from the Pareto front
- methods with strong PSNR/SSIM but weak downstream and methods with weak PSNR/SSIM but strong downstream are ranked below balanced methods when using the default weights

- [ ] **Step 2: Run tests and confirm they fail because the script does not exist**

Run: `pytest tests/test_build_multiobjective_method_ranking.py -q`

Expected: import failure for `scripts.build_multiobjective_method_ranking`.

### Task 2: Ranking Script

**Files:**
- Create: `scripts/build_multiobjective_method_ranking.py`

- [ ] **Step 1: Implement importable helpers**

Implement:
- `canonical_method_name(method: str) -> str`
- `normalize_series(series, higher_is_better=True)`
- `compute_pareto_front(df, metric_specs)`
- `score_methods(df, image_weight=0.65, downstream_weight=0.35)`

- [ ] **Step 2: Implement CLI artifact loading**

Support:
- image summaries from `outputs/icdm2026/metrics/*_summary.json`
- downstream summaries from `outputs/icdm2026/favorable_downstream_protocol_sweep/summary/dataset_method_summary.csv`
- optional extra downstream summary CSVs passed by `--downstream_csv`

- [ ] **Step 3: Write outputs**

Write:
- `dataset_method_ranking.csv`
- `cross_dataset_ranking.csv`
- `pareto_front.csv`
- `multiobjective_ranking_cn.md`
- `multiobjective_ranking.md`

### Task 3: Verify and Commit

**Files:**
- Test: `tests/test_build_multiobjective_method_ranking.py`
- Script: `scripts/build_multiobjective_method_ranking.py`
- Output docs/tables under `outputs/icdm2026/tables/multiobjective_method_ranking`

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/test_build_multiobjective_method_ranking.py -q`

- [ ] **Step 2: Run the ranking script**

Run: `python scripts/build_multiobjective_method_ranking.py --output_dir outputs/icdm2026/tables/multiobjective_method_ranking`

- [ ] **Step 3: Inspect top-ranked methods and Pareto front**

Confirm the outputs include private and ADNI datasets, plus cross-dataset aggregate rows.

- [ ] **Step 4: Commit and push only the new ranking code/docs**

Run:
`git add docs/superpowers/plans/2026-06-18-multiobjective-method-ranking.md scripts/build_multiobjective_method_ranking.py tests/test_build_multiobjective_method_ranking.py`
`git commit -m "Add multiobjective method ranking"`
`git push`
