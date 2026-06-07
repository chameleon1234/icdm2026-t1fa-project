from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def _path_arg(path: str | Path) -> str:
    return Path(path).as_posix()


@dataclass(frozen=True)
class ComparisonMethod:
    name: str
    prediction_dir: Path
    display_name: str = ""
    metric_name: str = ""
    downstream_name: str = ""
    table_group: str = "main"
    table_groups: tuple[str, ...] = ()
    is_ours: bool = False
    notes: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def metric_key(self) -> str:
        return self.metric_name or self.name

    @property
    def downstream_key(self) -> str:
        return self.downstream_name or self.name

    @property
    def groups(self) -> tuple[str, ...]:
        return self.table_groups or (self.table_group,)


@dataclass(frozen=True)
class ComparisonBaselines:
    include_t1: bool = False
    include_fa_gt: bool = False


@dataclass(frozen=True)
class DownstreamConfig:
    tasks: str = "cn_scd_vs_mci_ad"
    repeat_seeds: str = ""
    n_splits: int = 5
    max_features: int = 12
    fusions: list[str] = field(default_factory=list)
    output_root: Path = Path("outputs/icdm2026/downstream_comparison")


@dataclass(frozen=True)
class ComparisonManifest:
    methods: list[ComparisonMethod]
    baselines: ComparisonBaselines = field(default_factory=ComparisonBaselines)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_comparison_manifest(path: str | Path, raw: dict[str, Any] | None = None) -> ComparisonManifest:
    data = raw if raw is not None else _read_yaml(Path(path))
    method_rows = data.get("methods", [])
    if not isinstance(method_rows, list):
        raise ValueError("Comparison manifest field 'methods' must be a list.")

    methods: list[ComparisonMethod] = []
    seen: set[str] = set()
    for row in method_rows:
        if not isinstance(row, dict):
            raise ValueError("Each comparison method must be a mapping.")
        name = str(row.get("name", "")).strip()
        pred_dir = str(row.get("prediction_dir", "")).strip()
        if not name or not pred_dir:
            raise ValueError("Each comparison method needs name and prediction_dir.")
        if name in seen:
            raise ValueError(f"Duplicate method name in comparison manifest: {name}")
        seen.add(name)
        raw_groups = row.get("table_groups", [])
        if isinstance(raw_groups, str):
            table_groups = tuple(group.strip() for group in raw_groups.split(",") if group.strip())
        else:
            table_groups = tuple(str(group).strip() for group in (raw_groups or []) if str(group).strip())
        methods.append(
            ComparisonMethod(
                name=name,
                prediction_dir=Path(pred_dir),
                display_name=str(row.get("display_name", "")).strip(),
                metric_name=str(row.get("metric_name", "")).strip(),
                downstream_name=str(row.get("downstream_name", "")).strip(),
                table_group=str(row.get("table_group", "main")).strip() or "main",
                table_groups=table_groups,
                is_ours=bool(row.get("is_ours", False)),
                notes=str(row.get("notes", "")).strip(),
            )
        )

    baseline_data = data.get("baselines", {}) or {}
    downstream_data = data.get("downstream", {}) or {}
    return ComparisonManifest(
        methods=methods,
        baselines=ComparisonBaselines(
            include_t1=bool(baseline_data.get("include_t1", False)),
            include_fa_gt=bool(baseline_data.get("include_fa_gt", False)),
        ),
        downstream=DownstreamConfig(
            tasks=str(downstream_data.get("tasks", "cn_scd_vs_mci_ad")),
            repeat_seeds=str(downstream_data.get("repeat_seeds", "")),
            n_splits=int(downstream_data.get("n_splits", 5)),
            max_features=int(downstream_data.get("max_features", 12)),
            fusions=list(downstream_data.get("fusions", []) or []),
            output_root=Path(downstream_data.get("output_root", "outputs/icdm2026/downstream_comparison")),
        ),
    )


def build_image_eval_command(
    method: ComparisonMethod,
    *,
    config: str = "configs/icdm2026.yaml",
    visualize_count: int = 8,
    force_visual_manifest: bool = False,
) -> list[str]:
    command = [
        "python",
        "scripts/evaluate_method_folder.py",
        "--pred_dir",
        _path_arg(method.prediction_dir),
        "--method",
        method.name,
        "--config",
        config,
        "--visualize_count",
        str(visualize_count),
    ]
    if force_visual_manifest:
        command.append("--reset_visualize_manifest")
    return command


def build_downstream_command(
    manifest: ComparisonManifest,
    *,
    output_root: Path | None = None,
    config: str = "configs/icdm2026.yaml",
) -> list[str]:
    downstream = manifest.downstream
    command = [
        "python",
        "scripts/evaluate_downstream_classification.py",
        "--config",
        config,
        "--tasks",
        downstream.tasks,
        "--n_splits",
        str(downstream.n_splits),
        "--max_features",
        str(downstream.max_features),
        "--output_root",
        str(output_root or downstream.output_root),
    ]
    if downstream.repeat_seeds:
        command.extend(["--repeat_seeds", downstream.repeat_seeds])
    if manifest.baselines.include_t1:
        command.append("--include_t1")
    if manifest.baselines.include_fa_gt:
        command.append("--include_fa_gt")
    for method in manifest.methods:
        command.extend(["--method", f"{method.name}={_path_arg(method.prediction_dir)}"])
    for fusion in downstream.fusions:
        command.extend(["--fusion", fusion])
    return command


def check_prediction_dirs(methods: Iterable[ComparisonMethod]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        exists = method.prediction_dir.exists()
        png_count = len(list(method.prediction_dir.glob("*.png"))) if exists else 0
        rows.append(
            {
                "method": method.name,
                "prediction_dir": str(method.prediction_dir),
                "exists": exists,
                "png_count": png_count,
            }
        )
    return rows


def run_commands(commands: Iterable[list[str]], *, dry_run: bool = False) -> list[dict[str, Any]]:
    results = []
    for command in commands:
        if dry_run:
            results.append({"command": command, "returncode": None, "skipped": True})
            print("DRY-RUN:", " ".join(command))
            continue
        completed = subprocess.run(command, check=False)
        results.append({"command": command, "returncode": completed.returncode, "skipped": False})
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")
    return results


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float):
        return None
    return value_float


def _downstream_lookup(downstream_root: Path, downstream_task: str) -> dict[str, dict[str, Any]]:
    repeated_path = downstream_root / "classification_repeated_summary.csv"
    single_path = downstream_root / "classification_summary.csv"
    if repeated_path.exists():
        df = pd.read_csv(repeated_path)
        if "task" in df.columns:
            df = df[df["task"] == downstream_task]
        return {
            str(row["method"]): row.to_dict()
            for _, row in df.iterrows()
        }
    if single_path.exists():
        df = pd.read_csv(single_path)
        if "task" in df.columns:
            df = df[df["task"] == downstream_task]
        return {
            str(row["method"]): row.to_dict()
            for _, row in df.iterrows()
        }
    return {}


def build_table_rows(
    manifest: ComparisonManifest,
    *,
    metrics_root: str | Path = "outputs/icdm2026/metrics",
    downstream_root: str | Path = "outputs/icdm2026/downstream_comparison",
    downstream_task: str = "cn_scd_vs_mci_ad",
) -> list[dict[str, Any]]:
    metrics_root = Path(metrics_root)
    downstream_root = Path(downstream_root)
    downstream_by_method = _downstream_lookup(downstream_root, downstream_task)

    rows = []
    for method in manifest.methods:
        summary_path = metrics_root / f"{method.metric_key}_summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        downstream = downstream_by_method.get(method.downstream_key, {})
        macro_f1 = downstream.get("macro_f1_mean", downstream.get("macro_f1"))
        macro_f1_std = downstream.get("macro_f1_std")
        rows.append(
            {
                "method": method.name,
                "display_name": method.label,
                "table_group": method.groups[0] if method.groups else method.table_group,
                "table_groups": list(method.groups),
                "is_ours": method.is_ours,
                "PSNR": _safe_float(summary.get("PSNR_mean")),
                "SSIM": _safe_float(summary.get("SSIM_mean")),
                "SharpRatio": _safe_float(summary.get("Sharpness_Ratio_mean")),
                "WM_MAE": _safe_float(summary.get("WM_Masked_MAE_mean")),
                "ROI_CCC": _safe_float(summary.get("ROI_CCC")),
                "Macro_F1": _safe_float(macro_f1),
                "Macro_F1_Std": _safe_float(macro_f1_std),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(value_float):
        return "inf"
    return f"{value_float:.{digits}f}"


def _fmt_macro(row: dict[str, Any]) -> str:
    if row.get("Macro_F1") is None:
        return "-"
    if row.get("Macro_F1_Std") is None:
        return _fmt(row.get("Macro_F1"), 3)
    return f"{_fmt(row.get('Macro_F1'), 3)} +/- {_fmt(row.get('Macro_F1_Std'), 3)}"


def _markdown_lines(rows: list[dict[str, Any]], *, title: str, cn: bool = False) -> list[str]:
    if cn:
        lines = [
            f"# {title}",
            "",
            "说明：PSNR、SSIM、SharpRatio、ROI-CCC 和 Macro-F1 越高越好；WM-MAE 越低越好。",
            "",
            "| Method | PSNR | SSIM | SharpRatio | WM-MAE | ROI-CCC | Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            f"# {title}",
            "",
            "Note: higher is better for PSNR, SSIM, SharpRatio, ROI-CCC, and Macro-F1; lower is better for WM-MAE.",
            "",
            "| Method | PSNR | SSIM | SharpRatio | WM-MAE | ROI-CCC | Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        name = str(row.get("display_name") or row.get("method"))
        if row.get("is_ours"):
            name = f"**{name}**"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(row.get("PSNR"), 2),
                    _fmt(row.get("SSIM"), 4),
                    _fmt(row.get("SharpRatio"), 4),
                    _fmt(row.get("WM_MAE"), 5),
                    _fmt(row.get("ROI_CCC"), 4),
                    _fmt_macro(row),
                ]
            )
            + " |"
        )
    return lines


def write_markdown_table(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    title_cn: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(_markdown_lines(rows, title=title, cn=False)) + "\n", encoding="utf-8")
    cn_path = output_path.with_name(f"{output_path.stem}_cn{output_path.suffix}")
    cn_path.write_text("\n".join(_markdown_lines(rows, title=title_cn, cn=True)) + "\n", encoding="utf-8")


def write_table_bundle(rows: list[dict[str, Any]], output_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_root / "comparison_results_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    main_rows = [row for row in rows if "main" in (row.get("table_groups") or [row.get("table_group")])]
    ablation_rows = [row for row in rows if "ablation" in (row.get("table_groups") or [row.get("table_group")])]
    sota_rows = [row for row in rows if "sota" in (row.get("table_groups") or [row.get("table_group")])]
    frequency_rows = [row for row in rows if "frequency" in (row.get("table_groups") or [row.get("table_group")])]
    write_markdown_table(main_rows or rows, output_root / "main_results_table.md", title="Main Results Table", title_cn="主结果表")
    write_markdown_table(ablation_rows, output_root / "ablation_table.md", title="Ablation Table", title_cn="消融实验表")
    write_markdown_table(sota_rows, output_root / "sota_comparison_table.md", title="SOTA Comparison Table", title_cn="SOTA 对比表")
    write_markdown_table(frequency_rows, output_root / "frequency_fusion_table.md", title="Frequency Fusion Table", title_cn="频率融合表")
    return {
        "csv": csv_path,
        "main": output_root / "main_results_table.md",
        "main_cn": output_root / "main_results_table_cn.md",
        "ablation": output_root / "ablation_table.md",
        "ablation_cn": output_root / "ablation_table_cn.md",
        "sota": output_root / "sota_comparison_table.md",
        "sota_cn": output_root / "sota_comparison_table_cn.md",
        "frequency": output_root / "frequency_fusion_table.md",
        "frequency_cn": output_root / "frequency_fusion_table_cn.md",
    }
