import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BINARY_TASKS = [
    "cn_vs_scd",
    "cn_vs_mci",
    "cn_vs_ad",
    "scd_vs_mci",
    "mci_vs_ad",
    "cn_vs_mci_ad",
    "cn_scd_vs_ad",
    "cn_scd_vs_mci_ad",
    "cn_scd_mci_vs_ad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all AD-staging binary downstream tasks for selected T1-to-FA methods.")
    parser.add_argument("--manifest", default="configs/icdm2026_comparison_methods.yaml")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--output_root", default="outputs/icdm2026/downstream_binary_tasks")
    parser.add_argument("--tasks", default=",".join(DEFAULT_BINARY_TASKS))
    parser.add_argument("--repeat_seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load_manifest_methods(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    methods = []
    for row in manifest.get("methods", []):
        pred_dir = Path(row["prediction_dir"])
        if pred_dir.exists() and any(pred_dir.glob("*.png")):
            methods.append((str(row["name"]), pred_dir.as_posix()))
    return methods


def main() -> None:
    args = parse_args()
    methods = _load_manifest_methods(Path(args.manifest))
    command = [
        sys.executable,
        "scripts/evaluate_downstream_classification.py",
        "--config",
        args.config,
        "--tasks",
        args.tasks,
        "--repeat_seeds",
        args.repeat_seeds,
        "--n_splits",
        str(args.n_splits),
        "--max_features",
        str(args.max_features),
        "--output_root",
        args.output_root,
        "--include_t1",
        "--include_fa_gt",
    ]
    for name, pred_dir in methods:
        command.extend(["--method", f"{name}={pred_dir}"])
    if args.dry_run:
        print("DRY-RUN:", " ".join(command))
        return
    result = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
