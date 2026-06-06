import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.comparison_experiments import (
    build_downstream_command,
    build_image_eval_command,
    check_prediction_dirs,
    load_comparison_manifest,
    run_commands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ICDM 2026 comparison experiment suite from a method manifest.")
    parser.add_argument("--manifest", default="configs/icdm2026_comparison_methods.yaml")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--skip_image_metrics", action="store_true")
    parser.add_argument("--skip_downstream", action="store_true")
    parser.add_argument("--force_image_metrics", action="store_true", help="Recompute image metrics even if summary JSON exists.")
    parser.add_argument("--visualize_count", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--status_json", default="outputs/icdm2026/tables/comparison_status.json")
    return parser.parse_args()


def _summary_exists(method_name: str) -> bool:
    return (Path("outputs/icdm2026/metrics") / f"{method_name}_summary.json").exists()


def main() -> None:
    args = parse_args()
    manifest = load_comparison_manifest(args.manifest)
    status_rows = check_prediction_dirs(manifest.methods)
    missing = [row for row in status_rows if not row["exists"] or row["png_count"] == 0]
    if missing:
        print("Missing or empty prediction folders:")
        for row in missing:
            print(f"  - {row['method']}: {row['prediction_dir']} png_count={row['png_count']}")

    commands: list[list[str]] = []
    if not args.skip_image_metrics:
        for method, status in zip(manifest.methods, status_rows):
            if not status["exists"] or status["png_count"] == 0:
                continue
            if _summary_exists(method.metric_key) and not args.force_image_metrics:
                continue
            commands.append(
                build_image_eval_command(
                    method,
                    config=args.config,
                    visualize_count=args.visualize_count,
                )
            )
    if not args.skip_downstream:
        available_names = {row["method"] for row in status_rows if row["exists"] and row["png_count"] > 0}
        available_manifest = load_comparison_manifest(
            args.manifest,
            raw={
                "methods": [
                    {
                        "name": method.name,
                        "prediction_dir": str(method.prediction_dir),
                        "display_name": method.display_name,
                        "metric_name": method.metric_name,
                        "downstream_name": method.downstream_name,
                        "table_group": method.table_group,
                        "table_groups": list(method.groups),
                        "is_ours": method.is_ours,
                        "notes": method.notes,
                    }
                    for method in manifest.methods
                    if method.name in available_names
                ],
                "baselines": {
                    "include_t1": manifest.baselines.include_t1,
                    "include_fa_gt": manifest.baselines.include_fa_gt,
                },
                "downstream": {
                    "tasks": manifest.downstream.tasks,
                    "repeat_seeds": manifest.downstream.repeat_seeds,
                    "n_splits": manifest.downstream.n_splits,
                    "max_features": manifest.downstream.max_features,
                    "output_root": str(manifest.downstream.output_root),
                    "fusions": manifest.downstream.fusions,
                },
            },
        )
        commands.append(build_downstream_command(available_manifest, config=args.config))

    results = run_commands(commands, dry_run=args.dry_run)
    status_path = Path(args.status_json)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"prediction_status": status_rows, "commands": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved comparison status to: {status_path}")


if __name__ == "__main__":
    main()
