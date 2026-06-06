import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.comparison_experiments import build_table_rows, load_comparison_manifest, write_table_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICDM 2026 paper tables from metric JSON and downstream CSV outputs.")
    parser.add_argument("--manifest", default="configs/icdm2026_comparison_methods.yaml")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--downstream_root", default="outputs/icdm2026/downstream_comparison")
    parser.add_argument("--downstream_task", default="cn_scd_vs_mci_ad")
    parser.add_argument("--output_root", default="outputs/icdm2026/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_comparison_manifest(args.manifest)
    rows = build_table_rows(
        manifest,
        metrics_root=args.metrics_root,
        downstream_root=args.downstream_root,
        downstream_task=args.downstream_task,
    )
    paths = write_table_bundle(rows, args.output_root)
    print("Saved paper tables:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
