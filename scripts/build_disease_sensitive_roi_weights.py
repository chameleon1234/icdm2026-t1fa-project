import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index
from src.eval.disease_sensitive_roi import build_disease_roi_weights
from src.eval.downstream_utility import extract_subject_features_from_folder


DEFAULT_TASKS = "cn_vs_ad,cn_vs_mci,mci_vs_ad,cn_scd_vs_mci_ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disease-sensitive ROI weights from training-set real FA.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--output_csv", default="outputs/icdm2026/roi_weights/private_disease_sensitive_roi_weights.csv")
    parser.add_argument("--feature_csv", default="", help="Optional path to save extracted train FA ROI features.")
    return parser.parse_args()


def build_command_args(project_root: Path, output_csv: Path) -> list[str]:
    return [
        str(project_root / "data/processed/train/fa_slices"),
        "--split",
        "train",
        "--tasks",
        DEFAULT_TASKS,
        "--output_csv",
        str(output_csv),
    ]


def _read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_subject_index(config_path: str | Path) -> pd.DataFrame:
    config = _read_yaml(config_path)
    data_config = config["data"]
    return load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    subject_index = _load_subject_index(args.config)
    features = extract_subject_features_from_folder(
        image_dir=args.fa_dir,
        method="FA_GT",
        subject_index=subject_index,
        split=args.split,
    )
    weights = build_disease_roi_weights(features, tasks=tasks, method="FA_GT", top_k=args.top_k)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output_csv, index=False, encoding="utf-8")
    if args.feature_csv:
        feature_csv = Path(args.feature_csv)
        feature_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_csv, index=False, encoding="utf-8")
    print(f"Saved disease-sensitive ROI weights to: {output_csv}")
    print(weights.to_string(index=False))


if __name__ == "__main__":
    main()
