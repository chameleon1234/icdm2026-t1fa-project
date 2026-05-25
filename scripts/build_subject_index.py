import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import GROUP_MAPPING, load_subject_index, save_subject_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level label index for ICDM 2026 experiments")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_config = config["data"]
    output_path = Path(config["outputs"]["subject_index"])
    df = load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )
    save_subject_index(df, output_path)

    split_counts = df.groupby("split").size().to_dict()
    group_counts = df.groupby("group_name").size().to_dict()
    group_summary = ", ".join(f"{name}={group_counts[name]}" for name in ["CN", "SCD", "MCI", "AD"])
    print(f"Saved subject index to {output_path.as_posix()}")
    print(f"Subjects: {len(df)}")
    print(
        "Splits: "
        f"train={split_counts.get('train', 0)}, "
        f"val={split_counts.get('val', 0)}, "
        f"test={split_counts.get('test', 0)}"
    )
    print(f"Groups: {group_summary}")
    print("Mapping: " + ", ".join(f"{key}={value}" for key, value in GROUP_MAPPING.items()))


if __name__ == "__main__":
    main()
