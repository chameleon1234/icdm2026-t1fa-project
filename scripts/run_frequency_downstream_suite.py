import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full/lowpass/highpass downstream utility views.")
    parser.add_argument("--output_base", default="outputs/icdm2026/downstream_frequency")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--repeat_seeds", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views = ["full", "lowpass", "highpass"]
    for view in views:
        command = [
            sys.executable,
            "scripts/run_binary_downstream_suite.py",
            "--feature_view",
            view,
            "--output_root",
            str(Path(args.output_base) / view),
        ]
        if args.tasks:
            command.extend(["--tasks", args.tasks])
        if args.repeat_seeds:
            command.extend(["--repeat_seeds", args.repeat_seeds])
        if args.dry_run:
            command.append("--dry_run")
            print("DRY-RUN:", " ".join(command))
            continue
        result = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
