from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E2EDiT full training only after a GO smoke decision.")
    p.add_argument("--smoke_root", default="outputs/icdm2026/e2edit/smoke")
    p.add_argument("--output_root", default="outputs/icdm2026/e2edit/full/ADNI_E2EDIT_FULL")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--mixed_precision", default="bf16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decision = Path(args.smoke_root) / "reports" / "go_no_go_decision.md"
    if not decision.exists() or "Decision: GO" not in decision.read_text(encoding="utf-8"):
        raise SystemExit(f"Refusing full training because smoke GO was not found: {decision}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/train_e2edit_smoke.py",
        "--output_root",
        str(output_root),
        "--train_limit",
        "0",
        "--val_limit",
        "0",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--width",
        str(args.width),
        "--num_blocks",
        str(args.num_blocks),
        "--lr",
        str(args.lr),
        "--mixed_precision",
        str(args.mixed_precision),
    ]
    print("Running full E2EDiT via:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
