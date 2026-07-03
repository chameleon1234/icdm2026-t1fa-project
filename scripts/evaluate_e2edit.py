from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pmrf_t1fa.e2edit import compute_batch_metrics, make_wm_proxy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an E2EDiT prediction folder.")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--target_dir", default="data/adni_processed/test/fa_slices")
    p.add_argument("--t1_dir", default="data/adni_processed/test/t1_slices")
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def _read(path: Path) -> torch.Tensor:
    stream = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    arr = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).view(1, 1, img.shape[0], img.shape[1])


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for row in rows for k in row})
    return {f"{k}_mean": float(np.nanmean([row.get(k, math.nan) for row in rows])) for k in keys}


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    target_dir = Path(args.target_dir)
    t1_dir = Path(args.t1_dir)
    out_dir = Path(args.output_dir) if args.output_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_path in tqdm(sorted(pred_dir.glob("*.png")), desc="Evaluate E2EDiT"):
        target_path = target_dir / pred_path.name
        t1_path = t1_dir / pred_path.name
        if not target_path.exists() or not t1_path.exists():
            continue
        pred = _read(pred_path)
        target = _read(target_path)
        t1 = _read(t1_path)
        wm = make_wm_proxy(t1)
        metrics = compute_batch_metrics(pred, target, wm)
        rows.append({"filename": pred_path.name, **metrics})
    summary = {"n_slices": len(rows), **_mean(rows)}
    with (out_dir / "e2edit_slice_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "e2edit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
