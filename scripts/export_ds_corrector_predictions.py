from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import prepare_stage1_input
from pmrf_t1fa.train_pmrf_t1fa_stage2 import (
    build_training_masks,
    load_stage1_model,
    make_slice_dataset,
    predict_stage1_batch,
)
from pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector import (
    SingleSliceCorrector,
    build_condition,
    build_loss,
    load_disease_roi_weights,
    roi_weight_map,
)
from pmrf_t1fa.train_pmrf_t1fa_stage2 import SSIMLoss
from scripts.export_pm_dirf_predictions import save_prediction_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export single-slice DS Stage2 corrector predictions.")
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--ds_corrector_ckpt", required=True)
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1_model(args.stage1_ckpt, device)
    if stage1_channels != 1:
        raise ValueError(f"DS corrector export is single-slice only, got stage1_channels={stage1_channels}")
    checkpoint = torch.load(args.ds_corrector_ckpt, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    variant = str(ckpt_args.get("variant", "hybrid"))
    width = int(ckpt_args.get("width", 48))
    num_blocks = int(ckpt_args.get("num_blocks", 8))
    model = SingleSliceCorrector(
        in_channels=9,
        width=width,
        num_blocks=num_blocks,
        uncertainty=variant in {"uncertainty", "hybrid"},
        variant=variant,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = make_slice_dataset(args.test_t1_dir, args.test_fa_dir, stage1_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    roi_weights = load_disease_roi_weights(
        str(ckpt_args.get("disease_roi_csv", "")),
        int(ckpt_args.get("roi_rows", 4)),
        int(ckpt_args.get("roi_cols", 4)),
        device,
    )
    ssim_loss = SSIMLoss().to(device)
    rows: list[dict[str, Any]] = []
    exported = 0
    for batch in tqdm(loader, desc="Export DS corrector", leave=False):
        t1 = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        target = batch["fa_slice"].to(device)
        target = target.mean(dim=1, keepdim=True) if target.shape[1] > 1 else target
        coarse = predict_stage1_batch(stage1, t1, device, prediction_mode, detail_scale).to(device)
        brain_mask, wm_mask = build_training_masks(
            t1,
            target,
            float(ckpt_args.get("brain_t1_threshold", 0.05)),
            float(ckpt_args.get("brain_fa_threshold", 0.02)),
            float(ckpt_args.get("wm_quantile", 0.65)),
            float(ckpt_args.get("wm_min_threshold", 0.20)),
        )
        rmap = roi_weight_map(roi_weights, coarse, wm_mask)
        condition = build_condition(
            t1,
            coarse,
            wm_mask,
            rmap,
            variant,
            int(ckpt_args.get("lowpass_kernel", 13)),
            int(ckpt_args.get("hp_kernel", 5)),
        )
        raw, log_sigma = model(condition)
        _, _, refined = build_loss(raw, log_sigma, coarse, target, brain_mask, wm_mask, rmap, roi_weights, ssim_loss, argparse.Namespace(**ckpt_args))
        for index, fname in enumerate(batch["fname"]):
            if args.limit > 0 and exported >= args.limit:
                break
            out_path = output_dir / str(fname)
            save_prediction_png(refined[index : index + 1], out_path)
            rows.append({"method": "DS_CORRECTOR", "variant": variant, "fname": str(fname), "output_path": str(out_path)})
            exported += 1
        if args.limit > 0 and exported >= args.limit:
            break
    manifest = output_dir / "export_manifest.csv"
    _write_manifest(manifest, rows)
    summary = {
        "method": "DS_CORRECTOR",
        "variant": variant,
        "output_dir": str(output_dir),
        "exported": exported,
        "stage1_ckpt": str(args.stage1_ckpt),
        "ds_corrector_ckpt": str(args.ds_corrector_ckpt),
        "stage1_channels": stage1_channels,
        "manifest": str(manifest),
        "settings": ckpt_args,
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"DS_CORRECTOR: exported={exported} output_dir={output_dir}")


if __name__ == "__main__":
    main()
