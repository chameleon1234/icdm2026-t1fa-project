import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import prepare_stage1_input, predict_stage1_fa, reduce_rgb_to_single_channel
from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import (
    HighPassRefinerNet,
    apply_hp_residual_cap,
    build_hp_refiner_input,
    build_residual_gate,
)
from scripts.export_pm_dirf_predictions import load_stage1, make_slice_dataset, save_prediction_png


METHOD = "PM_DIRF_HP_REFINER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-pass residual PM-DIRF predictions as aligned PNG files.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--hp_refiner_ckpt", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--test_t1_dir", default="")
    parser.add_argument("--test_fa_dir", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--residual_scale_override", type=float, default=-1.0)
    parser.add_argument("--residual_gate_mode_override", default="", choices=["", "none", "edge"])
    parser.add_argument("--residual_gate_min_override", type=float, default=-1.0)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def checkpoint_args(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {})
    return {}


def resolve_output_dir(predictions_root: str | Path, output_dir: str | Path) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(predictions_root) / METHOD


def load_hp_refiner(
    hp_refiner_ckpt: str | Path,
    device: torch.device,
    stage1_channels: int,
) -> tuple[HighPassRefinerNet, dict[str, Any]]:
    checkpoint = torch.load(hp_refiner_ckpt, map_location="cpu")
    ckpt_args = checkpoint_args(checkpoint)
    width = int(ckpt_args.get("width", 32))
    num_blocks = int(ckpt_args.get("num_blocks", 8))
    model = HighPassRefinerNet(in_channels=stage1_channels + 3, width=width, num_blocks=num_blocks)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    hp_args = {
        "width": width,
        "num_blocks": num_blocks,
        "residual_scale": float(ckpt_args.get("residual_scale", 0.0)),
        "residual_gate_mode": str(ckpt_args.get("residual_gate_mode", "none")),
        "residual_gate_min": float(ckpt_args.get("residual_gate_min", 0.2)),
    }
    return model, hp_args


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_yaml(args.config)
    processed_root = Path(config["data"]["processed_root"])
    predictions_root = Path(config["outputs"]["predictions_root"])
    test_t1_dir = Path(args.test_t1_dir) if args.test_t1_dir else processed_root / "test" / "t1_slices"
    test_fa_dir = Path(args.test_fa_dir) if args.test_fa_dir else processed_root / "test" / "fa_slices"
    output_dir = resolve_output_dir(predictions_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.stage1_ckpt).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {args.stage1_ckpt}")
    if not Path(args.hp_refiner_ckpt).exists():
        raise FileNotFoundError(f"Missing HP refiner checkpoint: {args.hp_refiner_ckpt}")

    device = torch.device(args.device)
    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1(args.stage1_ckpt, device)
    hp_refiner, hp_args = load_hp_refiner(args.hp_refiner_ckpt, device, stage1_channels)
    if args.residual_scale_override >= 0:
        hp_args["residual_scale"] = float(args.residual_scale_override)
    if args.residual_gate_mode_override:
        hp_args["residual_gate_mode"] = args.residual_gate_mode_override
    if args.residual_gate_min_override >= 0:
        hp_args["residual_gate_min"] = float(args.residual_gate_min_override)

    dataset = make_slice_dataset(test_t1_dir, test_fa_dir, stage1_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    exported_rows: list[dict[str, Any]] = []
    exported = 0
    for batch in tqdm(loader, desc=f"Exporting {METHOD}", leave=False):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        coarse = predict_stage1_fa(
            stage1,
            t1_img,
            clamp=True,
            prediction_mode=stage1_prediction_mode,
            detail_scale=stage1_detail_scale,
        )
        residual_gate = build_residual_gate(
            t1_img,
            coarse,
            mode=hp_args["residual_gate_mode"],
            gate_min=hp_args["residual_gate_min"],
        )
        hp_pred = apply_hp_residual_cap(
            hp_refiner(build_hp_refiner_input(t1_img, coarse)),
            residual_scale=hp_args["residual_scale"],
            residual_gate=residual_gate,
        )
        prediction = torch.clamp(coarse + hp_pred, -1.0, 1.0)

        for i, filename in enumerate(batch["fname"]):
            if args.limit > 0 and exported >= args.limit:
                break
            output_path = output_dir / str(filename)
            save_prediction_png(prediction[i : i + 1], output_path)
            exported_rows.append(
                {
                    "method": METHOD,
                    "fname": str(filename),
                    "output_path": str(output_path),
                }
            )
            exported += 1
        if args.limit > 0 and exported >= args.limit:
            break

    manifest_path = output_dir / "export_manifest.csv"
    _write_manifest(manifest_path, exported_rows)
    summary = {
        "method": METHOD,
        "output_dir": str(output_dir),
        "exported": exported,
        "stage1_ckpt": str(args.stage1_ckpt),
        "hp_refiner_ckpt": str(args.hp_refiner_ckpt),
        "stage1_prediction_mode": stage1_prediction_mode,
        "stage1_detail_scale": stage1_detail_scale,
        "stage1_channels": stage1_channels,
        "residual_scale": hp_args["residual_scale"],
        "residual_gate_mode": hp_args["residual_gate_mode"],
        "residual_gate_min": hp_args["residual_gate_min"],
        "manifest": str(manifest_path),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"{METHOD}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
