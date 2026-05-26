import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import (
    RefinementFlowUNet,
    Stage1Net,
    euler_refine,
    infer_stage1_prediction_mode,
    predict_stage1_fa,
    reduce_rgb_to_single_channel,
)
from src.datasets import T1FADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PMRF/PM-DIRF predictions as test-set-aligned PNG files.")
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2"], help="stage1 exports PM_STAGE1; stage2 exports PM_DIRF.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt")
    parser.add_argument("--stage2_ckpt", default="outputs/pm_dirf_default/checkpoints/best_stage2.pt")
    parser.add_argument("--output_dir", default="", help="Override default prediction output directory.")
    parser.add_argument("--test_t1_dir", default="", help="Override test T1 slice directory from config.")
    parser.add_argument("--test_fa_dir", default="", help="Override test FA slice directory from config.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=-1, help="Use checkpoint eval_steps when -1.")
    parser.add_argument("--condition_on_coarse", action="store_true", help="Force Stage 2 coarse conditioning on.")
    parser.add_argument("--disable_condition_on_coarse", action="store_false", dest="condition_on_coarse")
    parser.add_argument("--auto_condition_from_ckpt", action="store_true")
    parser.set_defaults(condition_on_coarse=True)
    return parser.parse_args()


def stage_to_method(stage: str) -> str:
    if stage == "stage1":
        return "PM_STAGE1"
    if stage == "stage2":
        return "PM_DIRF"
    raise ValueError(f"Unsupported stage: {stage}")


def resolve_output_dir(stage: str, predictions_root: str | Path, output_dir: str | Path) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(predictions_root) / stage_to_method(stage)


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


def load_stage1(stage1_ckpt: str | Path, device: torch.device) -> tuple[Stage1Net, str]:
    checkpoint = torch.load(stage1_ckpt, map_location="cpu")
    model = Stage1Net(in_channels=1, out_channels=1)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model, infer_stage1_prediction_mode(checkpoint_args(checkpoint))


def resolve_stage2_settings(args: argparse.Namespace) -> tuple[bool, int]:
    if args.stage != "stage2":
        return False, 0
    checkpoint = torch.load(args.stage2_ckpt, map_location="cpu")
    ckpt_args = checkpoint_args(checkpoint)
    condition_on_coarse = bool(args.condition_on_coarse)
    if args.auto_condition_from_ckpt:
        condition_on_coarse = bool(ckpt_args.get("condition_on_coarse", condition_on_coarse))
    eval_steps = args.eval_steps if args.eval_steps > 0 else int(ckpt_args.get("eval_steps", 1))
    return condition_on_coarse, eval_steps


def load_stage2(stage2_ckpt: str | Path, device: torch.device, condition_on_coarse: bool) -> RefinementFlowUNet:
    checkpoint = torch.load(stage2_ckpt, map_location="cpu")
    condition_channels = 1 if condition_on_coarse else 0
    model = RefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def save_prediction_png(prediction: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = prediction.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
    image_u8 = (image_01.numpy() * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image_u8)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {output_path}")
    encoded.tofile(str(output_path))


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
    output_dir = resolve_output_dir(args.stage, predictions_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.stage1_ckpt).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {args.stage1_ckpt}")
    if args.stage == "stage2" and not Path(args.stage2_ckpt).exists():
        raise FileNotFoundError(f"Missing Stage 2 checkpoint: {args.stage2_ckpt}")

    device = torch.device(args.device)
    dataset = T1FADataset(str(test_t1_dir), str(test_fa_dir), preload_ram=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    stage1, stage1_prediction_mode = load_stage1(args.stage1_ckpt, device)
    condition_on_coarse, eval_steps = resolve_stage2_settings(args)
    stage2 = None
    if args.stage == "stage2":
        stage2 = load_stage2(args.stage2_ckpt, device, condition_on_coarse=condition_on_coarse)

    exported_rows: list[dict[str, Any]] = []
    exported = 0
    method = stage_to_method(args.stage)
    for batch in tqdm(loader, desc=f"Exporting {method}", leave=False):
        t1_img = reduce_rgb_to_single_channel(batch["t1_slice"].to(device))
        coarse = predict_stage1_fa(stage1, t1_img, clamp=True, prediction_mode=stage1_prediction_mode)
        if args.stage == "stage1":
            prediction = coarse
        else:
            assert stage2 is not None
            condition = coarse if condition_on_coarse else None
            prediction = euler_refine(stage2, coarse, num_steps=eval_steps, condition=condition, clamp=True)

        batch_filenames = batch["fname"]
        for i, filename in enumerate(batch_filenames):
            if args.limit > 0 and exported >= args.limit:
                break
            output_path = output_dir / str(filename)
            save_prediction_png(prediction[i : i + 1], output_path)
            exported_rows.append(
                {
                    "method": method,
                    "stage": args.stage,
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
        "method": method,
        "stage": args.stage,
        "output_dir": str(output_dir),
        "exported": exported,
        "stage1_ckpt": str(args.stage1_ckpt),
        "stage1_prediction_mode": stage1_prediction_mode,
        "stage2_ckpt": str(args.stage2_ckpt) if args.stage == "stage2" else "",
        "condition_on_coarse": condition_on_coarse if args.stage == "stage2" else False,
        "eval_steps": eval_steps if args.stage == "stage2" else 0,
        "manifest": str(manifest_path),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"{method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
