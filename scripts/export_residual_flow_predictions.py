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

from pmrf_t1fa.models.pmrf_t1fa import prepare_stage1_input, predict_stage1_fa
from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import ResidualFlowStage2, euler_sample_residual_flow
from scripts.export_pm_dirf_predictions import load_stage1, make_slice_dataset, save_prediction_png


METHOD = "PM_DIRF_RESIDUAL_FLOW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PMRF-T1FA residual-flow predictions as aligned PNG files.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--residual_flow_ckpt", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--test_t1_dir", default="")
    parser.add_argument("--test_fa_dir", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps_override", type=int, default=0)
    parser.add_argument("--eval_noise_scale_override", type=float, default=-1.0)
    parser.add_argument("--eval_samples_override", type=int, default=0)
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


def load_residual_flow(
    residual_flow_ckpt: str | Path,
    device: torch.device,
    stage1_channels: int,
) -> tuple[ResidualFlowStage2, dict[str, Any]]:
    checkpoint = torch.load(residual_flow_ckpt, map_location="cpu")
    ckpt_args = checkpoint_args(checkpoint)
    width = int(ckpt_args.get("width", 48))
    num_blocks = int(ckpt_args.get("num_blocks", 8))
    sigma_min = float(ckpt_args.get("sigma_min", 0.01))
    sigma_max = float(ckpt_args.get("sigma_max", 0.45))
    model = ResidualFlowStage2(
        stage1_channels=stage1_channels,
        width=width,
        num_blocks=num_blocks,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    flow_args = {
        "width": width,
        "num_blocks": num_blocks,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "eval_steps": int(ckpt_args.get("eval_steps", 8)),
        "eval_noise_scale": float(ckpt_args.get("eval_noise_scale", 0.0)),
        "eval_samples": int(ckpt_args.get("eval_samples", 1)),
    }
    return model, flow_args


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

    device = torch.device(args.device)
    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1(args.stage1_ckpt, device)
    residual_flow, flow_args = load_residual_flow(args.residual_flow_ckpt, device, stage1_channels)
    if args.eval_steps_override > 0:
        flow_args["eval_steps"] = int(args.eval_steps_override)
    if args.eval_noise_scale_override >= 0:
        flow_args["eval_noise_scale"] = float(args.eval_noise_scale_override)
    if args.eval_samples_override > 0:
        flow_args["eval_samples"] = int(args.eval_samples_override)

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
        sigma = residual_flow.predict_sigma(t1_img, coarse)
        residual_samples = []
        for _ in range(max(int(flow_args["eval_samples"]), 1)):
            residual_samples.append(
                euler_sample_residual_flow(
                    residual_flow,
                    t1_img,
                    coarse,
                    sigma,
                    steps=int(flow_args["eval_steps"]),
                    noise_scale=float(flow_args["eval_noise_scale"]),
                )
            )
        pred_residual = torch.stack(residual_samples, dim=0).mean(dim=0)
        prediction = torch.clamp(coarse + pred_residual, -1.0, 1.0)
        for i, filename in enumerate(batch["fname"]):
            if args.limit > 0 and exported >= args.limit:
                break
            output_path = output_dir / str(filename)
            save_prediction_png(prediction[i : i + 1], output_path)
            exported_rows.append({"method": METHOD, "fname": str(filename), "output_path": str(output_path)})
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
        "residual_flow_ckpt": str(args.residual_flow_ckpt),
        "stage1_prediction_mode": stage1_prediction_mode,
        "stage1_detail_scale": stage1_detail_scale,
        "stage1_channels": stage1_channels,
        "flow_args": flow_args,
        "manifest": str(manifest_path),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"{METHOD}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
