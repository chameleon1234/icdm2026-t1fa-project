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
from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
    FidelityCorrector,
    compose_frequency_preserving_output,
    sample_fidelity_correction,
)
from scripts.export_pm_dirf_predictions import load_stage1, make_slice_dataset, save_prediction_png


METHOD = "PM_DIRF_FIDELITY_CORRECTOR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frequency-preserving fidelity-corrector predictions.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument(
        "--stage1_ckpt",
        default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt",
    )
    parser.add_argument("--fidelity_corrector_ckpt", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--test_t1_dir", default="")
    parser.add_argument("--test_fa_dir", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps_override", type=int, default=0)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_output_dir(predictions_root: str | Path, output_dir: str | Path) -> Path:
    return Path(output_dir) if output_dir else Path(predictions_root) / METHOD


def load_fidelity_corrector(
    checkpoint_path: str | Path,
    device: torch.device,
    stage1_channels: int,
) -> tuple[FidelityCorrector, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    mode = str(ckpt_args.get("corrector_mode", "flow"))
    width = int(ckpt_args.get("width", 48))
    num_blocks = int(ckpt_args.get("num_blocks", 8))
    model = FidelityCorrector(stage1_channels, mode=mode, width=width, num_blocks=num_blocks)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    settings = {
        "corrector_mode": mode,
        "width": width,
        "num_blocks": num_blocks,
        "frequency_cutoff": float(ckpt_args.get("frequency_cutoff", 0.12)),
        "frequency_transition": float(ckpt_args.get("frequency_transition", 0.04)),
        "eval_steps": int(ckpt_args.get("eval_steps", 4)),
    }
    return model, settings


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
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1(args.stage1_ckpt, device)
    corrector, settings = load_fidelity_corrector(args.fidelity_corrector_ckpt, device, stage1_channels)
    if args.eval_steps_override > 0:
        settings["eval_steps"] = int(args.eval_steps_override)
    dataset = make_slice_dataset(test_t1_dir, test_fa_dir, stage1_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    rows: list[dict[str, Any]] = []
    exported = 0
    for batch in tqdm(loader, desc=f"Exporting {METHOD}", leave=False):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        coarse = predict_stage1_fa(
            stage1,
            t1_img,
            clamp=True,
            prediction_mode=prediction_mode,
            detail_scale=detail_scale,
        )
        raw_correction = sample_fidelity_correction(
            corrector,
            t1_img,
            coarse,
            settings["frequency_cutoff"],
            settings["frequency_transition"],
            settings["eval_steps"],
        )
        prediction = compose_frequency_preserving_output(
            coarse,
            raw_correction,
            settings["frequency_cutoff"],
            settings["frequency_transition"],
        )
        for index, filename in enumerate(batch["fname"]):
            if args.limit > 0 and exported >= args.limit:
                break
            output_path = output_dir / str(filename)
            save_prediction_png(prediction[index : index + 1], output_path)
            rows.append({"method": METHOD, "fname": str(filename), "output_path": str(output_path)})
            exported += 1
        if args.limit > 0 and exported >= args.limit:
            break
    manifest_path = output_dir / "export_manifest.csv"
    _write_manifest(manifest_path, rows)
    summary = {
        "method": METHOD,
        "output_dir": str(output_dir),
        "exported": exported,
        "stage1_ckpt": str(args.stage1_ckpt),
        "fidelity_corrector_ckpt": str(args.fidelity_corrector_ckpt),
        "stage1_channels": stage1_channels,
        "stage1_prediction_mode": prediction_mode,
        "stage1_detail_scale": detail_scale,
        "corrector_settings": settings,
        "manifest": str(manifest_path),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"{METHOD}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
