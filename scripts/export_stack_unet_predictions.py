import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_legacy_baseline_predictions import (
    strip_state_dict_prefix,
    tensor_to_gray01,
    write_png01,
)
from scripts.train_stack_unet_current_split import StackUNet
from src.data.t1fa_stack_dataset import T1FAStackDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 2.5D Stack-UNet predictions as common grayscale PNG files.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="StackUNet5")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--context_slices", type=int, default=0, help="0 reads from train_summary.json if available.")
    parser.add_argument("--width", type=int, default=0, help="0 reads from train_summary.json if available.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_raw_state_dict(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a state_dict-like object, got {type(checkpoint)!r}")
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _read_training_meta(ckpt: str | Path) -> dict[str, Any]:
    ckpt_path = Path(ckpt)
    summary_path = ckpt_path.parents[1] / "train_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    args = summary.get("args", {})
    return {
        "context_slices": int(args.get("context_slices", 0) or 0),
        "width": int(args.get("width", 0) or 0),
        "train_summary": str(summary_path),
    }


def load_stack_unet_model(
    ckpt: str | Path,
    device: torch.device,
    context_slices: int = 0,
    width: int = 0,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    meta = _read_training_meta(ckpt)
    resolved_context = int(context_slices or meta.get("context_slices") or 5)
    resolved_width = int(width or meta.get("width") or 32)
    model = StackUNet(in_channels=resolved_context, out_channels=1, base_channels=resolved_width)
    state_dict = strip_state_dict_prefix(_load_raw_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    serious_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if serious_missing or unexpected:
        raise RuntimeError(
            f"Failed to load StackUNet checkpoint. missing={serious_missing[:8]} unexpected={unexpected[:8]}"
        )
    out_meta = {
        **meta,
        "context_slices": resolved_context,
        "width": resolved_width,
    }
    return model.to(device).eval(), out_meta


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, model_meta = load_stack_unet_model(args.ckpt, device, args.context_slices, args.width)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = T1FAStackDataset(args.test_t1_dir, args.test_fa_dir, context_slices=model_meta["context_slices"])
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    manifest_rows: list[dict[str, Any]] = []
    exported = 0
    with torch.inference_mode():
        for batch in loader:
            t1 = batch["t1_slice"].to(device)
            pred = model(t1)
            pred01 = tensor_to_gray01(pred).detach().cpu().numpy()
            for idx, filename in enumerate(list(batch["fname"])):
                out_path = output_dir / filename
                write_png01(out_path, pred01[idx, 0])
                manifest_rows.append({"filename": filename, "prediction_path": str(out_path)})
                exported += 1

    import pandas as pd

    manifest_path = output_dir / "export_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    summary = {
        "method": args.method,
        "model_type": "stack_unet",
        "checkpoint": str(args.ckpt),
        "output_dir": str(output_dir),
        "exported": exported,
        **model_meta,
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"{args.method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
