import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_legacy_baseline_predictions import strip_state_dict_prefix, tensor_to_gray01, write_png01
from scripts.train_restormer_current_split import OutputActivatedModel, _parse_int_tuple
from src.data.t1fa_stack_dataset import T1FAStackDataset
from train_baseline_restormer import RestormerBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Restormer current-split predictions as common grayscale PNGs.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="Restormer")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--dim", type=int, default=0, help="0 reads from train_summary.json if available.")
    parser.add_argument("--num_blocks", default="", help="Comma-separated blocks; empty reads from train_summary.json.")
    parser.add_argument("--num_heads", default="", help="Comma-separated heads; empty reads from train_summary.json.")
    parser.add_argument("--expansion_factor", type=float, default=0.0, help="0 reads from train_summary.json.")
    parser.add_argument("--output_activation", default="", choices=["", "tanh", "linear"], help="Empty reads from train_summary.json.")
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
        "dim": int(args.get("dim", 48)),
        "num_blocks": args.get("num_blocks", "2,2,2,2"),
        "num_heads": args.get("num_heads", "1,2,4,8"),
        "expansion_factor": float(args.get("expansion_factor", 2.66)),
        "output_activation": args.get("output_activation", "tanh"),
        "train_summary": str(summary_path),
    }


def load_restormer_model(
    ckpt: str | Path,
    device: torch.device,
    dim: int = 0,
    num_blocks: str = "",
    num_heads: str = "",
    expansion_factor: float = 0.0,
    output_activation: str = "",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    meta = _read_training_meta(ckpt)
    resolved_dim = int(dim or meta.get("dim") or 48)
    resolved_blocks = str(num_blocks or meta.get("num_blocks") or "2,2,2,2")
    resolved_heads = str(num_heads or meta.get("num_heads") or "1,2,4,8")
    resolved_expansion = float(expansion_factor or meta.get("expansion_factor") or 2.66)
    resolved_activation = str(output_activation or meta.get("output_activation") or "linear")
    base_model = RestormerBaseline(
        in_channels=1,
        out_channels=1,
        dim=resolved_dim,
        num_blocks=_parse_int_tuple(resolved_blocks),
        num_heads=_parse_int_tuple(resolved_heads),
        expansion_factor=resolved_expansion,
    )
    model = OutputActivatedModel(base_model, resolved_activation)
    state_dict = strip_state_dict_prefix(_load_raw_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing and not any(key.startswith("model.") for key in state_dict):
        missing_base, unexpected_base = base_model.load_state_dict(state_dict, strict=False)
        if not missing_base and not unexpected_base:
            missing, unexpected = [], []
    serious_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if serious_missing or unexpected:
        raise RuntimeError(
            f"Failed to load Restormer checkpoint. missing={serious_missing[:8]} unexpected={unexpected[:8]}"
        )
    out_meta = {
        **meta,
        "dim": resolved_dim,
        "num_blocks": resolved_blocks,
        "num_heads": resolved_heads,
        "expansion_factor": resolved_expansion,
        "output_activation": resolved_activation,
        "context_slices": 1,
    }
    return model.to(device).eval(), out_meta


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, model_meta = load_restormer_model(
        args.ckpt,
        device,
        args.dim,
        args.num_blocks,
        args.num_heads,
        args.expansion_factor,
        args.output_activation,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = T1FAStackDataset(args.test_t1_dir, args.test_fa_dir, context_slices=1, target_size=(224, 224))
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

    manifest_path = output_dir / "export_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    summary = {
        "method": args.method,
        "model_type": "restormer_current_split",
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
