import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTFM_ROOT = PROJECT_ROOT / "MOTFM-main"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MOTFM_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTFM_ROOT))

from scripts.export_legacy_baseline_predictions import write_png01
from src.datasets import T1FADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOTFM T1-to-FA predictions as common grayscale PNGs.")
    parser.add_argument("--config_path", default="MOTFM-main/configs/t1_fa_config.yaml")
    parser.add_argument("--ckpt", default="MOTFM-main/checkpoints_t1_fa/t1_fa_config/last.ckpt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="MOTFM")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_motfm_model(config_path: str | Path, ckpt: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from utils.general_utils import load_config
    from trainer import FlowMatchingLightningModule

    config = load_config(str(config_path))
    checkpoint = torch.load(ckpt, map_location="cpu")
    module = FlowMatchingLightningModule(config)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    serious_missing = [key for key in missing if not key.startswith("lpips_loss.")]
    if serious_missing or unexpected:
        raise RuntimeError(f"Failed to load MOTFM checkpoint. missing={serious_missing[:8]} unexpected={unexpected[:8]}")
    model = module.model.to(device).eval()
    meta = {
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "global_step": checkpoint.get("global_step") if isinstance(checkpoint, dict) else None,
        "config": str(config_path),
    }
    return model, meta


@torch.no_grad()
def euler_t1_to_fa(model: torch.nn.Module, t1: torch.Tensor, sample_steps: int) -> torch.Tensor:
    x = t1.clone()
    steps = max(sample_steps, 1)
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((x.shape[0],), step / steps, device=x.device)
        velocity = model(x=x, t=t, masks=None, cond=None)
        x = (x + velocity * dt).clamp(-1.0, 1.0)
    return x


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, model_meta = _load_motfm_model(args.config_path, args.ckpt, device)

    dataset = T1FADataset(args.test_t1_dir, args.test_fa_dir, preload_ram=False)
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    exported = 0
    for batch in loader:
        t1 = batch["t1_slice"].to(device).mean(dim=1, keepdim=True)
        pred = euler_t1_to_fa(model, t1, args.sample_steps)
        pred01 = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0).detach().cpu().numpy()
        for idx, filename in enumerate(list(batch["fname"])):
            out_path = output_dir / filename
            write_png01(out_path, pred01[idx, 0])
            rows.append({"filename": filename, "prediction_path": str(out_path)})
            exported += 1

    manifest_path = output_dir / "export_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    summary = {
        "method": args.method,
        "model_type": "motfm_t1_to_fa_euler",
        "checkpoint": str(args.ckpt),
        "output_dir": str(output_dir),
        "exported": exported,
        "sample_steps": args.sample_steps,
        **model_meta,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
