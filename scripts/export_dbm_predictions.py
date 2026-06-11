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

from scripts.export_legacy_baseline_predictions import write_png01
from src.datasets import T1FADataset
from train_baseline_dbm import DBM_UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Diffusion Bridge Model predictions as common grayscale PNGs.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="DBM")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--gamma_max", type=float, default=0.125)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_state_dict(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


@torch.no_grad()
def dbm_sample(
    model: DBM_UNet,
    t1_img: torch.Tensor,
    *,
    sample_steps: int,
    gamma_max: float,
) -> torch.Tensor:
    batch_size = t1_img.shape[0]
    device = t1_img.device
    timesteps = torch.linspace(0.999, 0.0, max(sample_steps, 1), device=device)
    x_t = t1_img.clone()
    for idx, t_curr in enumerate(timesteps):
        t_next = timesteps[idx + 1] if idx < len(timesteps) - 1 else torch.tensor(0.0, device=device)
        t_tensor = torch.full((batch_size,), float(t_curr.item()), device=device)

        a_curr = 1.0 - t_curr
        b_curr = t_curr
        g_curr = 2.0 * gamma_max * torch.sqrt(t_curr * (1.0 - t_curr))

        x0_pred = model(x_t, t1_img, t_tensor)
        z_hat = (x_t - a_curr * x0_pred - b_curr * t1_img) / (g_curr + 1e-8)

        a_next = 1.0 - t_next
        b_next = t_next
        g_next = 2.0 * gamma_max * torch.sqrt(t_next * (1.0 - t_next))
        x_t = a_next * x0_pred + b_next * t1_img + g_next * z_hat
    return x_t.clamp(-1.0, 1.0)


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = DBM_UNet().to(device)
    model.load_state_dict(_load_state_dict(args.ckpt), strict=True)
    model.eval()

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
        pred = dbm_sample(model, t1, sample_steps=args.sample_steps, gamma_max=args.gamma_max)
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
        "model_type": "dbm",
        "checkpoint": str(args.ckpt),
        "output_dir": str(output_dir),
        "exported": exported,
        "sample_steps": args.sample_steps,
        "gamma_max": args.gamma_max,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
