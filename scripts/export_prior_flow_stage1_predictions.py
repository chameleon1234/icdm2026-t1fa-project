import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import (
    TemplatePriorFlowNet,
    center_channel,
    euler_sample_prior_flow,
    prepare_stage1_input,
)
from pmrf_t1fa.train_pmrf_t1fa_stage1 import (
    _batch_slice_ids,
    build_t1_anatomical_wm_prob_map,
    gather_template_batch_from_slice_ids,
    make_slice_dataset,
)
from pmrf_t1fa.train_pmrf_t1fa_stage1_prior_flow import normalize_z_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export template-guided prior-flow Stage1 predictions.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_t1_dir", default="data/adni_processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/adni_processed/test/fa_slices")
    parser.add_argument(
        "--flow_source",
        default="checkpoint",
        choices=["checkpoint", "template", "pred_folder", "t1"],
        help="Use checkpoint source setting or override it. Use t1 to start from the center T1 slice.",
    )
    parser.add_argument("--flow_source_pred_dir", default="", help="PNG folder used as x0 for pred_folder source.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flow_steps", type=int, default=-1, help="Use checkpoint flow_steps when -1.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _checkpoint_args(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {})
    return {}


def _parse_slice_id(filename: str) -> int:
    match = re.search(r"_z(\d+)\.png$", str(filename))
    if not match:
        raise ValueError(f"Cannot parse slice id from {filename}")
    return int(match.group(1))


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    image = torch.clamp((tensor.detach().cpu().squeeze().float() + 1.0) / 2.0, 0.0, 1.0)
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def _load_source_png(pred_dir: Path, fname: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    path = pred_dir / str(fname)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # cv2.imread can fail on non-ASCII paths on Windows; imdecode handles them.
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if img is None:
        raise FileNotFoundError(f"Cannot read flow source prediction: {path}")
    tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).view(1, img.shape[0], img.shape[1])
    return tensor.to(device=device, dtype=dtype)


def maybe_limit_dataset(dataset, limit: int):
    if int(limit) <= 0:
        return dataset
    return Subset(dataset, range(min(int(limit), len(dataset))))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = _checkpoint_args(checkpoint)
    context_slices = int(ckpt_args.get("context_slices", 1))
    width = int(ckpt_args.get("width", 48))
    num_blocks = int(ckpt_args.get("num_blocks", 8))
    flow_steps = int(args.flow_steps) if int(args.flow_steps) > 0 else int(ckpt_args.get("flow_steps", 4))
    flow_source = args.flow_source
    if flow_source == "checkpoint":
        flow_source = str(ckpt_args.get("flow_source", "template"))
    pred_dir_arg = args.flow_source_pred_dir or str(ckpt_args.get("flow_source_pred_dir", ""))
    pred_dir = Path(pred_dir_arg) if flow_source == "pred_folder" else None
    if flow_source == "pred_folder" and (pred_dir is None or not pred_dir.exists()):
        raise FileNotFoundError(f"flow_source_pred_dir does not exist: {pred_dir_arg}")
    model = TemplatePriorFlowNet(t1_channels=context_slices, width=width, num_blocks=num_blocks)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    template_by_slice = {int(k): v for k, v in checkpoint["stage1_template_by_slice"].items()}
    default_template = checkpoint["stage1_template_default"]
    z_min = float(min(template_by_slice.keys()))
    z_max = float(max(template_by_slice.keys()))
    dataset = maybe_limit_dataset(make_slice_dataset(args.test_t1_dir, args.test_fa_dir, context_slices), args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    exported = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Export prior-flow Stage1"):
            t1_img = prepare_stage1_input(batch["t1_slice"].to(device), context_slices).float()
            slice_ids = _batch_slice_ids(batch, device)
            if flow_source == "pred_folder":
                source = torch.cat(
                    [_load_source_png(pred_dir, str(fname), device, t1_img.dtype).unsqueeze(0) for fname in batch["fname"]],
                    dim=0,
                )
            elif flow_source == "t1":
                source = center_channel(t1_img)
            else:
                source = gather_template_batch_from_slice_ids(template_by_slice, slice_ids, t1_img, default_template)
            wm_prob = build_t1_anatomical_wm_prob_map(t1_img).to(device=device, dtype=t1_img.dtype)
            z_map = normalize_z_map(slice_ids, t1_img, z_min, z_max)
            pred = euler_sample_prior_flow(model, source, t1_img, wm_prob, z_map, steps=flow_steps, clamp=False)
            pred = torch.clamp(pred, -1.0, 1.0)
            for i, fname in enumerate(batch["fname"]):
                name = str(fname)
                output_path = out_dir / name
                _save_png(output_path, pred[i : i + 1])
                manifest_rows.append(
                    {
                        "fname": name,
                        "prediction": str(output_path),
                        "slice_id": _parse_slice_id(name),
                        "flow_steps": flow_steps,
                        "flow_source": flow_source,
                    }
                )
                exported += 1
    with open(out_dir / "export_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fname", "prediction", "slice_id", "flow_steps", "flow_source"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"PRIOR_FLOW_STAGE1: exported={exported} output_dir={out_dir}")
    print(f"Saved manifest to: {out_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
