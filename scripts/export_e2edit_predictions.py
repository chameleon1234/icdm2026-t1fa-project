from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import T1FADataset
from pmrf_t1fa.e2edit import E2EDiTCorrector, E2EDiTStage1, make_wm_proxy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export E2EDiT M3 predictions.")
    p.add_argument("--checkpoint", required=True, help="Path to M3 best.pt checkpoint.")
    p.add_argument("--test_t1_dir", default="data/adni_processed/test/t1_slices")
    p.add_argument("--test_fa_dir", default="data/adni_processed/test/fa_slices")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=6)
    return p.parse_args()


def _to_png(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().squeeze().numpy()
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    stage1 = E2EDiTStage1(args.width, args.num_blocks).to(device)
    corrector = E2EDiTCorrector(args.width, args.num_blocks).to(device)
    stage1.load_state_dict(ckpt["stage1"])
    corrector.load_state_dict(ckpt["corrector"])
    stage1.eval()
    corrector.eval()
    dataset = T1FADataset(args.test_t1_dir, args.test_fa_dir, preload_ram=False)
    if args.limit and args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    exported = 0
    for batch in tqdm(loader, desc="Export E2EDiT"):
        t1 = batch["t1_slice"].to(device).float()
        if t1.shape[1] > 1:
            t1 = t1[:, :1]
        wm = make_wm_proxy(t1)
        s1 = stage1(t1)
        out = corrector(t1, s1["prior"], wm)
        for i, name in enumerate(batch["fname"]):
            cv2.imwrite(str(out_dir / name), _to_png(out["final"][i]))
            rows.append({"filename": name, "prediction": str(out_dir / name)})
            exported += 1
    with (out_dir / "export_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "E2EDiT",
        "checkpoint": args.checkpoint,
        "output_dir": str(out_dir),
        "exported": exported,
        "test_t1_dir": args.test_t1_dir,
        "test_fa_dir": args.test_fa_dir,
        "uses_a080": False,
        "uses_priorflow": False,
        "uses_template_source": False,
        "uses_offline_blend": False,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"E2EDiT exported={exported} output_dir={out_dir}")


if __name__ == "__main__":
    main()
