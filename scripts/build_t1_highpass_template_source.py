import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.train_pmrf_t1fa_stage1 import build_fa_slice_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FA-template + T1-highpass source PNGs for prior-flow training.")
    parser.add_argument("--t1_dir", required=True)
    parser.add_argument("--template_fa_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.25, help="T1 high-pass injection strength.")
    parser.add_argument("--kernel", type=int, default=13, help="Average-pool low-pass kernel for T1 high-pass.")
    parser.add_argument("--target_size", type=int, default=224)
    return parser.parse_args()


def parse_slice_id(filename: str) -> int:
    match = re.search(r"_z(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Cannot parse slice id from {filename}")
    return int(match.group(1))


def read_gray(path: Path, target_size: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).view(1, 1, target_size, target_size)


def write_gray(path: Path, tensor: torch.Tensor) -> None:
    image = torch.clamp((tensor.detach().cpu().squeeze().float() + 1.0) / 2.0, 0.0, 1.0)
    arr = (image.numpy() * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)


def main() -> None:
    args = parse_args()
    t1_dir = Path(args.t1_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template_by_slice, default_template = build_fa_slice_template(args.template_fa_dir, target_size=(args.target_size, args.target_size))
    rows = []
    files = sorted(t1_dir.glob("*.png"))
    for path in tqdm(files, desc="Build T1-highpass template source"):
        t1 = read_gray(path, args.target_size)
        slice_id = parse_slice_id(path.name)
        template = template_by_slice.get(slice_id, default_template).view_as(t1)
        t1_hp = t1 - lowpass(t1, args.kernel)
        source = torch.clamp(template + float(args.alpha) * t1_hp, -1.0, 1.0)
        out_path = out_dir / path.name
        write_gray(out_path, source)
        rows.append({"fname": path.name, "slice_id": slice_id, "source": str(out_path), "alpha": args.alpha, "kernel": args.kernel})
    with open(out_dir / "source_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fname", "slice_id", "source", "alpha", "kernel"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Built {len(rows)} source PNGs in {out_dir}")


if __name__ == "__main__":
    main()
