import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def lowpass(image: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend low-frequency structure from a base method with high-frequency residual from a detail method."
    )
    parser.add_argument("--base_dir", required=True, help="Prediction folder providing stable low-frequency image.")
    parser.add_argument("--detail_dir", required=True, help="Prediction folder providing detail/high-frequency image.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.25, help="High-frequency residual blend strength.")
    parser.add_argument("--kernel", type=int, default=9, help="Gaussian kernel for low/high-frequency split.")
    parser.add_argument("--clip_delta", type=float, default=0.06, help="Clamp high-frequency delta before adding.")
    parser.add_argument("--manifest_name", default="blend_manifest.csv")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    detail_dir = Path(args.detail_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    exported = 0
    for base_path in sorted(base_dir.glob("*.png")):
        detail_path = detail_dir / base_path.name
        if not detail_path.exists():
            continue

        base = load_gray(base_path)
        detail = load_gray(detail_path)
        base_hp = base - lowpass(base, args.kernel)
        detail_hp = detail - lowpass(detail, args.kernel)
        hp_delta = np.clip(detail_hp - base_hp, -args.clip_delta, args.clip_delta)
        blended = base + args.alpha * hp_delta

        save_gray(output_dir / base_path.name, blended)
        rows.append(
            {
                "filename": base_path.name,
                "base": str(base_path),
                "detail": str(detail_path),
                "alpha": args.alpha,
                "kernel": args.kernel,
                "clip_delta": args.clip_delta,
            }
        )
        exported += 1

    with open(output_dir / args.manifest_name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "base", "detail", "alpha", "kernel", "clip_delta"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"exported={exported} output_dir={output_dir}")


if __name__ == "__main__":
    main()
