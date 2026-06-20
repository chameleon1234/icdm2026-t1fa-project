from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build horizontal T1/FA/Stage1 prediction comparison panels.")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--predictions_root", default="outputs/icdm2026/predictions")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "PM_STAGE1=PM_STAGE1",
            "LPIPS_GAN_5SLICE=PM_STAGE1_LPIPS_GAN_5SLICE_FINAL",
            "SINGLE_SHARP_FULL=PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_FULL",
        ],
        help="Method specs as LABEL=prediction_dir_name_or_path.",
    )
    parser.add_argument(
        "--priority",
        nargs="*",
        default=["sub-002_z020.png", "sub-011_z064.png", "sub-190_z048.png"],
        help="Slice names to put first if present.",
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--panel_size", type=int, default=144)
    parser.add_argument("--output_dir", default="outputs/icdm2026/figures/method_slices/STAGE1_HORIZONTAL_COMPARISON")
    return parser.parse_args()


def load_gray(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def parse_methods(specs: list[str], predictions_root: Path) -> list[tuple[str, Path]]:
    methods: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            label, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            label = Path(spec).name
        path = Path(raw_path)
        if not path.is_absolute():
            path = predictions_root / path
        if path.exists():
            methods.append((label, path))
        else:
            print(f"Skipping missing method: {label} -> {path}")
    return methods


def select_names(t1_dir: Path, fa_dir: Path, methods: list[tuple[str, Path]], priority: list[str], count: int) -> list[str]:
    candidates = sorted(
        p.name
        for p in t1_dir.glob("*.png")
        if (fa_dir / p.name).exists() and all((method_dir / p.name).exists() for _, method_dir in methods)
    )
    selected: list[str] = []
    for name in priority:
        if name in candidates and name not in selected:
            selected.append(name)
    for name in candidates:
        if name not in selected:
            selected.append(name)
        if len(selected) >= count:
            break
    return selected


def label_panel(image: Image.Image, label: str, panel_size: int) -> Image.Image:
    image = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
    label_h = 30
    canvas = Image.new("RGB", (panel_size, panel_size + label_h), "white")
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), label, fill="black")
    return canvas


def save_panel(parts: list[tuple[str, Image.Image]], out_path: Path, title: str, panel_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gap = 8
    title_h = 30
    panels = [label_panel(image, label, panel_size) for label, image in parts]
    width = len(panels) * panel_size + (len(panels) - 1) * gap
    height = title_h + panel_size + 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill="black")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel_size + gap
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    t1_dir = Path(args.test_t1_dir)
    fa_dir = Path(args.test_fa_dir)
    predictions_root = Path(args.predictions_root)
    output_dir = Path(args.output_dir)
    methods = parse_methods(args.methods, predictions_root)
    if not methods:
        raise SystemExit("No valid method prediction directories were found.")
    names = select_names(t1_dir, fa_dir, methods, args.priority, args.count)
    if not names:
        raise SystemExit("No matched slices were found across T1, FA, and method predictions.")

    for name in names:
        t1 = load_gray(t1_dir / name)
        fa = load_gray(fa_dir / name, t1.size)
        parts: list[tuple[str, Image.Image]] = [("T1", t1), ("FA_GT", fa)]
        for label, method_dir in methods:
            parts.append((label, load_gray(method_dir / name, t1.size)))
        out_path = output_dir / f"{Path(name).stem}_stage1_horizontal.png"
        save_panel(parts, out_path, Path(name).stem, args.panel_size)

    print(f"Generated {len(names)} horizontal comparison panels.")
    print(f"Output dir: {output_dir}")
    print("Methods:")
    for label, method_dir in methods:
        print(f"  {label}: {method_dir}")


if __name__ == "__main__":
    main()
