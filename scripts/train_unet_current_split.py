import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_legacy_baseline_predictions import LegacyUNet
from src.datasets import T1FADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fair U-Net baseline on the current ICDM train/val split.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--run_name", default="unet_current_split_e100")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    return parser.parse_args()


def collect_subjects_from_slice_dir(slice_dir: str | Path) -> set[str]:
    return {path.name.split("_z")[0] for path in Path(slice_dir).glob("*.png")}


def assert_disjoint_subject_splits(train_subjects: set[str], val_subjects: set[str], test_subjects: set[str]) -> None:
    checks = [
        ("train/val", train_subjects & val_subjects),
        ("train/test", train_subjects & test_subjects),
        ("val/test", val_subjects & test_subjects),
    ]
    bad = [(name, sorted(overlap)) for name, overlap in checks if overlap]
    if bad:
        details = "; ".join(f"{name}: {subjects[:8]}" for name, subjects in bad)
        raise ValueError(f"Subject leakage detected between splits: {details}")


def _limit_dataset(dataset: torch.utils.data.Dataset, limit: int) -> torch.utils.data.Dataset:
    if limit <= 0:
        return dataset
    return torch.utils.data.Subset(dataset, list(range(min(limit, len(dataset)))))


def _collate(batch: list[Any]) -> Any:
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return None
    return torch.utils.data.dataloader.default_collate(filtered)


def _to01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    import math

    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    n_pixels = 0
    n_items = 0
    for batch in tqdm(loader, desc="UNet Val", leave=False):
        if batch is None:
            continue
        t1 = batch["t1_slice"].to(device, non_blocking=True)
        fa = batch["fa_slice"].to(device, non_blocking=True)
        pred = model(t1)
        pred01 = _to01(pred)
        fa01 = _to01(fa)
        diff = pred01 - fa01
        total_l1 += diff.abs().sum().item()
        total_mse += (diff * diff).sum().item()
        n_pixels += diff.numel()
        n_items += t1.shape[0]
    mse = total_mse / max(n_pixels, 1)
    mae = total_l1 / max(n_pixels, 1)
    return {"psnr": _psnr_from_mse(mse), "mse": mse, "mae": mae, "n_items": float(n_items)}


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_subjects = collect_subjects_from_slice_dir(args.train_t1_dir)
    val_subjects = collect_subjects_from_slice_dir(args.val_t1_dir)
    test_subjects = collect_subjects_from_slice_dir(args.test_t1_dir)
    assert_disjoint_subject_splits(train_subjects, val_subjects, test_subjects)

    run_dir = Path(args.output_root) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _limit_dataset(T1FADataset(args.train_t1_dir, args.train_fa_dir, preload_ram=False), args.train_limit)
    val_dataset = _limit_dataset(T1FADataset(args.val_t1_dir, args.val_fa_dir, preload_ram=False), args.val_limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    model = LegacyUNet(n_channels=3, n_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    split_summary = {
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "test_split_checked": True,
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_psnr = float("-inf")
    best_epoch = -1
    started = time.time()
    print(
        f"U-Net current split training | train={len(train_dataset)} val={len(val_dataset)} "
        f"| subjects={len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)} "
        f"| device={device} | mp={args.mixed_precision}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"UNet Epoch {epoch}/{args.epochs}")
        for batch in progress:
            if batch is None:
                continue
            t1 = batch["t1_slice"].to(device, non_blocking=True)
            fa = batch["fa_slice"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                pred = model(t1)
                loss = criterion(pred, fa)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.5f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(steps, 1)
        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_unet.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"unet_epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_l1": avg_loss, **val_metrics, "best": int(is_best)}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
        print(
            f"[UNet][Epoch {epoch}] train_l1={avg_loss:.6f} "
            f"val_PSNR={val_metrics['psnr']:.4f} val_MAE={val_metrics['mae']:.6f} Best={int(is_best)}"
        )

    summary = {
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "elapsed_sec": time.time() - started,
        "args": vars(args),
        "split_summary": split_summary,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved best checkpoint: {ckpt_dir / 'best_unet.pt'}")
    return summary


if __name__ == "__main__":
    train(parse_args())
