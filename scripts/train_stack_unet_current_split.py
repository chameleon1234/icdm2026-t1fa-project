import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_unet_current_split import (
    _autocast_context,
    _collate,
    _limit_dataset,
    _psnr_from_mse,
    _to01,
    assert_disjoint_subject_splits,
    collect_subjects_from_slice_dir,
)
from src.data.t1fa_stack_dataset import T1FAStackDataset


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class StackUNet(nn.Module):
    """A fair 2.5D U-Net baseline: adjacent T1 slices in, center FA slice out."""

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = int(base_channels)
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.bottleneck = Down(c * 8, c * 8)
        self.up1 = Up(c * 8, c * 8, c * 4)
        self.up2 = Up(c * 4, c * 4, c * 2)
        self.up3 = Up(c * 2, c * 2, c)
        self.up4 = Up(c, c, c)
        self.outc = nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_dataset(t1_dir: str | Path, fa_dir: str | Path, context_slices: int) -> T1FAStackDataset:
    return T1FAStackDataset(t1_dir, fa_dir, context_slices=context_slices, target_size=(224, 224))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fair 2.5D Stack-UNet baseline on the current split.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--run_name", default="stack_unet5_current_split_e50")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context_slices", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    n_pixels = 0
    n_items = 0
    for batch in tqdm(loader, desc="StackUNet Val", leave=False):
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
    return {
        "psnr": _psnr_from_mse(mse),
        "mse": mse,
        "mae": total_l1 / max(n_pixels, 1),
        "n_items": float(n_items),
    }


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

    train_dataset = _limit_dataset(build_dataset(args.train_t1_dir, args.train_fa_dir, args.context_slices), args.train_limit)
    val_dataset = _limit_dataset(build_dataset(args.val_t1_dir, args.val_fa_dir, args.context_slices), args.val_limit)
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

    model = StackUNet(args.context_slices, 1, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    split_summary = {
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "context_slices": args.context_slices,
        "test_split_checked": True,
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_psnr = float("-inf")
    best_epoch = -1
    started = time.time()
    print(
        f"Stack-UNet current split training | train={len(train_dataset)} val={len(val_dataset)} "
        f"| subjects={len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)} "
        f"| context={args.context_slices} | width={args.width} | device={device} | mp={args.mixed_precision}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"StackUNet Epoch {epoch}/{args.epochs}")
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
            torch.save(model.state_dict(), ckpt_dir / "best_stack_unet.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"stack_unet_epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_l1": avg_loss, **val_metrics, "best": int(is_best)}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
        print(
            f"[StackUNet][Epoch {epoch}] train_l1={avg_loss:.6f} "
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
    print(f"Saved best checkpoint: {ckpt_dir / 'best_stack_unet.pt'}")
    return summary


if __name__ == "__main__":
    train(parse_args())
