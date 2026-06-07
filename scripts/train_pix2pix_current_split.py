import argparse
import json
import math
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

from scripts.export_legacy_baseline_predictions import ResNetBlock, ResNetGenerator
from src.datasets import T1FADataset


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 2, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        model: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for layer_idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_idx, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fair Pix2Pix baseline on the current ICDM train/val split.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--run_name", default="pix2pix_current_split_e100")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    return parser.parse_args(argv)


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


def _gray(batch_tensor: torch.Tensor) -> torch.Tensor:
    return batch_tensor.mean(dim=1, keepdim=True) if batch_tensor.shape[1] != 1 else batch_tensor


def _to01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(generator: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    generator.eval()
    total_l1 = 0.0
    total_mse = 0.0
    n_pixels = 0
    n_items = 0
    for batch in tqdm(loader, desc="Pix2Pix Val", leave=False):
        if batch is None:
            continue
        t1 = _gray(batch["t1_slice"].to(device, non_blocking=True))
        fa = _gray(batch["fa_slice"].to(device, non_blocking=True))
        pred = generator(t1)
        diff = _to01(pred) - _to01(fa)
        total_l1 += diff.abs().sum().item()
        total_mse += (diff * diff).sum().item()
        n_pixels += diff.numel()
        n_items += t1.shape[0]
    mse = total_mse / max(n_pixels, 1)
    mae = total_l1 / max(n_pixels, 1)
    return {"psnr": _psnr_from_mse(mse), "mse": mse, "mae": mae, "n_items": float(n_items)}


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

    train_dataset = _limit_dataset(T1FADataset(args.train_t1_dir, args.train_fa_dir, preload_ram=False), args.train_limit)
    val_dataset = _limit_dataset(T1FADataset(args.val_t1_dir, args.val_fa_dir, preload_ram=False), args.val_limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    generator = ResNetGenerator(input_nc=1, output_nc=1, ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    discriminator = ConditionalDiscriminator(input_nc=2, ndf=args.ndf).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
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
        f"Pix2Pix current split training | train={len(train_dataset)} val={len(val_dataset)} "
        f"| subjects={len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)} "
        f"| device={device} | mp={args.mixed_precision} | lambda_l1={args.lambda_l1}"
    )

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_g = 0.0
        total_d = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"Pix2Pix Epoch {epoch}/{args.epochs}")
        for batch in progress:
            if batch is None:
                continue
            real_t1 = _gray(batch["t1_slice"].to(device, non_blocking=True))
            real_fa = _gray(batch["fa_slice"].to(device, non_blocking=True))

            opt_d.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                fake_fa = generator(real_t1)
                pred_real = discriminator(torch.cat([real_t1, real_fa], dim=1))
                pred_fake = discriminator(torch.cat([real_t1, fake_fa.detach()], dim=1))
                loss_d = 0.5 * (
                    gan_loss(pred_real, torch.ones_like(pred_real))
                    + gan_loss(pred_fake, torch.zeros_like(pred_fake))
                )
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            opt_g.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                fake_fa = generator(real_t1)
                pred_fake_for_g = discriminator(torch.cat([real_t1, fake_fa], dim=1))
                loss_g_gan = gan_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                loss_g_l1 = l1_loss(fake_fa, real_fa)
                loss_g = loss_g_gan + args.lambda_l1 * loss_g_l1
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            total_g += float(loss_g.item())
            total_d += float(loss_d.item())
            steps += 1
            progress.set_postfix(g=f"{loss_g.item():.4f}", d=f"{loss_d.item():.4f}")

        val_metrics = evaluate(generator, val_loader, device)
        avg_g = total_g / max(steps, 1)
        avg_d = total_d / max(steps, 1)
        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
            best_epoch = epoch
            torch.save(generator.state_dict(), ckpt_dir / "best_pix2pix_generator.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(generator.state_dict(), ckpt_dir / f"pix2pix_generator_epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_g": avg_g, "train_d": avg_d, **val_metrics, "best": int(is_best)}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
        print(
            f"[Pix2Pix][Epoch {epoch}] train_G={avg_g:.5f} train_D={avg_d:.5f} "
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
    print(f"Saved best checkpoint: {ckpt_dir / 'best_pix2pix_generator.pt'}")
    return summary


if __name__ == "__main__":
    train(parse_args())
