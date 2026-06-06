import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import T1FADataset


METHOD_PREFIX = {
    "cyclegan": "CycleGAN",
    "pix2pix": "Pix2Pix",
    "unet": "UNet",
    "ddim": "DDIM",
}


class ResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc: int = 1, output_nc: int = 1, ngf: int = 64, n_blocks: int = 6):
        super().__init__()
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        for i in range(2):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2**2
        for _ in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.conv(x))


class LegacyUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 3, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb_tensor = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb_tensor = x[:, None] * emb_tensor[None, :]
        return torch.cat((emb_tensor.sin(), emb_tensor.cos()), dim=-1)


class DDIMUNetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.time_mlp = nn.Linear(time_c, out_c)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t))[..., None, None]
        return F.silu(self.norm2(self.conv2(h)))


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, 3, channels, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * (channels**-0.5)
        attn = F.softmax(attn, dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(batch, channels, height, width)
        return x + self.proj(out)


class LegacyDDIMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(128)
        self.time_mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.inc = nn.Conv2d(2, 64, 3, padding=1)
        self.down1 = DDIMUNetBlock(64, 128, 256)
        self.down2 = DDIMUNetBlock(128, 256, 256)
        self.down3 = DDIMUNetBlock(256, 512, 256)
        self.attn = AttentionBlock(512)
        self.up1 = DDIMUNetBlock(512 + 256, 256, 256)
        self.up2 = DDIMUNetBlock(256 + 128, 128, 256)
        self.up3 = DDIMUNetBlock(128 + 64, 64, 256)
        self.outc = nn.Conv2d(64, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, noisy_fa: torch.Tensor, t1_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        x = torch.cat([noisy_fa, t1_cond], dim=1)
        x0 = self.inc(x)
        x1 = self.down1(self.pool(x0), t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)
        x3 = self.attn(x3)
        u1 = self.up1(torch.cat([self.up(x3), x2], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.up(u1), x1], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.up(u2), x0], dim=1), t_emb)
        return self.outc(u3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export legacy baseline checkpoints as common grayscale PNG predictions."
    )
    parser.add_argument("--model_type", required=True, choices=sorted(METHOD_PREFIX))
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--output_dir", default="", help="Prediction folder. Defaults to outputs/icdm2026/predictions/METHOD_CKPT.")
    parser.add_argument("--method", default="", help="Method name stored in export_summary.json.")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test limit. 0 exports all slices.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=50, help="DDIM sampling steps.")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="DDIM training timestep count.")
    return parser.parse_args()


def strip_state_dict_prefix(state_dict: dict[str, Any], prefix: str = "_orig_mod.") -> dict[str, Any]:
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def resolve_method_name(method: str, model_type: str, ckpt: str | Path) -> str:
    if method:
        return method
    prefix = METHOD_PREFIX[model_type]
    return f"{prefix}_{Path(ckpt).stem}"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    method = resolve_method_name(args.method, args.model_type, args.ckpt)
    return Path("outputs/icdm2026/predictions") / method


def tensor_to_gray01(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] > 1:
        tensor = tensor.mean(dim=1, keepdim=True)
    return ((tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def write_png01(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (np.asarray(image, dtype=np.float32).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image_u8)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def _load_raw_state_dict(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a state_dict-like object, got {type(checkpoint)!r}")
    for key in ("state_dict", "model", "model_state_dict", "generator", "G", "netG"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_model(model_type: str, ckpt: str | Path, device: torch.device) -> torch.nn.Module:
    if model_type == "cyclegan":
        model = ResNetGenerator(input_nc=1, output_nc=1, n_blocks=6)
    elif model_type == "pix2pix":
        model = ResNetGenerator(input_nc=1, output_nc=1, n_blocks=9)
    elif model_type == "unet":
        model = LegacyUNet(n_channels=3, n_classes=3)
    elif model_type == "ddim":
        model = LegacyDDIMUNet()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state_dict = strip_state_dict_prefix(_load_raw_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    serious_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if serious_missing or unexpected:
        raise RuntimeError(
            f"Failed to load {model_type} checkpoint. missing={serious_missing[:8]} unexpected={unexpected[:8]}"
        )
    return model.to(device).eval()


def _ddim_sample(
    model: torch.nn.Module,
    t1_gray: torch.Tensor,
    sample_steps: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, prediction_type="sample")
    scheduler.set_timesteps(sample_steps, device=t1_gray.device)
    sample = torch.randn_like(t1_gray)
    for timestep in scheduler.timesteps:
        t = torch.full((t1_gray.shape[0],), int(timestep.item()), device=t1_gray.device, dtype=torch.long)
        predicted_sample = model(sample, t1_gray, t)
        sample = scheduler.step(predicted_sample, timestep, sample).prev_sample
    return sample.clamp(-1.0, 1.0)


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    method = resolve_method_name(args.method, args.model_type, args.ckpt)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = T1FADataset(args.test_t1_dir, args.test_fa_dir, preload_ram=False)
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_model(args.model_type, args.ckpt, device)

    manifest_rows: list[dict[str, Any]] = []
    exported = 0
    with torch.inference_mode():
        for batch in loader:
            t1 = batch["t1_slice"].to(device)
            filenames = list(batch["fname"])
            if args.model_type == "unet":
                pred = model(t1)
            elif args.model_type == "ddim":
                pred = _ddim_sample(
                    model,
                    t1_gray=t1.mean(dim=1, keepdim=True),
                    sample_steps=args.sample_steps,
                    num_train_timesteps=args.num_train_timesteps,
                )
            else:
                pred = model(t1.mean(dim=1, keepdim=True))
            pred01 = tensor_to_gray01(pred).detach().cpu().numpy()
            for idx, filename in enumerate(filenames):
                out_path = output_dir / filename
                write_png01(out_path, pred01[idx, 0])
                manifest_rows.append({"filename": filename, "prediction_path": str(out_path)})
                exported += 1

    import pandas as pd

    manifest_path = output_dir / "export_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    summary = {
        "method": method,
        "model_type": args.model_type,
        "checkpoint": str(args.ckpt),
        "output_dir": str(output_dir),
        "exported": exported,
        "sample_steps": args.sample_steps if args.model_type == "ddim" else None,
        "num_train_timesteps": args.num_train_timesteps if args.model_type == "ddim" else None,
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"{method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
