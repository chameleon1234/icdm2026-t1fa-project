import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_rgb_to_single_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def center_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    center = x.shape[1] // 2
    return x[:, center : center + 1]


def single_channel_laplacian(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected B1HW tensor, got shape {tuple(x.shape)}")
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def prepare_stage1_input(x: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    if expected_channels == 1:
        return reduce_rgb_to_single_channel(x)
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"Stage 1 expects {expected_channels} input channels, got tensor shape {tuple(x.shape)}"
        )
    return x


def checkpoint_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def infer_stage1_in_channels(checkpoint: Any) -> int:
    if isinstance(checkpoint, dict):
        args = checkpoint.get("args", {})
        if "context_slices" in args:
            return int(args["context_slices"])
    state_dict = checkpoint_state_dict(checkpoint)
    weight = state_dict.get("patch_embed.weight") if isinstance(state_dict, dict) else None
    if weight is None:
        return 1
    return int(weight.shape[1])


def _expand_time(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.view(-1, 1, 1, 1).to(dtype=ref.dtype, device=ref.device)
    if t.dim() == 4:
        return t.to(dtype=ref.dtype, device=ref.device)
    raise ValueError(f"Unsupported t shape: {tuple(t.shape)}")


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(window_size).float() - (window_size - 1) / 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
        return gauss / gauss.sum()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.dim() == 3:
            img1 = img1.unsqueeze(1)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(1)

        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        c1 = 0.01**2
        c2 = 0.03**2

        window = self.window.view(1, 1, -1, 1)
        window = window * window.transpose(2, 3)
        window = window.to(img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(
            F.conv2d(img1**2, window, padding=self.window_size // 2) - mu1_sq, min=0.0
        )
        sigma2_sq = torch.clamp(
            F.conv2d(img2**2, window, padding=self.window_size // 2) - mu2_sq, min=0.0
        )
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = self._gradients(pred)
        target_x, target_y = self._gradients(target)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MDTA(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class RestormerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, expansion_factor: float):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Stage1Net(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        expansion_factor: float = 2.66,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.encoder_level1 = nn.ModuleList(
            [RestormerBlock(dim, num_heads[0], expansion_factor) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim, dim * 2)
        self.encoder_level2 = nn.ModuleList(
            [RestormerBlock(dim * 2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2, dim * 4)
        self.encoder_level3 = nn.ModuleList(
            [RestormerBlock(dim * 4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4, dim * 8)
        self.bottleneck = nn.ModuleList(
            [RestormerBlock(dim * 8, num_heads[3], expansion_factor) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(dim * 8, dim * 4)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = nn.ModuleList(
            [RestormerBlock(dim * 4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4, dim * 2)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.ModuleList(
            [RestormerBlock(dim * 2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2, dim)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.decoder_level1 = nn.ModuleList(
            [RestormerBlock(dim, num_heads[0], expansion_factor) for _ in range(num_blocks[0])]
        )

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.patch_embed(x)
        for blk in self.encoder_level1:
            enc1 = blk(enc1)

        enc2 = self.down1_2(enc1)
        for blk in self.encoder_level2:
            enc2 = blk(enc2)

        enc3 = self.down2_3(enc2)
        for blk in self.encoder_level3:
            enc3 = blk(enc3)

        x = self.down3_4(enc3)
        for blk in self.bottleneck:
            x = blk(x)

        x = self.up4_3(x)
        x = self.reduce_chan_level3(torch.cat([x, enc3], dim=1))
        for blk in self.decoder_level3:
            x = blk(x)

        x = self.up3_2(x)
        x = self.reduce_chan_level2(torch.cat([x, enc2], dim=1))
        for blk in self.decoder_level2:
            x = blk(x)

        x = self.up2_1(x)
        x = self.reduce_chan_level1(torch.cat([x, enc1], dim=1))
        for blk in self.decoder_level1:
            x = blk(x)

        return self.output(x)


def infer_stage1_prediction_mode(checkpoint_args: Optional[dict]) -> str:
    if not checkpoint_args:
        return "absolute"
    mode = checkpoint_args.get("stage1_prediction_mode", "absolute")
    if mode not in {"absolute", "residual"}:
        return "absolute"
    return mode


STAGE2_CONDITION_MODES = {"none", "coarse", "t1", "coarse_t1", "coarse_t1_edge"}


def infer_stage2_condition_mode(checkpoint_args: Optional[dict]) -> str:
    if not checkpoint_args:
        return "coarse"
    mode = checkpoint_args.get("condition_mode")
    if mode in STAGE2_CONDITION_MODES:
        return str(mode)
    return "coarse" if checkpoint_args.get("condition_on_coarse", True) else "none"


def stage2_condition_channels(condition_mode: str, stage1_channels: int) -> int:
    if condition_mode not in STAGE2_CONDITION_MODES:
        raise ValueError(f"Unsupported Stage 2 condition mode: {condition_mode}")
    if condition_mode == "none":
        return 0
    if condition_mode == "coarse":
        return 1
    if condition_mode == "t1":
        return int(stage1_channels)
    if condition_mode == "coarse_t1":
        return 1 + int(stage1_channels)
    if condition_mode == "coarse_t1_edge":
        return 7 + int(stage1_channels)
    raise ValueError(f"Unsupported Stage 2 condition mode: {condition_mode}")


def build_stage2_condition(
    coarse: torch.Tensor,
    t1_img: torch.Tensor,
    condition_mode: str,
) -> Optional[torch.Tensor]:
    if condition_mode not in STAGE2_CONDITION_MODES:
        raise ValueError(f"Unsupported Stage 2 condition mode: {condition_mode}")
    if condition_mode == "none":
        return None
    if condition_mode == "coarse":
        return coarse
    if condition_mode == "t1":
        return t1_img
    if condition_mode == "coarse_t1":
        return torch.cat([coarse, t1_img], dim=1)

    center_t1 = center_channel(t1_img)
    stack_mean = t1_img.mean(dim=1, keepdim=True)
    stack_range = t1_img.max(dim=1, keepdim=True).values - t1_img.min(dim=1, keepdim=True).values
    t1_edge = single_channel_laplacian(center_t1)
    center_offset = center_t1 - stack_mean
    coarse_edge = single_channel_laplacian(coarse)
    coarse_residual = coarse - center_t1
    residual_edge = single_channel_laplacian(coarse_residual)
    return torch.cat(
        [
            coarse,
            t1_img,
            t1_edge,
            center_offset,
            stack_range,
            coarse_edge,
            coarse_residual,
            residual_edge,
        ],
        dim=1,
    )


def predict_stage1_fa(
    model: nn.Module,
    t1_img: torch.Tensor,
    clamp: bool = False,
    prediction_mode: str = "residual",
) -> torch.Tensor:
    pred = model(t1_img)
    if prediction_mode == "residual":
        coarse = center_channel(t1_img) + pred
    elif prediction_mode == "absolute":
        coarse = pred
    else:
        raise ValueError(f"Unsupported Stage 1 prediction mode: {prediction_mode}")
    if clamp:
        coarse = torch.clamp(coarse, -1.0, 1.0)
    return coarse


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 1000.0
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        time_term = self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term
        return F.silu(self.norm2(self.conv2(h)))


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(b, c, h, w)
        return x + self.proj(out)


class RefinementFlowUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        condition_channels: int = 0,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()
        total_in_channels = input_channels + condition_channels
        self.condition_channels = condition_channels
        self.time_emb = SinusoidalPosEmb(time_dim)
        hidden_time_dim = time_dim * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_time_dim),
            nn.SiLU(),
            nn.Linear(hidden_time_dim, hidden_time_dim),
        )

        self.inc = nn.Conv2d(total_in_channels, base_channels, 3, padding=1)
        self.down1 = UNetBlock(base_channels, base_channels * 2, hidden_time_dim)
        self.down2 = UNetBlock(base_channels * 2, base_channels * 4, hidden_time_dim)
        self.down3 = UNetBlock(base_channels * 4, base_channels * 8, hidden_time_dim)
        self.attn = AttentionBlock(base_channels * 8)
        self.up1 = UNetBlock(base_channels * 8 + base_channels * 4, base_channels * 4, hidden_time_dim)
        self.up2 = UNetBlock(base_channels * 4 + base_channels * 2, base_channels * 2, hidden_time_dim)
        self.up3 = UNetBlock(base_channels * 2 + base_channels, base_channels, hidden_time_dim)
        self.outc = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_channels > 0:
            if condition is None:
                raise ValueError("Condition tensor is required when condition_channels > 0")
            x_t = torch.cat([x_t, condition], dim=1)

        t_emb = self.time_mlp(self.time_emb(t))
        x0 = self.inc(x_t)
        x1 = self.down1(self.pool(x0), t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)
        x3 = self.attn(x3)

        u1 = self.up1(torch.cat([self.up(x3), x2], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.up(u1), x1], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.up(u2), x0], dim=1), t_emb)
        return self.outc(u3)


class DetailRefinementFlowUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        condition_channels: int = 0,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()
        total_in_channels = input_channels + condition_channels
        self.condition_channels = condition_channels
        self.time_emb = SinusoidalPosEmb(time_dim)
        hidden_time_dim = time_dim * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_time_dim),
            nn.SiLU(),
            nn.Linear(hidden_time_dim, hidden_time_dim),
        )

        self.inc = nn.Conv2d(total_in_channels, base_channels, 3, padding=1)
        self.down1 = UNetBlock(base_channels, base_channels * 2, hidden_time_dim)
        self.down2 = UNetBlock(base_channels * 2, base_channels * 4, hidden_time_dim)
        self.down3 = UNetBlock(base_channels * 4, base_channels * 8, hidden_time_dim)
        self.attn = AttentionBlock(base_channels * 8)
        self.up1 = UNetBlock(base_channels * 8 + base_channels * 4, base_channels * 4, hidden_time_dim)
        self.up2 = UNetBlock(base_channels * 4 + base_channels * 2, base_channels * 2, hidden_time_dim)
        self.up3 = UNetBlock(base_channels * 2 + base_channels, base_channels, hidden_time_dim)
        self.out_avg_velocity = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.out_detail = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.condition_channels > 0:
            if condition is None:
                raise ValueError("Condition tensor is required when condition_channels > 0")
            x_t = torch.cat([x_t, condition], dim=1)

        t_emb = self.time_mlp(self.time_emb(t))
        x0 = self.inc(x_t)
        x1 = self.down1(self.pool(x0), t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)
        x3 = self.attn(x3)

        u1 = self.up1(torch.cat([self.up(x3), x2], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.up(u1), x1], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.up(u2), x0], dim=1), t_emb)
        return self.out_avg_velocity(u3), self.out_detail(u3)


def compose_stage2_velocity(
    model_output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    t: torch.Tensor,
    detail_boost: float = 0.0,
) -> torch.Tensor:
    if not isinstance(model_output, tuple):
        return model_output
    v_avg, v_detail = model_output
    gate = 0.65 + 0.35 * (1.0 - _expand_time(t, v_avg))
    return v_avg + float(detail_boost) * gate * v_detail


def split_stage2_output(
    model_output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(model_output, tuple):
        return model_output
    return model_output, None


def build_xt(
    target: torch.Tensor,
    coarse: torch.Tensor,
    t: torch.Tensor,
    source_noise_std: float = 0.05,
    noise: Optional[torch.Tensor] = None,
    deterministic_source: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if deterministic_source:
        source = coarse
    else:
        if noise is None:
            noise = torch.randn_like(coarse)
        source = coarse + source_noise_std * noise
    t_view = _expand_time(t, target)
    x_t = (1.0 - t_view) * source + t_view * target
    v_target = target - source
    return x_t, source, v_target


def project_endpoint(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = _expand_time(t, x_t)
    return x_t + (1.0 - t_view) * v_pred


@torch.no_grad()
def euler_refine(
    model: nn.Module,
    coarse: torch.Tensor,
    num_steps: int,
    condition: Optional[torch.Tensor] = None,
    clamp: bool = True,
    detail_boost: float = 0.0,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    x_pred = coarse.clone()
    dt = 1.0 / num_steps
    batch_size = coarse.shape[0]
    device = coarse.device

    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=coarse.dtype)
        v_pred = compose_stage2_velocity(
            model(x_pred, t_tensor, condition=condition),
            t_tensor,
            detail_boost=detail_boost,
        )
        x_pred = x_pred + v_pred * dt

    if clamp:
        x_pred = torch.clamp(x_pred, -1.0, 1.0)
    return x_pred


def euler_refine_train(
    model: nn.Module,
    coarse: torch.Tensor,
    num_steps: int,
    condition: Optional[torch.Tensor] = None,
    clamp: bool = True,
    detail_boost: float = 0.0,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    x_pred = coarse
    dt = 1.0 / num_steps
    batch_size = coarse.shape[0]
    device = coarse.device

    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=coarse.dtype)
        v_pred = compose_stage2_velocity(
            model(x_pred, t_tensor, condition=condition),
            t_tensor,
            detail_boost=detail_boost,
        )
        x_pred = x_pred + v_pred * dt
        if clamp:
            x_pred = torch.clamp(x_pred, -1.0, 1.0)
    return x_pred
