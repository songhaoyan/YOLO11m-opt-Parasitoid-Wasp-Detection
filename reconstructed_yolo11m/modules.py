"""Reconstructed components for YOLO11m-opt.

Implementation choices not fully determined from paper:
- SPDConv: space-to-depth (2x2) followed by 1x1 Conv/BN/SiLU.
- DySample: point-sampling, LP style, scale=2, groups=4, no dyscope.
- PaperCBAM: canonical avg+max pooled channel attention and spatial attention.
These are plausible reconstructions, not recovered original code.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SPDConv(nn.Module):
    """Space-to-depth + convolution for a factor-of-two downsampling.

    Ultralytics parser will pass (c1, scaled_c2, kernel_size).
    Width and height must be even. YOLO inputs divisible by 32 satisfy this.
    """
    def __init__(self, c1: int, c2: int, k: int = 1):
        super().__init__()
        if k % 2 != 1:
            raise ValueError("SPDConv kernel size must be odd")
        self.space_to_depth = nn.PixelUnshuffle(2)
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=k, stride=1, padding=k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            raise ValueError(f"SPDConv expects even spatial dimensions; got {tuple(x.shape[-2:])}")
        return self.conv(self.space_to_depth(x))


class DySample(nn.Module):
    """Dynamic point-sampling upsampler (LP variant of DySample, ICCV 2023).

    Interface for YAML: - [-1, 1, DySample, [2, 4]]
    Parser supplies c1 as first argument. Output channels are unchanged.
    """
    def __init__(self, c1: int, scale: int = 2, groups: int = 4):
        super().__init__()
        if c1 % groups:
            raise ValueError(f"DySample: c1={c1} must be divisible by groups={groups}")
        if scale < 2:
            raise ValueError("DySample scale must be >=2")
        self.scale, self.groups = scale, groups
        self.offset = nn.Conv2d(c1, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        h = torch.arange(-(scale - 1) / 2, (scale - 1) / 2 + 1) / scale
        init_pos = torch.stack(torch.meshgrid(h, h, indexing="ij"))
        init_pos = init_pos.transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1)
        self.register_buffer("init_pos", init_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s, g = self.scale, self.groups
        offset = self.offset(x) * 0.25 + self.init_pos
        offset = offset.view(b, 2, -1, h, w)
        ys = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        xs = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((gx, gy), dim=0).view(1, 2, 1, h, w)
        norm = x.new_tensor([w, h]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / norm - 1.0
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), s)
        coords = coords.view(b, 2, -1, h * s, w * s).permute(0, 2, 3, 4, 1)
        coords = coords.contiguous().reshape(b * g, h * s, w * s, 2)
        return F.grid_sample(
            x.reshape(b * g, c // g, h, w), coords,
            mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(b, c, h * s, w * s)


class PaperCBAM(nn.Module):
    """Channel + spatial CBAM; preserves channels and spatial dimensions."""
    def __init__(self, c1: int, kernel_size: int = 7, reduction: int = 16):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM kernel_size must be 3 or 7")
        hidden = max(1, c1 // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = torch.sigmoid(
            self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        )
        x = x * channel
        spatial = torch.sigmoid(
            self.spatial(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1))
        )
        return x * spatial
