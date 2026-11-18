"""Four-level 2D U-Net used by the segmentation baseline."""

from __future__ import annotations

import torch
from torch import nn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UNet2D(nn.Module):
    """Four-resolution-level U-Net returning one-channel logits."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_width: int = 16) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_width) < 1:
            raise ValueError("channel counts and base_width must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_width = base_width
        widths = [base_width * (2**level) for level in range(4)]
        self.encoder = nn.ModuleList([
            _DoubleConv(in_channels, widths[0]),
            _DoubleConv(widths[0], widths[1]),
            _DoubleConv(widths[1], widths[2]),
            _DoubleConv(widths[2], widths[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(widths[3], widths[3] * 2)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(widths[3] * 2, widths[3], 2, stride=2),
            nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2),
            nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2),
            nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2),
        ])
        self.decoder = nn.ModuleList([
            _DoubleConv(widths[3] * 2, widths[3]),
            _DoubleConv(widths[2] * 2, widths[2]),
            _DoubleConv(widths[1] * 2, widths[1]),
            _DoubleConv(widths[0] * 2, widths[0]),
        ])
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (N, C, H, W), got {tuple(x.shape)}")
        skips: list[torch.Tensor] = []
        for level, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if level < len(self.encoder) - 1:
                x = self.pool(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("input spatial dimensions must be divisible by 16")
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return torch.sigmoid(self(x)) >= threshold
