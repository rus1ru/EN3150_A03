"""A controlled depthwise-separable counterpart to the team's Model A."""
from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Sequential):
    """One spatial filter per input channel, then learned 1x1 channel mixing.

    A depthwise bias is redundant with the following pointwise bias because
    there is no intermediate nonlinearity. ReLU follows the full factorization.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


class ModelB(nn.Module):
    """3x64x64 -> 16x32x32 -> 32x16x16 -> 64x8x8 -> 64 -> 32 -> K.

    Widths, pooling and classifier match Model A to isolate convolution type.
    Ten classes require 5,589 trainable parameters (21.83 KiB of FP32 weights).
    This custom architecture uses established operations; it is not MobileNet.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.features = nn.Sequential(
            DepthwiseSeparableConv(3, 16),
            DepthwiseSeparableConv(16, 32),
            DepthwiseSeparableConv(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        if sum(p.numel() for p in self.parameters()) > 100_000:
            raise ValueError("Model B exceeds the assignment's 100,000-parameter cap")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
