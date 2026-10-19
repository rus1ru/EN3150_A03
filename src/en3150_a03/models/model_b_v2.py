"""Custom sub-100k CNN inspired by MobileNetV2 and SE/MobileNetV3.

References: Sandler et al., CVPR 2018 (arXiv:1801.04381);
Hu et al., CVPR 2018 (arXiv:1709.01507); Howard et al., ICCV 2019
(arXiv:1905.02244). This is a new small configuration, trained from scratch,
not a pretrained MobileNet or a reproduction of a paper's benchmark results.
"""
from __future__ import annotations
import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, activation=True):
        layers = [nn.Conv2d(cin, cout, kernel, stride, kernel//2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """Reduced channel gating; hard sigmoid avoids an exponential at inference."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 8)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Hardsigmoid())

    def forward(self, x):
        return x * self.gate(x)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride=1, attention=False):
        super().__init__()
        expanded = 2 * cin
        self.transform = nn.Sequential(
            ConvNormAct(cin, expanded),
            ConvNormAct(expanded, expanded, 3, stride, groups=expanded),
            SqueezeExcitation(expanded) if attention else nn.Identity(),
            ConvNormAct(expanded, cout, activation=False),
        )
        self.use_residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.transform(x)
        return x + y if self.use_residual else y


class ModelBV2(nn.Module):
    """64x64 RGB -> 32x32 -> 16x16 -> 8x8 -> GAP -> ten logits.

    All spatial convolutions are depthwise. A pointwise RGB lift before the
    first spatial filter avoids v1's one-filter-per-raw-colour bottleneck.
    Seven expanded residual blocks and two SE gates use 96,538 parameters.
    The no-attention ablation has 87,546 parameters. No pretrained weights.
    """
    def __init__(self, num_classes=10, attention=True, dropout=0.1):
        super().__init__()
        if num_classes < 2 or not 0 <= dropout < 1:
            raise ValueError("Require >=2 classes and 0 <= dropout < 1")
        self.stem = nn.Sequential(ConvNormAct(3, 24), ConvNormAct(24, 24, 3, 2, groups=24))
        self.blocks = nn.Sequential(
            InvertedResidual(24, 24),
            InvertedResidual(24, 32),
            InvertedResidual(32, 32),
            InvertedResidual(32, 48, stride=2),
            InvertedResidual(48, 48, attention=attention),
            InvertedResidual(48, 80, stride=2),
            InvertedResidual(80, 80, attention=attention),
        )
        self.head = nn.Sequential(ConvNormAct(80, 192), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Dropout(dropout), nn.Linear(192, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
        if sum(p.numel() for p in self.parameters()) > 100_000:
            raise ValueError("Model B v2 exceeds the 100,000-parameter budget")

    def forward(self, images):
        return self.head(self.blocks(self.stem(images)))
