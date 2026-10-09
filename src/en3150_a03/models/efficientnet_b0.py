"""EfficientNet-B0 for the 10-class EuroSAT RGB task (64 x 64 inputs)."""

from torch import nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Replace the ImageNet classifier and fine-tune the whole network."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_transforms(train: bool = False) -> transforms.Compose:
    """Keep the original 64 x 64 resolution; normalize for ImageNet weights."""
    steps = []
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(
        mean=EfficientNet_B0_Weights.DEFAULT.transforms().mean,
        std=EfficientNet_B0_Weights.DEFAULT.transforms().std,
    )]
    return transforms.Compose(steps)
