"""ImageNet-pretrained MobileNetV2 with a EuroSAT classification head."""
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(num_classes=10, pretrained=True):
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def set_training_stage(model, head_only):
    """Freeze the backbone for warm-up; unfreeze it for fine-tuning."""
    for parameter in model.features.parameters():
        parameter.requires_grad = not head_only
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_train_mode(model, head_only):
    model.train()
    if head_only:
        # Frozen backbone must also retain its pretrained BatchNorm statistics.
        model.features.eval()
