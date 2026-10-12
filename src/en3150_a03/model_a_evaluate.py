"""Held-out evaluation for Model A checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from en3150_a03.model_a_metrics import classification_metrics
from en3150_a03.models.model_a import count_trainable_parameters, estimate_float32_size_kb


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """Evaluate test data and report classification scores and model footprint."""
    model.to(device)
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())
    result = classification_metrics(targets_all, predictions_all, class_names)
    parameter_count = count_trainable_parameters(model)
    result["trainable_parameters"] = parameter_count
    result["estimated_float32_size_kb"] = estimate_float32_size_kb(model)
    return result


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    """Load a weights-only checkpoint and return its training metadata."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=True
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Missing model_state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def save_evaluation(result: dict, output_path: Path) -> None:
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
