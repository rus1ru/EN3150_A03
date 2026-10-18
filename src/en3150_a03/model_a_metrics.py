"""Classification metrics used by the Model A experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def classification_metrics(
    targets: Sequence[int] | torch.Tensor,
    predictions: Sequence[int] | torch.Tensor,
    class_names: Sequence[str],
) -> dict:
    """Calculate accuracy, per-class precision/recall, macro scores and confusion."""
    target_tensor = torch.as_tensor(targets, dtype=torch.int64).reshape(-1)
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.int64).reshape(-1)
    if target_tensor.numel() == 0 or target_tensor.numel() != prediction_tensor.numel():
        raise ValueError("targets and predictions must be non-empty and equally sized")
    if not class_names:
        raise ValueError("class_names must not be empty")
    class_count = len(class_names)
    if (
        torch.any(target_tensor < 0)
        or torch.any(target_tensor >= class_count)
        or torch.any(prediction_tensor < 0)
        or torch.any(prediction_tensor >= class_count)
    ):
        raise ValueError("target or prediction index is outside class_names")

    encoded = target_tensor * class_count + prediction_tensor
    confusion = torch.bincount(encoded, minlength=class_count**2).reshape(
        class_count, class_count
    )
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "accuracy": float(true_positive.sum() / support.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(class_names),
    }
