"""Device selection for Model A training and evaluation commands."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device and reject unavailable CUDA instead of falling back."""
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {requested!r}") from exc

    if device.type != "cuda":
        return device
    if torch.version.cuda is None:
        raise RuntimeError(
            f"CUDA was requested, but installed PyTorch {torch.__version__} is CPU-only. "
            "Install a CUDA-enabled PyTorch build in this Python environment."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver and CUDA-enabled PyTorch installation."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )
    torch.cuda.set_device(device)
    return device
