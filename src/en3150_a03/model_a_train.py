"""Model A training and validation utilities for EuroSAT."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


def make_optimizer(model: nn.Module, name: str, learning_rate: float,
                   weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Build an optimizer from the documented Model A comparison settings."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Cannot calculate loss for an empty data loader")
    return total_loss / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    optimizer_name: str,
    learning_rate: float,
) -> dict:
    """Train for the requested epochs and save the validation-best checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        _synchronize(device)
        started = time.perf_counter()
        train_loss = _epoch_loss(model, train_loader, criterion, device, optimizer)
        val_loss = _epoch_loss(model, val_loader, criterion, device, None)
        _synchronize(device)
        epoch_seconds = time.perf_counter() - started
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
        })
        print(
            f"{optimizer_name} epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"time={epoch_seconds:.2f}s",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    _save_loss_curve(history, output_dir / "loss_curves.png", optimizer_name)
    result = {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "mean_epoch_seconds": sum(float(row["epoch_seconds"]) for row in history) / epochs,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "loss_curve": str(output_dir / "loss_curves.png"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _save_loss_curve(history: list[dict[str, float | int]], path: Path, label: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="Training")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"Model A loss curves — {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
