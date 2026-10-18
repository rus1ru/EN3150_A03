"""Model B training with epoch-boundary recovery for Colab disconnections."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

# Reuse the team's optimizer definitions; do not alter Model A's implementation.
from en3150_a03.model_a_train import make_optimizer

OPTIMIZERS = {
    "adam": {"learning_rate": 0.001, "momentum": None},
    "sgd": {"learning_rate": 0.01, "momentum": None},
    "sgd_momentum": {"learning_rate": 0.01, "momentum": 0.9},
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value) -> None:
    temporary = path.with_name(path.stem + ".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.enable_grad() if training else torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; run stopped without replacing the checkpoint")
            if training:
                loss.backward()
                optimizer.step()
            count += targets.numel()
            loss_sum += loss.item() * targets.numel()
            correct += (logits.argmax(1) == targets).sum().item()
    if count == 0:
        raise ValueError("Cannot train/evaluate an empty loader")
    return loss_sum / count, correct / count


def save_history(history, output):
    path = output / "training_history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def save_curves(history, output, optimizer_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split, label in (("train", "Training"), ("val", "Validation")):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=label)
        ax.set(xlabel="Epoch", ylabel="Cross-entropy" if metric == "loss" else "Accuracy (fraction)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Model B - {optimizer_name}")
    fig.tight_layout()
    fig.savefig(output / "loss_curves.png", dpi=160)
    plt.close(fig)


def train_model(model, train_loader, val_loader, device, epochs, output_dir,
                optimizer_name, contract, resume=False):
    """Persist optimizer, shuffle and device RNG state after every whole epoch.

    `contract` binds checkpoints to architecture, preprocessing, class mapping,
    manifests and run settings. Only epoch-boundary continuation is supported.
    Full experiment entry points enforce >=20 epochs; tiny tests call this loop.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = OPTIMIZERS[optimizer_name]
    model.to(device)
    optimizer = make_optimizer(model, optimizer_name, settings["learning_rate"])
    history, best_loss, best_epoch = [], float("inf"), 0
    last_path, best_path = output / "last_model.pt", output / "best_model.pt"
    if last_path.exists():
        if not resume:
            raise FileExistsError("Existing checkpoint requires explicit resume")
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state["contract"] != contract or state["optimizer"] != optimizer_name:
            raise ValueError("Resume contract differs from checkpoint")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history, best_loss, best_epoch = state["history"], state["best_val_loss"], state["best_epoch"]
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        if train_loader.generator is not None:
            train_loader.generator.set_state(state["loader_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        if not best_path.is_file():
            raise FileNotFoundError("Best checkpoint missing from interrupted run")
    elif resume and any(output.iterdir()):
        raise FileNotFoundError("Incomplete run has no recoverable last_model.pt")

    for epoch in range(len(history) + 1, epochs + 1):
        synchronize(device)
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, device)
        synchronize(device)
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                   val_loss=val_loss, val_accuracy=val_accuracy,
                   epoch_seconds=time.perf_counter() - started)
        history.append(row)
        state = dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     optimizer=optimizer_name, epoch=epoch, val_loss=val_loss, contract=contract)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            atomic_checkpoint(best_path, state)
        state.update(optimizer_state_dict=optimizer.state_dict(), history=history,
                     best_val_loss=best_loss, best_epoch=best_epoch,
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     loader_rng=train_loader.generator.get_state() if train_loader.generator is not None else None,
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        atomic_checkpoint(last_path, state)
        save_history(history, output)
        print(f"{optimizer_name} {epoch:02}/{epochs}: train={train_loss:.5f} "
              f"val={val_loss:.5f} val_acc={val_accuracy:.4f} {row['epoch_seconds']:.2f}s", flush=True)
    save_history(history, output)
    save_curves(history, output, optimizer_name)
    summary = dict(optimizer=optimizer_name, **settings, epochs=epochs,
                   best_epoch=best_epoch, best_val_loss=best_loss,
                   mean_epoch_seconds=sum(r["epoch_seconds"] for r in history) / len(history),
                   checkpoint="best_model.pt", history_csv="training_history.csv", loss_curve="loss_curves.png",
                   timing_definition="Training plus validation; synchronized device; excludes checkpoint/plot writes")
    atomic_json(output / "training_summary.json", summary)
    return summary
