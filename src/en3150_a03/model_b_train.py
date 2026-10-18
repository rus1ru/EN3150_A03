"""Checkpoint, history and device utilities used by the final Model B trainer.

Module name is retained so completed-run source contracts remain valid.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import torch


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


def save_history(history, output):
    path = output / "training_history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)
