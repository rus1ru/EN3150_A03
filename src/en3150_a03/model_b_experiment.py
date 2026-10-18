"""Metric export and environment utilities for the final Model B workflow.

Module name is retained so completed-run source contracts remain valid.
"""
from __future__ import annotations
import csv
import math
import platform
import subprocess
from pathlib import Path
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]


def wilson_interval(correct, total):
    """95% Wilson binomial interval; tile independence is only an approximation."""
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [center - half, center + half]


def save_confusion(result, output):
    import matplotlib.pyplot as plt
    classes, matrix = result["class_names"], result["confusion_matrix"]
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["actual / predicted", *classes])
        writer.writerows([name, *row] for name, row in zip(classes, matrix))
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "precision", "recall", "support"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"class": name, **result["per_class"][name]} for name in classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(heat, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted class", ylabel="True class", title="Model B held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    threshold = max(map(max, matrix)) / 2
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def environment(device):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cpu_threads=torch.get_num_threads(), git_commit=commit, working_tree_dirty=dirty)
