"""Generate missing report figures from saved EuroSAT run metrics, without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_run(run_dir: Path, model: str):
    if model == "efficientnet-b0":
        run = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if run["smoke_test"]:
            raise ValueError("Smoke-test metrics are not report results")
        metrics = run["test"]
        classes = metrics["class_order"]
        history = run["history"]
    else:
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        classes = metrics["class_names"]
        with (run_dir / "training_history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    return metrics, classes, history


def generate(run_dir: Path, model: str):
    run_dir = Path(run_dir)
    metrics, classes, history = load_run(run_dir, model)
    expected = json.loads((ROOT / "data/splits_torch_v1/metadata.json").read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]
    if classes != expected["classes"] or len(matrix) != len(classes) or any(
        len(row) != len(classes) or any(not isinstance(n, int) or n < 0 for n in row)
        for row in matrix
    ):
        raise ValueError("Confusion matrix/class order does not match the canonical split")
    if sum(map(sum, matrix)) != expected["counts"]["test"]:
        raise ValueError("Confusion matrix does not cover the full held-out test set")
    accuracy = sum(matrix[i][i] for i in range(len(classes))) / sum(map(sum, matrix))
    if abs(accuracy - metrics["accuracy"]) > 1e-12:
        raise ValueError("Confusion matrix and recorded accuracy disagree")
    if model == "efficientnet-b0":
        run = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if not history or len(history) != run["epochs"]:
            raise ValueError("Incomplete EfficientNet-B0 epoch history")
    files = [run_dir / "confusion_matrix.csv", run_dir / "confusion_matrix.png"]
    if model == "efficientnet-b0":
        files.append(run_dir / "training_curves.png")
    if any(path.exists() for path in files):
        raise FileExistsError("Refusing to overwrite existing report assets: " + ", ".join(
            str(path) for path in files if path.exists()))

    with files[0].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["actual / predicted", *classes])
        writer.writerows([name, *row] for name, row in zip(classes, matrix))

    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(heat, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted class",
           ylabel="True class", title=f"{model} held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    threshold = max(map(max, matrix)) / 2
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(files[1], dpi=160)
    plt.close(fig)

    if model == "efficientnet-b0":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training and validation loss")
        axes[0].legend()
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy (fraction)", title="Validation accuracy")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle("EfficientNet-B0 saved epoch history")
        fig.tight_layout()
        fig.savefig(files[2], dpi=160)
        plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("model-a", "efficientnet-b0"), required=True)
    args = parser.parse_args()
    for path in generate(args.run_dir, args.model):
        print(path)


if __name__ == "__main__":
    main()
