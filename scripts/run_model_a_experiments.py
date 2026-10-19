"""Train and compare Model A with AdamW, SGD and SGD with momentum."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torchvision.transforms import ToTensor

from en3150_a03.data import get_dataloaders
from en3150_a03.model_a_evaluate import evaluate_model, load_checkpoint, save_evaluation
from en3150_a03.models.model_a import ModelA, count_trainable_parameters
from en3150_a03.model_a_train import make_optimizer, train_model

OPTIMIZERS = {
    "adamw": {"learning_rate": 0.001, "momentum": None, "weight_decay": 0.0001},
    "sgd": {"learning_rate": 0.01, "momentum": None},
    "sgd_momentum": {"learning_rate": 0.01, "momentum": 0.9},
}

OPTIMIZER_RATIONALES = {
    "adam": "Historical 20-epoch Adam baseline without weight decay; retained for reading earlier run reports.",
    "adamw": "Primary optimizer: decoupled weight decay (0.0001) and adaptive per-parameter updates, using the historical Model A Adam learning rate for a new 60-epoch run.",
    "sgd": "Plain-SGD reference using the same fixed learning rate throughout training.",
    "sgd_momentum": (
        "Momentum (0.9) accumulates a smoothed gradient direction. It can accelerate "
        "consistent descent and damp mini-batch oscillation; excessive momentum can "
        "overshoot, so compare its measured loss curves and test results."
    ),
}


def write_experiment_report(run_dir: Path, run_config: dict, results: list[dict]) -> Path:
    """Write model rationale and measured optimizer results as one shareable Markdown file."""
    report_path = run_dir / "experiment_report.md"
    lines = [
        "# Model A: EuroSAT experiment report",
        "",
        f"- Dataset SHA-256: `{run_config['dataset_sha256']}`",
        f"- Classes: {', '.join(run_config['class_names'])}",
        f"- Epochs per optimizer: {run_config['epochs_per_optimizer']}",
        f"- Batch size: {run_config['batch_size']}",
        f"- Seed: {run_config['seed']}",
        f"- Device: {run_config['device']}",
        f"- Hardware: {run_config['hardware']['processor']} "
        f"({run_config['hardware']['platform']})",
        f"- Python: {run_config['python_version'].split()[0]}",
        f"- PyTorch: {run_config['pytorch_version']}",
        "",
        "## Architecture and parameter count",
        "",
        "Input is a 3×64×64 RGB image. Each standard convolution uses a 3×3 kernel, "
        "stride 1, padding 1, and a bias; each is followed by ReLU and 2×2 max pooling "
        "with stride 2. The channel/spatial progression is "
        "`3×64×64 → 16×32×32 → 32×16×16 → 64×8×8`. Adaptive average pooling "
        "produces 64 features, followed by fully connected layers `64 → 32 → 10`.",
        "",
        "| Layer | Parameter calculation | Parameters |",
        "| --- | --- | ---: |",
        "| Conv2d 3→16 | 16 × (3 × 3 × 3 + 1) | 448 |",
        "| Conv2d 16→32 | 32 × (16 × 3 × 3 + 1) | 4,640 |",
        "| Conv2d 32→64 | 64 × (32 × 3 × 3 + 1) | 18,496 |",
        "| Adaptive average pooling | No learned weights | 0 |",
        "| Linear 64→32 | 64 × 32 + 32 | 2,080 |",
        "| Linear 32→10 | 32 × 10 + 10 | 330 |",
        f"| **Total** |  | **{results[0]['trainable_parameters']:,}** |",
        "",
        "At FP32, the weights-only estimate is "
        f"**{results[0]['estimated_float32_size_kb']:.2f} KiB** "
        "(parameters × 4 bytes ÷ 1,024). This excludes checkpoint metadata; actual "
        "checkpoint sizes are reported below.",
        "",
        "### Activation choice",
        "",
        "ReLU computes max(0, x), avoiding the exponential used by sigmoid/tanh and "
        "being less prone to saturating gradients. It is supported efficiently by "
        "common CPUs, DSPs, and quantized inference kernels. In-place ReLU can reuse "
        "activation storage. Max pooling and global average pooling reduce the spatial "
        "activation footprint and the classifier input size.",
        "",
        "## Optimizer comparison and held-out test results",
        "",
        "All optimizers use cross-entropy and no learning-rate scheduler. AdamW uses "
        "decoupled weight decay of 0.0001; both SGD baselines have zero weight decay. "
        "The checkpoint with minimum validation loss is evaluated on the "
        "held-out test set.",
        "",
        "| Optimizer | Learning rate | Momentum | Accuracy | Macro precision | Macro recall | "
        "Parameters | FP32 weights (KiB) | Checkpoint (KiB) | Mean epoch (s) | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        momentum = "—" if result["momentum"] is None else str(result["momentum"])
        lines.append(
            f"| {result['optimizer']} | {result['learning_rate']:.4g} | {momentum} "
            f"| {result['accuracy']:.4f} | {result['macro_precision']:.4f} "
            f"| {result['macro_recall']:.4f} | {result['trainable_parameters']:,} "
            f"| {result['estimated_float32_size_kb']:.2f} "
            f"| {result['checkpoint_file_size_kb']:.2f} "
            f"| {result['mean_epoch_seconds']:.2f} | {result['best_epoch']} |"
        )
    lines.extend(["", "### Optimizer rationale and confusion matrices", ""])
    for result in results:
        name = result["optimizer"]
        lines.extend([
            f"#### {name}",
            "",
            OPTIMIZER_RATIONALES[name],
            "",
            f"![{name} training and validation loss](./{name}/loss_curves.png)",
            "",
            f"Loss history: [`{name}/training_history.csv`](./{name}/training_history.csv)  ",
            f"Detailed metrics JSON: [`{name}/test_metrics.json`](./{name}/test_metrics.json)",
            "",
            "Confusion matrix (rows = true class, columns = predicted class):",
            "",
            "| True / Predicted | " + " | ".join(result["class_names"]) + " |",
            "| " + " --- |" * (len(result["class_names"]) + 1),
        ])
        for class_name, row in zip(
            result["class_names"], result["confusion_matrix"], strict=True
        ):
            lines.append(f"| {class_name} | " + " | ".join(map(str, row)) + " |")
        lines.extend(["", "Per-class precision and recall:", ""])
        lines.append("| Class | Precision | Recall | Support |")
        lines.append("| --- | ---: | ---: | ---: |")
        for class_name in result["class_names"]:
            scores = result["per_class"][class_name]
            lines.append(
                f"| {class_name} | {scores['precision']:.4f} | "
                f"{scores['recall']:.4f} | {scores['support']} |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits_torch_v1")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "model_a")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    args = parser.parse_args()
    if args.epochs < 20:
        parser.error("--epochs must be at least 20")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error("--run-id must be a single directory name")

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        parser.error(f"Run directory already exists; choose another --run-id: {run_dir}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    loaders = get_dataloaders(
        args.data_root,
        args.split_dir,
        batch_size=args.batch_size,
        transform=ToTensor(),
        num_workers=args.num_workers,
        seed=args.seed,
    )
    class_names = loaders["train"].dataset.classes
    run_dir.mkdir(parents=True)
    run_config = {
        "model": "ModelA",
        "seed": args.seed,
        "epochs_per_optimizer": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "dataset_sha256": json.loads(
            (args.split_dir / "metadata.json").read_text(encoding="utf-8")
        )["dataset_sha256"],
        "class_names": class_names,
        "optimizer_settings": OPTIMIZERS,
        "checkpoint_selection": "minimum validation cross-entropy loss",
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    summaries = []
    for optimizer_name, settings in OPTIMIZERS.items():
        output_dir = run_dir / optimizer_name
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        if loaders["train"].generator is not None:
            loaders["train"].generator.manual_seed(args.seed)
        model = ModelA(num_classes=len(class_names))
        optimizer = make_optimizer(model, optimizer_name, settings["learning_rate"], settings.get("weight_decay", 0.0))
        training = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            output_dir=output_dir,
            optimizer_name=optimizer_name,
            learning_rate=settings["learning_rate"],
        )

        best_model = ModelA(num_classes=len(class_names))
        checkpoint = load_checkpoint(best_model, output_dir / "best_model.pt", device)
        evaluation = evaluate_model(best_model, loaders["test"], class_names, device)
        checkpoint_size_kb = (output_dir / "best_model.pt").stat().st_size / 1024
        evaluation.update({
            "optimizer": optimizer_name,
            "learning_rate": settings["learning_rate"],
            "momentum": settings["momentum"],
            "weight_decay": settings.get("weight_decay", 0.0),
            "best_epoch": checkpoint["epoch"],
            "best_validation_loss": checkpoint["val_loss"],
            "mean_epoch_seconds": training["mean_epoch_seconds"],
            "estimated_float32_size_kb": evaluation["estimated_float32_size_kb"],
            "checkpoint_file_size_kb": checkpoint_size_kb,
            "checkpoint": str(output_dir / "best_model.pt"),
            "training_history_csv": training["history_csv"],
            "loss_curve": training["loss_curve"],
        })
        save_evaluation(evaluation, output_dir / "test_metrics.json")
        summaries.append(evaluation)
        print(
            f"{optimizer_name}: test_accuracy={evaluation['accuracy']:.4f}, "
            f"macro_precision={evaluation['macro_precision']:.4f}, "
            f"macro_recall={evaluation['macro_recall']:.4f}",
            flush=True,
        )

    (run_dir / "comparison.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    summary_fields = (
        "optimizer", "learning_rate", "momentum", "weight_decay", "accuracy", "macro_precision",
        "macro_recall", "trainable_parameters", "estimated_float32_size_kb",
        "checkpoint_file_size_kb", "mean_epoch_seconds", "best_epoch",
    )
    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in summary_fields} for row in summaries)
    report_path = write_experiment_report(run_dir, run_config, summaries)
    print(f"Comparison saved to: {run_dir}")
    print(f"Full report saved to: {report_path}")
    print(f"Model A trainable parameters: {count_trainable_parameters(ModelA())}")


if __name__ == "__main__":
    main()
