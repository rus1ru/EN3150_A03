"""Model B experiment orchestration; all output lives in its own run directory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import ToTensor

from en3150_a03.data import get_dataloaders
from en3150_a03.model_a_evaluate import evaluate_model, load_checkpoint
from en3150_a03.model_b_measure import profile_model
from en3150_a03.model_b_train import OPTIMIZERS, atomic_json, train_model
from en3150_a03.models.model_a import ModelA
from en3150_a03.models.model_b import ModelB

ROOT = Path(__file__).resolve().parents[2]


def seed_run(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def write_report(output, config, results, profiles):
    a, b = profiles["model_a"], profiles["model_b"]
    lines = ["# Model B: EuroSAT experiment report", "",
             "Generated from saved results; scores are fractions, times are seconds.", "",
             f"Dataset SHA-256: `{config['dataset_sha256']}`. Seed: {config['seed']}. "
             f"Epochs per optimizer: {config['epochs_per_optimizer']}. Batch size: {config['batch_size']}.",
             "", f"Hardware/software: `{config['environment']}`", "",
             "## Controlled architecture comparison", "",
             "Both custom models use widths 16/32/64, 2x2 max pooling after each convolution block, "
             "global average pooling, and a 64 -> 32 -> 10 ReLU head. "
             "Model B factorizes each standard convolution into a depthwise 3x3 and pointwise 1x1. "
             "No normalization, augmentation, schedule or weight decay is added.", "",
             "| Model | Parameters | FP32 weights (KiB) | Conv/Linear MACs |", "| --- | ---: | ---: | ---: |",
             f"| A | {a['total_parameters']:,} | {a['estimated_float32_size_kb']:.2f} | {a['macs']:,} |",
             f"| B | {b['total_parameters']:,} | {b['estimated_float32_size_kb']:.2f} | {b['macs']:,} |", "",
             f"Weight reduction: {1-b['total_parameters']/a['total_parameters']:.2%}; "
             f"MAC reduction: {1-b['macs']/a['macs']:.2%}. {b['mac_convention']}", "",
             "## Model B optimizer comparison", "",
             "The checkpoint with minimum validation cross-entropy is evaluated once on the test split. "
             "Optimizer comparisons use independently reset model and shuffle seeds. "
             "Adam is the predeclared primary optimizer (lr=0.001); plain SGD and SGD+momentum "
             "use lr=0.01, with momentum=0.9 for the latter. No optimizer is selected using test accuracy.", "",
             "| Optimizer | Test accuracy | Macro precision | Macro recall | 95% Wilson CI | Mean epoch (s) | Best epoch |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in results:
        lo, hi = row["accuracy_wilson_95"]
        lines.append(f"| {row['optimizer']} | {row['accuracy']:.4f} | {row['macro_precision']:.4f} | "
                     f"{row['macro_recall']:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{row['mean_epoch_seconds']:.2f} | {row['best_epoch']} |")
    lines += ["", "The Wilson interval describes image-level binomial uncertainty, not variability across seeds. "
              "Nearby satellite tiles may be correlated. This is one seeded run per optimizer, "
              "not a geographical holdout or evidence of statistical superiority.", "",
              "## Curves and class errors", ""]
    for row in results:
        name = row["optimizer"]
        lines += [f"### {name}", "", f"![Loss and accuracy]({name}/loss_curves.png)", "",
                  f"![Test confusion matrix]({name}/confusion_matrix.png)", "",
                  f"[Per-class scores]({name}/per_class_metrics.csv) | "
                  f"[Detailed metrics]({name}/test_metrics.json)", ""]
    lines += ["## Interpretation and limitations", "",
              "Depthwise factorization restricts each input channel to one spatial filter before channel mixing. "
              "The resulting memory/operation saving can reduce representational capacity. "
              "ReLU uses a max-with-zero operation without exponentials. Global pooling avoids a large flattened head.", "",
              "Momentum accumulates a smoothed gradient direction; compare the saved validation curves to plain SGD "
              "to assess convergence. Equal SGD learning rates isolate the addition of momentum; these fixed "
              "settings are not equally tuned optimizer optima.", "",
              "Epoch times include synchronized training and validation, excluding checkpoint and plot writes. "
              "Do not compare Mac/Colab/Windows epoch times as architecture speedups. "
              "FP32 parameter storage excludes activations, runtime workspaces and optimizer state; actual "
              "state_dict and checkpoint bytes are recorded separately in test_metrics.json.", "",
              "Run scripts/compare_model_b.py to combine these results with teammates' existing metrics "
              "without editing their files. See docs/model_b.md for parameter derivations and references."]
    (output / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment(device):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cpu_threads=torch.get_num_threads(), git_commit=commit, working_tree_dirty=dirty)


def run(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select a Colab GPU runtime")
    seed_run(args.seed)
    # This verifies image bytes, checksums and overlap before training can start.
    loaders = get_dataloaders(args.data_root, args.split_dir, batch_size=args.batch_size,
                              transform=ToTensor(), num_workers=args.num_workers, seed=args.seed)
    mapping = loaders["train"].dataset.class_to_idx
    classes = sorted(mapping, key=mapping.get)
    metadata = json.loads((args.split_dir / "metadata.json").read_text())
    source_files = [ROOT / "src/en3150_a03" / name for name in (
        "models/model_b.py", "model_b_train.py", "model_b_experiment.py", "model_b_measure.py",
        "model_a_train.py", "model_a_evaluate.py", "model_a_metrics.py", "data.py")]
    config = dict(model="ModelB", architecture="matched_depthwise_v1", seed=args.seed,
                  epochs_per_optimizer=args.epochs, batch_size=args.batch_size,
                  num_workers=args.num_workers, optimizer_settings={name: OPTIMIZERS[name] for name in args.optimizers},
                  preprocessing="ToTensor RGB [0,1]; no augmentation or normalization", image_size=[64, 64],
                  class_to_idx=mapping, dataset_sha256=metadata["dataset_sha256"],
                  manifest_sha256=metadata["manifest_sha256"],
                  checkpoint_selection="minimum validation cross-entropy; earliest epoch breaks ties",
                  code_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    output = args.results_dir / args.run_id
    config_path = output / "experiment_config.json"
    if output.exists():
        if not args.resume or not config_path.is_file():
            raise FileExistsError("Use a new run ID or --resume for an existing Model B run")
        saved = json.loads(config_path.read_text())
        if {k: v for k, v in saved.items() if k != "environment"} != config:
            raise ValueError("Run configuration/source differs; use the original code/settings to resume")
        if saved["environment"] != environment(device):
            # Timing remains meaningful only when all epochs ran on the same device.
            previous = saved["environment"]
            current = environment(device)
            for key in ("platform", "processor", "torch", "torchvision", "device", "gpu", "cpu_threads"):
                if previous[key] != current[key]:
                    raise ValueError(f"Resume hardware/software differs ({key}); use a new run ID")
        config = saved
    else:
        output.mkdir(parents=True)
        config["environment"] = environment(device)
        atomic_json(config_path, config)
    profiles = {"model_a": profile_model(ModelA(len(classes))), "model_b": profile_model(ModelB(len(classes)))}
    atomic_json(output / "architecture_profiles.json", profiles)
    summaries = []
    # Finish all training before any test evaluation.
    for name in args.optimizers:
        seed_run(args.seed)
        loaders["train"].generator.manual_seed(args.seed)
        train_model(ModelB(len(classes)), loaders["train"], loaders["val"], device,
                    args.epochs, output / name, name, config, resume=args.resume)
    for name in args.optimizers:
        folder = output / name
        metrics_path = folder / "test_metrics.json"
        if args.resume and metrics_path.exists():
            result = json.loads(metrics_path.read_text())
        else:
            model = ModelB(len(classes))
            checkpoint = load_checkpoint(model, folder / "best_model.pt", device)
            if checkpoint["contract"] != config:
                raise ValueError("Best checkpoint contract mismatch")
            result = evaluate_model(model, loaders["test"], classes, device)
            training = json.loads((folder / "training_summary.json").read_text())
            weights = folder / "model_weights.pt"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
            matrix = result["confusion_matrix"]
            total = sum(map(sum, matrix))
            result.update(model="ModelB", optimizer=name, **OPTIMIZERS[name],
                          best_epoch=checkpoint["epoch"], best_validation_loss=checkpoint["val_loss"],
                          mean_epoch_seconds=training["mean_epoch_seconds"],
                          total_parameters=profiles["model_b"]["total_parameters"],
                          macs=profiles["model_b"]["macs"], mac_convention=profiles["model_b"]["mac_convention"],
                          checkpoint_file_size_kb=(folder / "best_model.pt").stat().st_size / 1024,
                          saved_model_size_bytes=weights.stat().st_size,
                          fp32_parameter_size_mb=profiles["model_b"]["fp32_parameter_size_mb"],
                          accuracy_wilson_95=wilson_interval(sum(matrix[i][i] for i in range(len(classes))), total),
                          dataset_sha256=config["dataset_sha256"], manifest_sha256=config["manifest_sha256"],
                          class_to_idx=mapping, checkpoint="best_model.pt", loss_curve="loss_curves.png")
            atomic_json(metrics_path, result)
        save_confusion(result, folder)
        summaries.append(result)
        print(f"{name}: test accuracy={result['accuracy']:.4f}", flush=True)
    atomic_json(output / "comparison.json", summaries)
    fields = ["optimizer", "learning_rate", "momentum", "accuracy", "macro_precision", "macro_recall",
              "trainable_parameters", "estimated_float32_size_kb", "macs", "mean_epoch_seconds", "best_epoch"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)
    write_report(output, config, summaries, profiles)
    print(f"Saved Model B report and metrics: {output}", flush=True)


def main(default_optimizers=("adam", "sgd", "sgd_momentum")):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/EuroSAT_RGB")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits_torch_v1")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/model_b")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--optimizers", nargs="+", choices=tuple(OPTIMIZERS), default=list(default_optimizers))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 20:
        parser.error("Custom models require at least 20 epochs")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive; workers must be nonnegative")
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error("Each optimizer may appear only once")
    if not args.run_id or args.run_id in (".", "..") or Path(args.run_id).name != args.run_id or "\\" in args.run_id:
        parser.error("Run ID must be a single directory name")
    run(args)
