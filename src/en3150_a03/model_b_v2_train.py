"""Model B v2: validation-only training, resumable epochs and explicit final test."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from torch import nn

from en3150_a03.data import verify_splits
from en3150_a03.model_a_evaluate import evaluate_model
from en3150_a03.model_b_experiment import environment, save_confusion, wilson_interval
from en3150_a03.model_b_measure import profile_model
from en3150_a03.model_b_train import atomic_json, atomic_checkpoint, synchronize, save_history
from en3150_a03.model_b_v2_data import training_statistics, make_loaders
from en3150_a03.models.model_b_v2 import ModelBV2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LR = {"adamw": .003, "adam": .003, "sgd": .05, "sgd_momentum": .05}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_optimizer(model, name, lr, weight_decay):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.}]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    return torch.optim.SGD(groups, lr=lr, momentum=.9 if name == "sgd_momentum" else 0.)


def learning_rate(epoch, epochs, base_lr, warmup=5):
    warmup = min(warmup, max(1, epochs//5))
    if epoch <= warmup:
        return base_lr * epoch / warmup
    fraction = (epoch-warmup) / max(1, epochs-warmup)
    return 1e-5 + .5 * (base_lr-1e-5) * (1 + math.cos(math.pi*fraction))


def run_epoch(model, loader, device, optimizer=None, scaler=None, smoothing=.05, amp=False):
    training = optimizer is not None
    model.train(training)
    loss_total = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing if training else 0.)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss: stopping without advancing the checkpoint")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            loss_total += loss.detach() * labels.numel()
            correct += (logits.argmax(1) == labels).sum()
            count += labels.numel()
    if not count:
        raise ValueError("Empty data loader")
    return float(loss_total/count), float(correct/count)


def plot_history(history, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Model B v2: training uses augmentation and smoothed loss; validation uses plain cross-entropy")
    fig.tight_layout()
    fig.savefig(output/"loss_curves.png", dpi=160)
    plt.close(fig)


def train(model, loaders, device, config, output, resume=False, stop_after=None):
    """Per-epoch seeds make worker augmentation reproducible at resume boundaries."""
    optimizer = make_optimizer(model, config["optimizer"], config["learning_rate"], config["weight_decay"])
    amp = device.type == "cuda" and config["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_loss, best_epoch = [], float("inf"), 0
    last = output/"last_model.pt"
    if last.exists():
        if not resume:
            raise FileExistsError("Use --resume for existing checkpoints")
        state = torch.load(last, map_location=device, weights_only=True)
        if state["contract"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        history, best_loss, best_epoch = state["history"], state["best_val_loss"], state["best_epoch"]
        if not (output/"best_model.pt").is_file():
            raise FileNotFoundError("Missing best checkpoint")
    for epoch in range(len(history)+1, config["epochs"]+1):
        seed_all(config["seed"] + epoch * 1009)
        for loader in loaders.values():
            loader.generator.manual_seed(config["seed"] + epoch * 1009)
        lr = learning_rate(epoch, config["epochs"], config["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        synchronize(device)
        start = time.perf_counter()
        loss, accuracy = run_epoch(model, loaders["train"], device, optimizer, scaler, config["label_smoothing"], amp)
        val_loss, val_accuracy = run_epoch(model, loaders["val"], device, amp=amp)
        synchronize(device)
        row = dict(epoch=epoch, learning_rate=lr, train_loss=loss, train_accuracy=accuracy,
                   val_loss=val_loss, val_accuracy=val_accuracy, epoch_seconds=time.perf_counter()-start)
        history.append(row)
        state = {"model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                 "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy, "contract": config}
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            atomic_checkpoint(output/"best_model.pt", state)
        state.update(optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                     history=history, best_val_loss=best_loss, best_epoch=best_epoch)
        atomic_checkpoint(last, state)
        save_history(history, output)
        atomic_json(output/"training_summary.json", dict(epochs_completed=len(history), planned_epochs=config["epochs"],
            complete=len(history)==config["epochs"], best_epoch=best_epoch, best_val_loss=best_loss,
            best_val_accuracy=history[best_epoch-1]["val_accuracy"],
            mean_epoch_seconds=sum(r["epoch_seconds"] for r in history)/len(history)))
        print(f"{config['optimizer']} {epoch:03}/{config['epochs']}: train={loss:.4f}/{accuracy:.2%} "
              f"val={val_loss:.4f}/{val_accuracy:.2%} lr={lr:.6f} {row['epoch_seconds']:.1f}s", flush=True)
        if epoch % 10 == 0 or epoch == config["epochs"]:
            plot_history(history, output)
        if stop_after is not None and epoch >= stop_after:
            break
    plot_history(history, output)
    return history


def source_hashes():
    files = ["models/model_b_v2.py", "model_b_v2_data.py", "model_b_v2_train.py", "data.py", "model_a_evaluate.py", "model_a_metrics.py"]
    return {name:hashlib.sha256((ROOT/"src/en3150_a03"/name).read_bytes()).hexdigest() for name in files}


def evaluate_final(args, output, device):
    config = json.loads((output/"experiment_config.json").read_text())
    summary = json.loads((output/"training_summary.json").read_text())
    if not summary["complete"] or summary["epochs_completed"] < 20:
        raise ValueError("Finish the planned training (at least 20 epochs) before final test evaluation")
    if source_hashes() != config["code_sha256"]:
        raise ValueError("Source changed since training; evaluate using the saved source version")
    audit = verify_splits(args.data_root, args.split_dir)
    metadata = json.loads((args.split_dir/"metadata.json").read_text())
    if audit["dataset_sha256"] != config["dataset_sha256"] or metadata["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Dataset/split differs from trained run")
    if (output/"test_metrics.json").exists():
        print("Existing final test results retained; no repeated evaluation.")
        return
    model = ModelBV2(len(config["class_to_idx"]), config["attention"], config["dropout"]).to(device)
    checkpoint = torch.load(output/"best_model.pt", map_location=device, weights_only=True)
    if checkpoint["contract"] != config:
        raise ValueError("Checkpoint contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loaders = make_loaders(args.data_root, args.split_dir, config["normalization"], config["batch_size"],
                           args.num_workers, config["seed"], include_test=True)
    classes = sorted(config["class_to_idx"], key=config["class_to_idx"].get)
    result = evaluate_model(model, loaders["test"], classes, device)
    profile = json.loads((output/"architecture_profile.json").read_text())
    torch.save(checkpoint["model_state_dict"], output/"model_weights.pt")
    matrix = result["confusion_matrix"]
    result.update(model="ModelBV2", optimizer=config["optimizer"], best_epoch=checkpoint["epoch"],
        best_validation_loss=checkpoint["val_loss"], best_validation_accuracy=checkpoint["val_accuracy"],
        total_parameters=profile["total_parameters"], macs=profile["macs"],
        mac_convention=profile["mac_convention"], mean_epoch_seconds=summary["mean_epoch_seconds"],
        dataset_sha256=config["dataset_sha256"], manifest_sha256=config["manifest_sha256"],
        saved_model_size_bytes=(output/"model_weights.pt").stat().st_size,
        accuracy_wilson_95=wilson_interval(sum(matrix[i][i] for i in range(len(classes))),sum(map(sum,matrix))))
    atomic_json(output/"test_metrics.json", result)
    save_confusion(result, output)
    print(json.dumps({k:result[k] for k in ("accuracy","macro_precision","macro_recall","total_parameters")},indent=2))


def run(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise ValueError("Select a Colab GPU runtime")
    torch.set_num_threads(args.cpu_threads)
    output = args.results_dir/args.run_id
    if args.evaluate_only:
        evaluate_final(args, output, device)
        return
    seed_all(args.seed)
    audit = verify_splits(args.data_root, args.split_dir)
    metadata = json.loads((args.split_dir/"metadata.json").read_text())
    existing = None
    if output.exists():
        if not args.resume or not (output/"experiment_config.json").exists():
            raise FileExistsError("Choose a new run ID or --resume")
        existing = json.loads((output/"experiment_config.json").read_text())
    statistics = existing["normalization"] if existing else training_statistics(args.data_root,args.split_dir)
    config = dict(architecture="model_b_v2_inverted_residual_se", attention=not args.no_attention,
        dropout=.1, optimizer=args.optimizer, learning_rate=args.learning_rate or DEFAULT_LR[args.optimizer],
        weight_decay=args.weight_decay, label_smoothing=.05, augmentation=not args.no_augmentation,
        epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, amp=not args.no_amp, normalization=statistics, class_to_idx=metadata["class_to_idx"],
        dataset_sha256=audit["dataset_sha256"], manifest_sha256=metadata["manifest_sha256"],
        code_sha256=source_hashes(), schedule="5-epoch linear warmup then cosine to 1e-5",
        checkpoint_selection="minimum validation plain cross-entropy; test requires explicit --evaluate-only")
    if existing:
        if {k:v for k,v in existing.items() if k!="environment"} != config:
            raise ValueError("Resume settings/source differ; use the same settings and code archive")
        current = environment(device)
        for key in ("device","gpu","torch","torchvision","cpu_threads"):
            if current[key] != existing["environment"][key]:
                raise ValueError(f"Resume environment differs: {key}; use a new run")
        config = existing
    else:
        output.mkdir(parents=True)
        config["environment"] = environment(device)
        atomic_json(output/"experiment_config.json",config)
    model = ModelBV2(len(metadata["class_to_idx"]), config["attention"], config["dropout"])
    profile = profile_model(model)
    profile["mac_convention"] += " Also excludes BN, residual additions and SE elementwise gates. Includes SE Conv2d MACs."
    atomic_json(output/"architecture_profile.json", profile)
    print(f"Model B v2: {profile['total_parameters']:,} parameters; {profile['macs']:,} Conv/Linear MACs",flush=True)
    loaders = make_loaders(args.data_root,args.split_dir,statistics,args.batch_size,args.num_workers,args.seed,config["augmentation"])
    # Re-seed after profiling so initialization is independent of loader/statistics work.
    seed_all(args.seed)
    model = ModelBV2(len(metadata["class_to_idx"]),config["attention"],config["dropout"]).to(device)
    train(model,loaders,device,config,output,resume=args.resume)
    print("Training complete. Review validation before running --evaluate-only. Test data was not evaluated.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ROOT/"data/raw/EuroSAT_RGB")
    parser.add_argument("--split-dir",type=Path,default=ROOT/"data/splits_torch_v1")
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results/model_b")
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--optimizer",choices=tuple(DEFAULT_LR),default="adamw")
    parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--num-workers",type=int,default=2)
    parser.add_argument("--cpu-threads",type=int,default=2)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--learning-rate",type=float)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    for flag in ("resume","evaluate-only","no-attention","no-augmentation","no-amp"):
        parser.add_argument("--"+flag,action="store_true")
    args=parser.parse_args()
    if args.epochs<20 or args.batch_size<2 or args.num_workers<0 or args.cpu_threads<1:
        parser.error("Require >=20 epochs, batch size >=2, nonnegative workers and positive CPU threads")
    if (args.learning_rate is not None and args.learning_rate<=1e-5) or args.weight_decay<0:
        parser.error("Require lr>1e-5 and weight decay >=0")
    if args.run_id in (".","..") or Path(args.run_id).name != args.run_id or "\\" in args.run_id:
        parser.error("Run ID must be a single directory name")
    run(args)
