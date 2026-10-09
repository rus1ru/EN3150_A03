"""Fine-tune EfficientNet-B0 on the published EuroSAT RGB split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from en3150_a03.data import get_dataloaders  # noqa: E402
from en3150_a03.models.efficientnet_b0 import build_model, get_transforms  # noqa: E402


def evaluate(model, loader, device, limit=None):
    model.eval()
    loss_sum, n, truth, predictions = 0.0, 0, [], []
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if limit is not None and batch_index >= limit:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            n += len(labels)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not n:
        raise ValueError("Evaluation loader is empty")
    return loss_sum / n, truth, predictions


def accuracy_score(truth, predictions):
    return sum(a == b for a, b in zip(truth, predictions)) / len(truth)


def classification_metrics(truth, predictions, classes):
    count = len(classes)
    matrix = [[0] * count for _ in classes]
    for actual, predicted in zip(truth, predictions):
        matrix[actual][predicted] += 1
    precisions, recalls, f1s = [], [], []
    for index in range(count):
        tp = matrix[index][index]
        precision = tp / (sum(row[index] for row in matrix) or 1)
        recall = tp / (sum(matrix[index]) or 1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"macro_precision": sum(precisions) / count,
            "macro_recall": sum(recalls) / count, "macro_f1": sum(f1s) / count,
            "confusion_matrix": matrix, "class_order": classes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="Skip pretrained ImageNet weights")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/EuroSAT_RGB")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits_torch_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/efficientnet_b0/adam")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Smoke test only: cap train/val batches and skip test metrics")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("epochs, batch-size, lr and max-batches must be positive; workers must be nonnegative")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(args.image_root, args.split_dir, batch_size=args.batch_size,
                              train_transform=get_transforms(train=True),
                              eval_transform=get_transforms(), num_workers=args.workers, seed=args.seed)
    classes = loaders["train"].dataset.classes
    model = build_model(len(classes), pretrained=not args.scratch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.pt"
    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        train_loss, train_n = 0.0, 0
        for batch_index, (images, labels) in enumerate(loaders["train"]):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_n += len(labels)
        val_loss, truth, predictions = evaluate(model, loaders["val"], device, args.max_batches)
        val_accuracy = accuracy_score(truth, predictions)
        record = {"epoch": epoch, "train_loss": train_loss / train_n, "val_loss": val_loss,
                  "val_accuracy": val_accuracy, "seconds": time.monotonic() - start}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "epoch": epoch, "val_accuracy": val_accuracy}, checkpoint)
    result = {"model": "efficientnet_b0", "pretrained": not args.scratch,
              "image_size": 64, "split_dir": str(args.split_dir), "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "device": str(device), "torch_version": torch.__version__,
              "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
              "history": history, "best_val_accuracy": best_accuracy,
              "smoke_test": args.max_batches is not None}
    if args.max_batches is None:
        # Load the checkpoint selected on validation data; touch test only once.
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        test_loss, truth, predictions = evaluate(model, loaders["test"], device)
        result["test"] = {"loss": test_loss, "accuracy": accuracy_score(truth, predictions),
                          **classification_metrics(truth, predictions, classes)}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
