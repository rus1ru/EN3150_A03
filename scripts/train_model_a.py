"""Train Model A with one optimizer and validation-loss checkpoint selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torchvision.transforms import ToTensor

from en3150_a03.data import get_dataloaders
from en3150_a03.models.model_a import ModelA
from en3150_a03.model_a_train import make_optimizer, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits_torch_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd", "sgd_momentum"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 20:
        parser.error("--epochs must be at least 20")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    learning_rates = {"adam": 0.001, "adamw": 0.001, "sgd": 0.01, "sgd_momentum": 0.01}
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else learning_rates[args.optimizer]
    )
    if learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    loaders = get_dataloaders(
        args.data_root,
        args.split_dir,
        batch_size=args.batch_size,
        transform=ToTensor(),
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = ModelA(num_classes=len(loaders["train"].dataset.classes))
    optimizer = make_optimizer(model, args.optimizer, learning_rate, 0.0001 if args.optimizer == "adamw" else 0.0)
    result = train_model(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        device,
        args.epochs,
        args.output_dir,
        args.optimizer,
        learning_rate,
    )
    print(result)


if __name__ == "__main__":
    main()
