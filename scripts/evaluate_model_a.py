"""Evaluate one Model A checkpoint on the held-out EuroSAT test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torchvision.transforms import ToTensor

from en3150_a03.data import get_dataloaders
from en3150_a03.model_a_evaluate import evaluate_model, load_checkpoint, save_evaluation
from en3150_a03.models.model_a import ModelA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits_torch_v1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
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
    )
    model = ModelA(num_classes=len(loaders["test"].dataset.classes))
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    result = evaluate_model(
        model, loaders["test"], loaders["test"].dataset.classes, device
    )
    result["optimizer"] = checkpoint.get("optimizer")
    result["checkpoint"] = str(args.checkpoint)
    output = args.output or args.checkpoint.with_name("test_metrics.json")
    save_evaluation(result, output)
    print(result)


if __name__ == "__main__":
    main()
