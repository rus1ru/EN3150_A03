"""Train MobileNetV2 on the existing EuroSAT split, then evaluate its best checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from en3150_a03.mobilenet_train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/raw/EuroSAT_RGB')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'data/splits_torch_v1')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this run')
    parser.add_argument('--epochs', type=int, default=20, help='Total epochs including warm-up')
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--finetune-lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Require 0 <= warmup-epochs < epochs')
    if args.batch_size < 2 or args.num_workers < 0 or min(args.head_lr, args.finetune_lr) <= 0:
        parser.error('Require batch-size >= 2, num-workers >= 0, and positive learning rates')
    if args.evaluate_only and not args.checkpoint:
        parser.error('--evaluate-only requires --checkpoint')
    run(args)


if __name__ == '__main__':
    main()
