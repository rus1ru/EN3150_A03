"""Two-stage fine-tuning, held-out evaluation, and reproducible result files."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import transforms

from en3150_a03.data import get_dataloaders
from en3150_a03.model_a_metrics import classification_metrics
from en3150_a03.models.mobilenet_v2 import (
    build_mobilenet_v2, set_training_stage, set_train_mode,
)


def make_transforms():
    # Keep the assignment's native 64x64 resolution, without upsampling.
    normalization = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    evaluation = transforms.Compose([transforms.ToTensor(), normalization])
    training = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(), normalization,
    ])
    return training, evaluation


def run_epoch(model, loader, device, optimizer=None, head_only=False):
    training = optimizer is not None
    if training:
        set_train_mode(model, head_only)
    else:
        model.eval()
    loss_sum, correct, count = 0., 0, 0
    with torch.enable_grad() if training else torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            count += targets.numel()
            loss_sum += loss.item() * targets.numel()
            correct += (logits.argmax(1) == targets).sum().item()
    if not count:
        raise ValueError('Empty loader')
    return loss_sum / count, correct / count


def evaluate(model, loader, device, classes, output):
    model.eval()
    targets_all, predictions = [], []
    with torch.inference_mode():
        for images, targets in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets_all.extend(targets.tolist())
    result = classification_metrics(targets_all, predictions, classes)
    f1_scores = []
    for values in result['per_class'].values():
        p, r = values['precision'], values['recall']
        values['f1'] = 2 * p * r / (p + r) if p + r else 0.
        f1_scores.append(values['f1'])
    result['macro_f1'] = sum(f1_scores) / len(f1_scores)
    result['total_parameters'] = sum(p.numel() for p in model.parameters())
    result['trainable_parameters_full_finetuning'] = result['total_parameters']
    result['head_parameters'] = sum(p.numel() for p in model.classifier.parameters())
    result['fp32_parameter_size_mb'] = result['total_parameters'] * 4 / 1_000_000
    weights = output / 'model_weights.pt'
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
    result['saved_model_size_bytes'] = weights.stat().st_size
    result['saved_model_size_mb'] = weights.stat().st_size / 1_000_000
    result['size_definition'] = 'FP32 state_dict including buffers; decimal MB = bytes / 1,000,000; excludes optimizer and runtime activations'
    with (output / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['actual / predicted', *classes])
        writer.writerows([name, *row] for name, row in zip(classes, result['confusion_matrix']))
    return result


def save_plots(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history_path = output / 'training_history.csv'
    if history_path.exists():
        with history_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric in zip(axes, ['loss', 'accuracy']):
            for split in ['train', 'val']:
                axis.plot([int(r['epoch']) for r in rows],
                          [float(r[f'{split}_{metric}']) for r in rows], label=split)
            axis.set(xlabel='Epoch', ylabel=metric.capitalize())
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / 'training_curves.png', dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8))
    matrix = result['confusion_matrix']
    display = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(display, ax=ax)
    classes = result['class_names']
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted class', ylabel='Actual class', title='MobileNetV2 test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=65, ha='right')
    threshold = max(map(max, matrix)) / 2
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value > threshold else 'black')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    # Refuse accidental overwrite of a previous experiment.
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = make_transforms()
    loaders = get_dataloaders(
        args.data_root, args.split_dir, batch_size=args.batch_size,
        train_transform=train_transform, eval_transform=eval_transform,
        num_workers=args.num_workers, seed=args.seed,
    )
    mapping = loaders['train'].dataset.class_to_idx
    classes = sorted(mapping, key=mapping.get)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update({
        'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'device_used': str(device),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'image_size': [64, 64], 'class_to_idx': mapping,
        'pretrained_weights': 'MobileNet_V2_Weights.IMAGENET1K_V2',
        'selection': 'minimum validation loss across all epochs; test only after selection',
        'augmentation': 'independent horizontal and vertical flips, probability 0.5',
        'normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
        'manifest_sha256': {
            name: hashlib.sha256((args.split_dir / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in ['train.csv', 'val.csv', 'test.csv', 'class_to_idx.json', 'metadata.json']
        },
    })
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    model = build_mobilenet_v2(len(classes), pretrained=not args.evaluate_only).to(device)
    if args.evaluate_only:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['class_to_idx'] != mapping:
            raise ValueError('Checkpoint class mapping differs from shared split')
        if checkpoint['manifest_sha256'] != settings['manifest_sha256']:
            raise ValueError('Checkpoint split differs from current split')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_path = args.checkpoint
    else:
        best_loss, history, optimizer = float('inf'), [], None
        best_path = output / 'best_model.pt'
        for epoch in range(args.epochs):
            head_only = epoch < args.warmup_epochs
            if epoch == 0 or epoch == args.warmup_epochs:
                set_training_stage(model, head_only)
                optimizer = torch.optim.Adam(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=args.head_lr if head_only else args.finetune_lr,
                )
            started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(model, loaders['train'], device, optimizer, head_only)
            val_loss, val_accuracy = run_epoch(model, loaders['val'], device)
            row = dict(epoch=epoch + 1, stage='head' if head_only else 'finetune',
                       train_loss=train_loss, train_accuracy=train_accuracy,
                       val_loss=val_loss, val_accuracy=val_accuracy,
                       seconds=time.perf_counter() - started)
            history.append(row)
            print(row, flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(dict(
                    model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    epoch=epoch + 1, stage=row['stage'], val_loss=val_loss,
                    class_to_idx=mapping, manifest_sha256=settings['manifest_sha256'],
                    settings=settings,
                ), best_path)
            with (output / 'training_history.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(history)
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    result = evaluate(model, loaders['test'], device, classes, output)
    result.update(best_epoch=checkpoint['epoch'], best_stage=checkpoint['stage'],
                  best_val_loss=checkpoint['val_loss'], checkpoint=str(best_path))
    (output / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'test_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                  'total_parameters', 'saved_model_size_mb', 'best_epoch', 'best_stage']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: result[field] for field in fields})
    save_plots(output, result)
    print(json.dumps(result, indent=2))
