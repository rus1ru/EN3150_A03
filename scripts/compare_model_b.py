"""Combine final Model B metrics with teammate runs without modifying inputs."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from en3150_a03.model_b_measure import profile_model
from en3150_a03.models.model_a import ModelA
from en3150_a03.models.model_b_v2 import ModelBV2
from en3150_a03.models.mobilenet_v2 import build_mobilenet_v2
from en3150_a03.models.efficientnet_b0 import build_model as build_efficientnet_b0


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def comparison(model_b_run, model_a_run, mobilenet_run, output, efficientnet_run=None):
    metadata = read(ROOT / 'data/splits_torch_v1/metadata.json')
    if output.exists():
        raise FileExistsError(f'Choose a new comparison directory: {output}')
    b_runs = [model_b_run] if isinstance(model_b_run, Path) else list(model_b_run)
    sources = [('ModelA', model_a_run), *[('ModelB', p) for p in b_runs], ('MobileNetV2', mobilenet_run)]
    rows = []
    for name, folder in sources:
        if folder is None:
            continue
        config = read(folder / ('settings.json' if name == 'MobileNetV2' else 'experiment_config.json'))
        if name == 'MobileNetV2':
            if config['class_to_idx'] != metadata['class_to_idx']:
                raise ValueError('MobileNet class mapping differs')
            if any(config['manifest_sha256'].get(k) != v for k,v in metadata['manifest_sha256'].items()):
                raise ValueError('MobileNet split provenance differs')
            model = build_mobilenet_v2(pretrained=False)
            metrics = [read(folder/'test_metrics.json')]
        else:
            if config['dataset_sha256'] != metadata['dataset_sha256']:
                raise ValueError(f'Dataset differs: {folder}')
            if name == 'ModelB':
                if config['manifest_sha256'] != metadata['manifest_sha256']:
                    raise ValueError('Model B split differs')
                if config['class_to_idx'] != metadata['class_to_idx']:
                    raise ValueError('Model B class mapping differs')
                summary = read(folder/'training_summary.json')
                if not summary['complete'] or summary['epochs_completed'] != config['epochs']:
                    raise ValueError('Model B training incomplete')
                model = ModelBV2(len(metadata['classes']), config['attention'], config['dropout'])
                metrics = [read(folder/'test_metrics.json')]
            else:
                model = ModelA()
                metrics = read(folder/'comparison.json')
        profile = profile_model(model)
        if name == 'ModelB':
            saved = read(folder/'architecture_profile.json')
            if any(saved[k] != profile[k] for k in ('total_parameters','macs')):
                raise ValueError('Saved Model B profile differs from architecture')
        for metric in metrics:
            if metric['class_names'] != metadata['classes']:
                raise ValueError(f'Metric class order differs: {folder}')
            if metric.get('total_parameters',metric.get('trainable_parameters')) != profile['total_parameters']:
                raise ValueError(f'Parameter count differs: {folder}')
            if name == 'ModelB' and (metric['manifest_sha256'] != config['manifest_sha256'] or metric['dataset_sha256'] != config['dataset_sha256']):
                raise ValueError('Model B metric provenance differs')
            source = folder.resolve().relative_to(ROOT) if folder.resolve().is_relative_to(ROOT) else folder
            rows.append(dict(model=name, optimizer=metric.get('optimizer','adam_two_stage'),
                accuracy=metric['accuracy'], macro_precision=metric['macro_precision'], macro_recall=metric['macro_recall'],
                parameters=profile['total_parameters'], fp32_weights_kib=profile['estimated_float32_size_kb'],
                macs=profile['macs'], mean_epoch_seconds=metric.get('mean_epoch_seconds'), source=str(source)))
    if efficientnet_run is not None:
        metric = read(efficientnet_run/'metrics.json')
        test = metric['test']
        if metric['smoke_test'] or metric['image_size'] != 64:
            raise ValueError('Expected a full 64x64 EfficientNet-B0 run')
        if test['class_order'] != metadata['classes']:
            raise ValueError('EfficientNet-B0 class order differs')
        profile = profile_model(build_efficientnet_b0(pretrained=False))
        if metric['trainable_parameters'] != profile['total_parameters']:
            raise ValueError('EfficientNet-B0 parameter count differs')
        if sum(map(sum, test['confusion_matrix'])) != metadata['counts']['test']:
            raise ValueError('EfficientNet-B0 test count differs')
        source = efficientnet_run.resolve().relative_to(ROOT) if efficientnet_run.resolve().is_relative_to(ROOT) else efficientnet_run
        rows.append(dict(model='EfficientNet-B0', optimizer='adam',
            accuracy=test['accuracy'], macro_precision=test['macro_precision'],
            macro_recall=test['macro_recall'], parameters=profile['total_parameters'],
            fp32_weights_kib=profile['estimated_float32_size_kb'], macs=profile['macs'],
            mean_epoch_seconds=sum(epoch['seconds'] for epoch in metric['history']) / len(metric['history']),
            source=str(source)))
    if not rows:
        raise ValueError('No results supplied')
    b_opts = {r['optimizer'] for r in rows if r['model']=='ModelB'}
    missing = sorted({'adamw','sgd','sgd_momentum'} - b_opts)
    lines = ['# Final Model B integration comparison', '',
        'Read from saved runs. Checkpoints were selected using validation loss; this table does not select an optimizer by test performance.', '',
        '| Model | Optimizer | Test accuracy | Macro precision | Macro recall | Parameters | FP32 weights (KiB) | Conv/Linear MACs | Mean epoch (s) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        seconds = 'unavailable' if row['mean_epoch_seconds'] is None else f"{row['mean_epoch_seconds']:.2f}"
        lines.append(f"| {row['model']} | {row['optimizer']} | {row['accuracy']:.4f} | {row['macro_precision']:.4f} | {row['macro_recall']:.4f} | {row['parameters']:,} | {row['fp32_weights_kib']:.2f} | {row['macs']:,} | {seconds} |")
    if missing:
        lines += ['', '**Final Model B optimizer comparison is incomplete:** missing '+', '.join(missing)+'. Train those optimizers on the same final architecture and recipe.']
    else:
        lines += ['', '**Final Model B optimizer comparison is complete:** the primary AdamW run and both SGD baselines are included.']
    lines += ['', 'Operation counts cover batch-one Conv2d and Linear layers at 64×64, including SE convolutions. One MAC is approximately two FLOPs. Bias adds, activations, pooling, batch normalization, residual additions, SE gate multiplication, memory traffic and framework overhead are excluded. These counts are not latency.', '',
        'Model A uses unaugmented ToTensor inputs. Final Model B uses train-only normalization, augmentation, a primary AdamW optimizer, a longer schedule and a larger custom architecture; MobileNetV2 and EfficientNet-B0 use pretrained weights and ImageNet preprocessing. Accuracy compares complete recipes, not an isolated convolution change. FP32 weights exclude runtime activations and optimizer state. Epoch times from different devices/runs cannot establish architecture speedups.', '',
        "Model A's saved metadata identifies the dataset and class order but lacks split manifest hashes, so that run's exact image split cannot be independently proven. Model B and MobileNetV2 manifest hashes match the canonical split.", '',
        "EfficientNet-B0 records class order and split path, but not manifest hashes or dataset fingerprint; "
        "its exact split provenance cannot be independently established from saved run metadata. "
        "The final group PDF remains unfinished. One seed on an image-level split does not establish "
        "statistical superiority or geographic generalization."]
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-b-run',type=Path,nargs='+',required=True)
    parser.add_argument('--model-a-run',type=Path,default=ROOT/'results/model_a/modelA_cuda_run')
    parser.add_argument('--mobilenet-run',type=Path,default=ROOT/'results/mobilenet_v2/run_01')
    parser.add_argument('--efficientnet-run',type=Path,default=ROOT/'results/efficientnet_b0/adam/run_20261009_193426')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    comparison(args.model_b_run,args.model_a_run,args.mobilenet_run,args.output_dir,args.efficientnet_run)
    print(f'Saved comparison: {args.output_dir}')

if __name__=='__main__': main()
