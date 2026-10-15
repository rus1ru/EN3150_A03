"""Offline tests: native resolution, transfer stages, metrics and saved checkpoint."""
import json
import tempfile
import unittest
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from en3150_a03.models.mobilenet_v2 import (
    build_mobilenet_v2, set_training_stage, set_train_mode,
)
from en3150_a03.data import prepare_splits
from en3150_a03.mobilenet_train import evaluate, run_epoch, run


class MobileNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_count_and_frozen_batchnorm(self):
        model = build_mobilenet_v2(pretrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2236682)
        set_training_stage(model, True)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 12810)
        set_train_mode(model, True)
        bn = next(m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
        before = bn.running_mean.clone()
        model(torch.randn(2, 3, 64, 64)).sum().backward()
        self.assertTrue(torch.equal(before, bn.running_mean))
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))
        self.assertIsNotNone(model.classifier[1].weight.grad)
        set_training_stage(model, False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_training_evaluation_and_reload(self):
        model = build_mobilenet_v2(pretrained=False)
        loader = DataLoader(TensorDataset(torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 0, 1])), batch_size=2)
        old = model.features[0][0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss, accuracy = run_epoch(model, loader, torch.device('cpu'), optimizer)
        self.assertGreater(loss, 0)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertFalse(torch.equal(old, model.features[0][0].weight))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = evaluate(model, loader, torch.device('cpu'), [str(i) for i in range(10)], output)
            self.assertEqual(sum(map(sum, result['confusion_matrix'])), 4)
            self.assertEqual(result['saved_model_size_bytes'], (output / 'model_weights.pt').stat().st_size)
            self.assertAlmostEqual(result['fp32_parameter_size_mb'], 8.946728)
            self.assertTrue(0 <= result['macro_f1'] <= 1)
            restored = build_mobilenet_v2(pretrained=False)
            restored.load_state_dict(torch.load(output / 'model_weights.pt', weights_only=True))
            restored.eval()
            model.eval()
            inputs = torch.randn(1, 3, 64, 64)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(inputs), restored(inputs)))

    def test_two_stage_pipeline_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ['A', 'B']:
                folder = root / 'images' / label
                folder.mkdir(parents=True)
                for index in range(10):
                    Image.new('RGB', (64, 64), (index * 20, 40, 80)).save(folder / f'{index}.jpg')
            splits = root / 'splits'
            prepare_splits(root / 'images', splits)
            original = {p.name: p.read_bytes() for p in splits.iterdir()}
            args = SimpleNamespace(seed=42, device='cpu', output_dir=root / 'train_run',
                data_root=root / 'images', split_dir=splits, batch_size=7, num_workers=0,
                epochs=2, warmup_epochs=1, head_lr=1e-3, finetune_lr=1e-4,
                evaluate_only=False, checkpoint=None)
            with patch('en3150_a03.mobilenet_train.build_mobilenet_v2',
                       side_effect=lambda n, pretrained: build_mobilenet_v2(n, pretrained=False)), contextlib.redirect_stdout(io.StringIO()):
                run(args)
                training_result = json.loads((args.output_dir / 'test_metrics.json').read_text())
                args.checkpoint = args.output_dir / 'best_model.pt'
                args.output_dir = root / 'evaluation_run'
                args.evaluate_only = True
                run(args)
            evaluation_result = json.loads((args.output_dir / 'test_metrics.json').read_text())
            self.assertEqual(training_result['confusion_matrix'], evaluation_result['confusion_matrix'])
            self.assertEqual(original, {p.name: p.read_bytes() for p in splits.iterdir()})
            self.assertTrue((args.output_dir / 'confusion_matrix.png').is_file())


if __name__ == '__main__':
    unittest.main()
