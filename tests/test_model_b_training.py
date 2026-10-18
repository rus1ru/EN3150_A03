"""Offline Model B training, interrupted recovery and frozen-split integration."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from en3150_a03.data import prepare_splits
from en3150_a03.model_b_experiment import run, seed_run
from en3150_a03.model_b_train import train_model
from en3150_a03.models.model_b import ModelB


class ModelBTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def loader(self):
        images = torch.arange(6*3*8*8, dtype=torch.float32).reshape(6, 3, 8, 8) / (6*3*8*8)
        return DataLoader(TensorDataset(images, torch.tensor([0, 1, 0, 1, 0, 1])),
                          batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))

    def test_interrupted_resume_matches_uninterrupted_weights(self):
        device = torch.device("cpu")
        contract = {"architecture": "test", "class_to_idx": {"A": 0, "B": 1}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            seed_run(42)
            full_loader = self.loader()
            train_model(ModelB(2), full_loader, full_loader, device, 2, root / "full", "adam", contract)
            seed_run(42)
            loader = self.loader()
            train_model(ModelB(2), loader, loader, device, 1, root / "resumed", "adam", contract)
            # A fresh process would construct a new model/loader and consume RNG.
            seed_run(123)
            loader = self.loader()
            train_model(ModelB(2), loader, loader, device, 2, root / "resumed", "adam", contract, resume=True)
            full = torch.load(root / "full/last_model.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last_model.pt", weights_only=True)
            for key in full["model_state_dict"]:
                self.assertTrue(torch.equal(full["model_state_dict"][key], resumed["model_state_dict"][key]), key)
            self.assertEqual([r["val_loss"] for r in full["history"]], [r["val_loss"] for r in resumed["history"]])
            with self.assertRaises(ValueError):
                train_model(ModelB(2), loader, loader, device, 2, root / "resumed", "adam", {"wrong": True}, resume=True)
            with self.assertRaises(FileExistsError):
                train_model(ModelB(2), loader, loader, device, 2, root / "resumed", "adam", contract)

    def test_pipeline_uses_frozen_manifests_and_shared_evaluator(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for label, color in (("A", (230, 20, 30)), ("B", (20, 210, 30))):
                folder = root / "images" / label
                folder.mkdir(parents=True)
                for i in range(10):
                    Image.new("RGB", (64, 64), (color[0], color[1], i*10)).save(folder / f"{i}.jpg")
            splits = root / "splits"
            prepare_splits(root / "images", splits)
            original = {p.name: p.read_bytes() for p in splits.iterdir()}
            # Internal runner allows tiny fixtures; CLI enforces the 20-epoch minimum.
            args = SimpleNamespace(device="cpu", seed=42, data_root=root / "images", split_dir=splits,
                batch_size=4, num_workers=0, epochs=2, results_dir=root / "results", run_id="fixture",
                optimizers=["adam", "sgd", "sgd_momentum"], resume=False)
            run(args)
            output = args.results_dir / args.run_id
            metrics = json.loads((output / "adam/test_metrics.json").read_text())
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 2)
            self.assertEqual(metrics["class_names"], ["A", "B"])
            self.assertEqual(metrics["best_epoch"], json.loads((output / "adam/training_summary.json").read_text())["best_epoch"])
            for name in args.optimizers:
                rows = list(csv.DictReader((output / name / "training_history.csv").open()))
                self.assertEqual(len(rows), 2)
                self.assertTrue((output / name / "confusion_matrix.png").is_file())
            before = (output / "adam/test_metrics.json").read_bytes()
            args.resume = True
            run(args)
            self.assertEqual(before, (output / "adam/test_metrics.json").read_bytes())
            self.assertEqual(original, {p.name: p.read_bytes() for p in splits.iterdir()})
            args.batch_size = 8
            with self.assertRaises(ValueError):
                run(args)

if __name__ == "__main__":
    unittest.main()
