"""Model A training and checkpoint smoke tests."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from en3150_a03.model_a_evaluate import evaluate_model, load_checkpoint
from en3150_a03.models.model_a import ModelA
from en3150_a03.model_a_train import make_optimizer, train_model


class ModelTrainingTests(unittest.TestCase):
    def test_training_saves_checkpoint_curve_history_and_evaluable_weights(self):
        torch.manual_seed(1)
        samples = torch.rand(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(samples, labels), batch_size=2)
        model = ModelA(num_classes=2)
        optimizer = make_optimizer(model, "adam", 0.001)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            summary = train_model(
                model, loader, loader, optimizer, torch.device("cpu"), 1, output_dir,
                "adam", 0.001,
            )
            checkpoint_model = ModelA(num_classes=2)
            load_checkpoint(checkpoint_model, output_dir / "best_model.pt", torch.device("cpu"))
            metrics = evaluate_model(
                checkpoint_model, loader, ["A", "B"], torch.device("cpu")
            )

            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 4)
            self.assertEqual(metrics["trainable_parameters"], 25_730)
            self.assertTrue((output_dir / "training_history.csv").is_file())
            self.assertTrue((output_dir / "loss_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
