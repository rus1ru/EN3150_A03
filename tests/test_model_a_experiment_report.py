"""Tests for the saved Model A experiment report."""

import tempfile
import unittest
from pathlib import Path

from scripts.run_model_a_experiments import write_experiment_report


class ExperimentReportTests(unittest.TestCase):
    def test_report_contains_rationale_metrics_and_loss_curve_links(self):
        result = {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "momentum": None,
            "accuracy": 0.8,
            "macro_precision": 0.79,
            "macro_recall": 0.8,
            "trainable_parameters": 25_994,
            "estimated_float32_size_kb": 101.54,
            "checkpoint_file_size_kb": 103.0,
            "mean_epoch_seconds": 12.3,
            "best_epoch": 4,
            "class_names": ["AnnualCrop", "Forest"],
            "confusion_matrix": [[8, 2], [2, 8]],
            "per_class": {
                "AnnualCrop": {"precision": 0.8, "recall": 0.8, "support": 10},
                "Forest": {"precision": 0.8, "recall": 0.8, "support": 10},
            },
        }
        config = {
            "dataset_sha256": "abc",
            "class_names": result["class_names"],
            "epochs_per_optimizer": 20,
            "batch_size": 64,
            "seed": 42,
            "device": "cpu",
            "hardware": {"processor": "CPU", "platform": "Windows"},
            "python_version": "3.13.2",
            "pytorch_version": "2.x",
        }

        with tempfile.TemporaryDirectory() as temporary:
            report_path = write_experiment_report(Path(temporary), config, [result])
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("25,994", report)
        self.assertIn("101.54 KiB", report)
        self.assertIn("0.8000", report)
        self.assertIn("Confusion matrix", report)
        self.assertIn("Per-class precision and recall", report)
        self.assertIn("./adam/loss_curves.png", report)
        self.assertIn("Activation choice", report)
        self.assertIn("optimizer rationale", report.lower())


if __name__ == "__main__":
    unittest.main()
