"""Saved-result report assets never require model checkpoints or a test rerun."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.make_figures import generate

ROOT = Path(__file__).resolve().parents[1]


class SavedFigureTests(unittest.TestCase):
    def test_efficientnet_assets_match_saved_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = ROOT / "results/efficientnet_b0/adam/run_20261009_193426/metrics.json"
            shutil.copyfile(source, folder / "metrics.json")
            created = generate(folder, "efficientnet-b0")
            self.assertEqual(len(created), 3)
            self.assertTrue(all(path.stat().st_size > 0 for path in created))
            metrics = json.loads(source.read_text())["test"]
            with created[0].open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0][1:], metrics["class_order"])
            self.assertEqual([[int(value) for value in row[1:]] for row in rows[1:]],
                             metrics["confusion_matrix"])
            with self.assertRaises(FileExistsError):
                generate(folder, "efficientnet-b0")

    def test_model_a_matrix_matches_saved_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = ROOT / "results/model_a/modelA_20261011/adam"
            shutil.copyfile(source / "test_metrics.json", folder / "test_metrics.json")
            shutil.copyfile(source / "training_history.csv", folder / "training_history.csv")
            created = generate(folder, "model-a")
            self.assertEqual(len(created), 2)
            with created[0].open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual([[int(value) for value in row[1:]] for row in rows[1:]],
                             json.loads((folder / "test_metrics.json").read_text())["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
