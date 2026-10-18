"""Entry-point guards, cross-version data setup and teammate result integration."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import importlib.util

from PIL import Image
from en3150_a03.data import prepare_splits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_model_b", ROOT / "scripts/compare_model_b.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ModelBIntegrationTests(unittest.TestCase):
    def test_cli_rejects_short_runs_before_touching_data(self):
        for flag in (["--epochs", "19"], ["--run-id", ".."], ["--optimizers", "adam", "adam"]):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_model_b_experiments.py"), *flag],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)

    def test_setup_verifies_existing_split_without_version_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "EuroSAT_RGB"
            for label in ("A", "B"):
                folder = images / label
                folder.mkdir(parents=True)
                for i in range(10):
                    Image.new("RGB", (8, 8), (i*20, 70, 100)).save(folder / f"{i}.jpg")
            splits = root / "splits"
            prepare_splits(images, splits)
            meta = json.loads((splits / "metadata.json").read_text())
            meta["torch_version"] = "different-historical-version"
            (splits / "metadata.json").write_text(json.dumps(meta))
            before = {p.name: p.read_bytes() for p in splits.iterdir()}
            result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_model_b_data.py"),
                "--image-root", str(images), "--split-dir", str(splits)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["total"], 20)
            self.assertEqual(before, {p.name: p.read_bytes() for p in splits.iterdir()})

    def test_comparison_reads_teammate_artifacts_without_modifying_them(self):
        # Synthetic Model B metrics exercise integration; never saved as experiment results.
        a = ROOT / "results/model_a/modelA_20261011"
        mobile = ROOT / "results/mobilenet_v2/run_01"
        tracked = [p for folder in (a, mobile) for p in folder.rglob("*") if p.is_file()]
        before = {p: p.read_bytes() for p in tracked}
        metadata = json.loads((ROOT / "data/splits_torch_v1/metadata.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = root / "b"
            b.mkdir()
            config = dict(dataset_sha256=metadata["dataset_sha256"], manifest_sha256=metadata["manifest_sha256"])
            (b / "experiment_config.json").write_text(json.dumps(config))
            result = copy.deepcopy(json.loads((a / "comparison.json").read_text())[0])
            result["trainable_parameters"] = 5589
            (b / "comparison.json").write_text(json.dumps([result]))
            rows = compare.comparison(b, a, mobile, root / "combined")
            self.assertEqual(len(rows), 5)
            self.assertEqual(next(r for r in rows if r["model"] == "ModelB")["macs"], 1_579_328)
            full = compare.comparison(b, a, mobile, root / "with_efficientnet",
                ROOT / "results/efficientnet_b0/adam/run_20261009_193426")
            efficientnet = next(r for r in full if r["model"] == "EfficientNet-B0")
            self.assertEqual(len(full), 6)
            self.assertGreater(efficientnet["accuracy"], 0.97)
            self.assertEqual(before, {p: p.read_bytes() for p in tracked})
            config["manifest_sha256"] = {"bad": "split"}
            (b / "experiment_config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                compare.comparison(b, a, mobile, root / "rejected")

if __name__ == "__main__":
    unittest.main()
