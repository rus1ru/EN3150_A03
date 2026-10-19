"""Entry-point guards, cross-version data setup and teammate result integration."""
import shutil
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
        for flag in (["--run-id", "guard", "--epochs", "19"], ["--run-id", ".."], ["--run-id", "guard", "--optimizer", "invalid"]):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_model_b_v2.py"), *flag],
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
        # Read the completed final run; all comparison output stays in a temporary directory.
        a = ROOT / "results/model_a/modelA_cuda_run"
        mobile = ROOT / "results/mobilenet_v2/run_01"
        tracked = [p for folder in (a, mobile) for p in folder.rglob("*") if p.is_file()]
        before = {p: p.read_bytes() for p in tracked}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = root / "b"
            shutil.copytree(ROOT / "results/model_b/modelB_v2_adamw_01", b)
            config = json.loads((b / "experiment_config.json").read_text())
            rows = compare.comparison(b, a, mobile, root / "combined")
            self.assertEqual(len(rows), 5)
            self.assertEqual(next(r for r in rows if r["model"] == "ModelB")["macs"], 21_358_976)
            full = compare.comparison(b, a, mobile, root / "with_efficientnet",
                ROOT / "results/efficientnet_b0/adam/run_20261009_193426")
            efficientnet = next(r for r in full if r["model"] == "EfficientNet-B0")
            self.assertEqual(len(full), 6)
            self.assertGreater(efficientnet["accuracy"], 0.97)
            self.assertEqual(before, {p: p.read_bytes() for p in tracked})
            self.assertIn("missing sgd, sgd_momentum", (root / "combined/comparison.md").read_text())
            config["manifest_sha256"] = {"bad": "split"}
            (b / "experiment_config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                compare.comparison(b, a, mobile, root / "rejected")

if __name__ == "__main__":
    unittest.main()
