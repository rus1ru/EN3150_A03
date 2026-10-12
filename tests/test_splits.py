"""Behavioral tests for the shared EuroSAT split contract."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import ToTensor

from en3150_a03.data import (EuroSATDataset, extract_zip_safely, get_dataloaders,
                            prepare_splits, verify_splits)


class SplitContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / "EuroSAT_RGB"
        for label in ("AnnualCrop", "Forest"):
            folder = self.images / label
            folder.mkdir(parents=True)
            for i in range(20):
                Image.new("RGB", (64, 64), (i, 50, 100)).save(folder / f"{label}_{i:03}.jpg")

    def test_stratified_splits_are_stable_portable_and_disjoint(self):
        first = self.base / "splits1"
        second = self.base / "splits2"
        prepare_splits(self.images, first, seed=42)
        prepare_splits(self.images, second, seed=42)
        seen = set()
        for name, expected_per_class in (("train", 14), ("val", 3), ("test", 3)):
            self.assertEqual((first / f"{name}.csv").read_bytes(), (second / f"{name}.csv").read_bytes())
            with (first / f"{name}.csv").open(newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(Counter(record["label"] for record in records),
                             {"AnnualCrop": expected_per_class, "Forest": expected_per_class})
            for row in records:
                self.assertFalse(Path(row["path"]).is_absolute())
                self.assertTrue((self.images / row["path"]).is_file())
                self.assertNotIn(row["path"], seen)
                seen.add(row["path"])
        self.assertEqual(len(seen), 40)
        dataset = EuroSATDataset(self.images, first / "val.csv")
        image, target = dataset[0]
        self.assertEqual(image.size, (64, 64))
        self.assertIn(target, (0, 1))
        self.assertEqual(dataset.class_to_idx, {"AnnualCrop": 0, "Forest": 1})

    def test_uses_seeded_pytorch_split_with_shared_dataloaders(self):
        split_dir = self.base / "torch_splits"
        prepare_splits(self.images, split_dir, seed=42)
        metadata = json.loads((split_dir / "metadata.json").read_text())
        self.assertEqual(metadata["splitter"], "torch.utils.data.random_split per class")
        ordered = sorted((self.images / "AnnualCrop").glob("*.jpg"))
        expected = torch.utils.data.random_split(
            ordered, [14, 3, 3], generator=torch.Generator().manual_seed(42)
        )
        with (split_dir / "train.csv").open(newline="") as stream:
            actual = {row["path"] for row in csv.DictReader(stream)
                      if row["label"] == "AnnualCrop"}
        self.assertEqual(actual, {path.relative_to(self.images).as_posix() for path in expected[0]})
        loaders = get_dataloaders(self.images, split_dir, batch_size=4, transform=ToTensor())
        images, labels = next(iter(loaders["train"]))
        self.assertEqual(tuple(images.shape), (4, 3, 64, 64))
        self.assertEqual(tuple(labels.shape), (4,))

    def test_existing_split_cannot_be_replaced_by_different_seed(self):
        split_dir = self.base / "splits"
        prepare_splits(self.images, split_dir, seed=42)
        original = (split_dir / "train.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_splits(self.images, split_dir, seed=99)
        self.assertEqual((split_dir / "train.csv").read_bytes(), original)

    def test_tampered_manifest_is_rejected(self):
        split_dir = self.base / "splits"
        prepare_splits(self.images, split_dir)
        self.assertEqual(verify_splits(self.images, split_dir)["total"], 40)
        with (split_dir / "test.csv").open("a", encoding="utf-8") as stream:
            stream.write("Forest/Forest_001.jpg,Forest,1\n")
        with self.assertRaises(ValueError):
            verify_splits(self.images, split_dir)

    def test_manifest_line_endings_are_portable(self):
        split_dir = self.base / "splits"
        prepare_splits(self.images, split_dir)
        for name in ("train.csv", "val.csv", "test.csv", "class_to_idx.json", "metadata.json"):
            manifest = split_dir / name
            manifest.write_bytes(manifest.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(verify_splits(self.images, split_dir)["total"], 40)
        self.assertEqual(prepare_splits(self.images, split_dir)["train"], 28)

    def test_dataset_rejects_unsafe_path(self):
        split_dir = self.base / "splits"
        prepare_splits(self.images, split_dir)
        split = split_dir / "val.csv"
        split.write_text("path,label,label_index\n../outside.jpg,Forest,1\n")
        with self.assertRaises(ValueError):
            EuroSATDataset(self.images, split)

    def test_invalid_image_is_rejected_before_manifests_are_written(self):
        (self.images / "Forest" / "bad.jpg").write_text("not a JPEG")
        split_dir = self.base / "splits"
        with self.assertRaises(ValueError):
            prepare_splits(self.images, split_dir)
        self.assertFalse((split_dir / "train.csv").exists())

    def test_symlinked_image_outside_dataset_is_rejected(self):
        external = self.base / "external.jpg"
        Image.new("RGB", (64, 64)).save(external)
        (self.images / "Forest" / "linked.jpg").symlink_to(external)
        with self.assertRaises(ValueError):
            prepare_splits(self.images, self.base / "splits")

    def test_zip_extraction_rejects_traversal(self):
        archive = self.base / "danger.zip"
        with zipfile.ZipFile(archive, "w") as destination:
            destination.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            extract_zip_safely(archive, self.base / "raw")
        self.assertFalse((self.base / "escape.txt").exists())

    def test_cli_prepares_local_images_without_download(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
        split_dir = self.base / "portable"
        run = subprocess.run(
            [sys.executable, str(script), "--image-root", str(self.images),
             "--split-dir", str(split_dir), "--seed", "42"],
            capture_output=True, text=True, env=os.environ.copy(), check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue((split_dir / "metadata.json").is_file())
        self.assertEqual(verify_splits(self.images, split_dir)["total"], 40)


if __name__ == "__main__":
    unittest.main()
