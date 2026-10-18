"""Reproducible EuroSAT RGB splitting and a shared map-style image dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, random_split

CSV_FIELDS = ("path", "label", "label_index")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")


def _canonical_manifest_bytes(content: bytes) -> bytes:
    """Normalize Git's Windows CRLF checkouts before comparing manifest checksums."""
    return content.replace(b"\r\n", b"\n")


def _safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or "\\" in raw or ":" in raw or any(
        part in ("", ".", "..") for part in raw.split("/")
    ):
        raise ValueError(f"Unsafe relative path: {raw!r}")
    return path


def _image_index(image_root: Path) -> tuple[list[str], dict[str, list[Path]], str]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    classes = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError("Expected at least two class directories")
    index: dict[str, list[Path]] = {}
    digest = hashlib.sha256()
    for label in classes:
        directory = image_root / label
        if directory.is_symlink():
            raise ValueError(f"Symlinked class directory: {directory}")
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(files) < 7:
            raise ValueError(f"Too few images for 70/15/15 split: {label}")
        index[label] = files
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Symlinked image: {path}")
            relative = path.relative_to(image_root).as_posix()
            try:
                with Image.open(path) as image:
                    if image.width > 64 or image.height > 64 or image.mode != "RGB":
                        raise ValueError(f"Expected RGB image <=64x64: {relative}")
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Invalid image: {relative}") from exc
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return classes, index, digest.hexdigest()


def _manifest_bytes(entries: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(entries)
    return output.getvalue().encode("utf-8")


def prepare_splits(image_root: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    """Create stable 70/15/15 class-stratified CSVs; never overwrite a published split."""
    image_root, output_dir = Path(image_root), Path(output_dir)
    classes, index, fingerprint = _image_index(image_root)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    rows: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    generator = torch.Generator().manual_seed(seed)
    for label in classes:
        files = list(index[label])
        train_end = round(len(files) * .70)
        val_end = train_end + round(len(files) * .15)
        subsets = random_split(files, [train_end, val_end - train_end,
                                       len(files) - val_end], generator=generator)
        for split, selected in zip(SPLIT_NAMES, subsets, strict=True):
            if not selected:
                raise ValueError(f"Empty {split} split for {label}")
            rows[split].extend({"path": path.relative_to(image_root).as_posix(),
                                "label": label, "label_index": class_to_idx[label]}
                               for path in selected)
    manifests = {f"{name}.csv": _manifest_bytes(rows[name]) for name in SPLIT_NAMES}
    manifests["class_to_idx.json"] = (json.dumps(class_to_idx, indent=2, sort_keys=True) + "\n").encode()
    metadata = {
        "dataset": "EuroSAT RGB", "source_record": "https://zenodo.org/records/7711810",
        "format_version": 1, "seed": seed, "splitter": "torch.utils.data.random_split per class",
        "torch_version": torch.__version__,
        "fractions": {"train": .70, "val": .15, "test": .15},
        "classes": classes, "class_to_idx": class_to_idx,
        "dataset_sha256": fingerprint,
        "counts": {name: len(rows[name]) for name in SPLIT_NAMES},
        "class_counts": {name: dict(sorted(Counter(row["label"] for row in rows[name]).items()))
                         for name in SPLIT_NAMES},
        "manifest_sha256": {name: hashlib.sha256(content).hexdigest()
                            for name, content in manifests.items()},
    }
    manifests["metadata.json"] = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    existing = [name for name in manifests if (output_dir / name).exists()]
    if existing:
        if len(existing) == len(manifests) and all(
            _canonical_manifest_bytes((output_dir / name).read_bytes())
            == _canonical_manifest_bytes(content)
            for name, content in manifests.items()
        ):
            return metadata["counts"]
        raise FileExistsError(f"Split directory differs from requested seed/data: {output_dir}. "
                              "Do not overwrite published manifests; use a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in manifests.items():
        (output_dir / name).write_bytes(content)
    return metadata["counts"]


def _load_rows(image_root: Path, split_csv: Path, class_to_idx: dict[str, int]) -> list[dict]:
    with split_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Unexpected manifest fields: {split_csv}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty split: {split_csv}")
    for row in rows:
        rel = _safe_relative_path(row["path"])
        if len(rel.parts) != 2 or row["label"] != rel.parts[0]:
            raise ValueError(f"Class/path mismatch: {row}")
        if row["label"] not in class_to_idx or str(class_to_idx[row["label"]]) != row["label_index"]:
            raise ValueError(f"Class/index mismatch: {row}")
        target = image_root.joinpath(*rel.parts)
        if not target.is_file() or not target.resolve().is_relative_to(image_root.resolve()):
            raise ValueError(f"Missing or unsafe image: {rel}")
    return rows


def verify_splits(image_root: Path, split_dir: Path) -> dict:
    """Audit CSV checksums, labels, overlap, completeness and dataset fingerprint."""
    image_root, split_dir = Path(image_root), Path(split_dir)
    metadata = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    class_to_idx = json.loads((split_dir / "class_to_idx.json").read_text(encoding="utf-8"))
    if class_to_idx != metadata["class_to_idx"]:
        raise ValueError("Class mapping differs from metadata")
    all_paths = set()
    counts = {}
    for split in SPLIT_NAMES:
        manifest = split_dir / f"{split}.csv"
        content = _canonical_manifest_bytes(manifest.read_bytes())
        if hashlib.sha256(content).hexdigest() != metadata["manifest_sha256"][manifest.name]:
            raise ValueError(f"Changed manifest: {manifest.name}")
        rows = _load_rows(image_root, manifest, class_to_idx)
        paths = {row["path"] for row in rows}
        if len(paths) != len(rows) or all_paths.intersection(paths):
            raise ValueError("Duplicate or overlapping image in splits")
        all_paths.update(paths)
        counts[split] = len(rows)
        if counts[split] != metadata["counts"][split]:
            raise ValueError(f"Count mismatch: {split}")
        if dict(sorted(Counter(row["label"] for row in rows).items())) != metadata["class_counts"][split]:
            raise ValueError(f"Per-class count mismatch: {split}")
    classes, index, fingerprint = _image_index(image_root)
    if classes != metadata["classes"] or fingerprint != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from the published split")
    if all_paths != {path.relative_to(image_root).as_posix()
                     for files in index.values() for path in files}:
        raise ValueError("Splits do not cover every image exactly once")
    return {"total": len(all_paths), "counts": counts, "dataset_sha256": fingerprint}


class EuroSATDataset:
    """Map-style PIL dataset. Pass a transform returning a tensor for PyTorch training."""

    def __init__(self, image_root: Path, split_csv: Path, transform: Callable | None = None):
        self.image_root = Path(image_root)
        self.transform = transform
        split_csv = Path(split_csv)
        self.class_to_idx = json.loads((split_csv.parent / "class_to_idx.json").read_text())
        self.classes = list(self.class_to_idx)
        self.records = _load_rows(self.image_root, split_csv, self.class_to_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        with Image.open(self.image_root / row["path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_index"])


def get_dataloaders(image_root: Path, split_dir: Path, batch_size: int = 64,
                    transform: Callable | None = None, train_transform: Callable | None = None,
                    eval_transform: Callable | None = None, num_workers: int = 0,
                    seed: int = 42) -> dict[str, DataLoader]:
    """Load the published split; only train shuffles and may use augmentation."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    verify_splits(image_root, split_dir)
    loaders = {}
    for name in SPLIT_NAMES:
        current_transform = ((train_transform or transform) if name == "train"
                             else (eval_transform or transform))
        if current_transform is None:
            raise ValueError("Provide a tensor transform (e.g. torchvision.transforms.ToTensor)")
        loaders[name] = DataLoader(
            EuroSATDataset(image_root, Path(split_dir) / f"{name}.csv", current_transform),
            batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
        )
    return loaders


def extract_zip_safely(archive: Path, destination: Path) -> Path:
    """Extract only regular files within destination; reject traversal and zip bombs."""
    archive, destination = Path(archive), Path(destination)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 50_000 or sum(entry.file_size for entry in entries) > 1_000_000_000:
            raise ValueError("Archive exceeds expected EuroSAT RGB size")
        names = set()
        for entry in entries:
            rel = _safe_relative_path(entry.filename.rstrip("/"))
            if entry.filename in names or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            names.add(entry.filename)
            if rel.parts[0] != "EuroSAT_RGB":
                raise ValueError(f"Unexpected archive layout: {entry.filename}")
        for entry in entries:
            target = destination.joinpath(*PurePosixPath(entry.filename).parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                if target.exists():
                    raise FileExistsError(f"Will not replace extracted file: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return destination / "EuroSAT_RGB"
