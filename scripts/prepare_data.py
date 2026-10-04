"""Prepare the fixed, collaborative EuroSAT RGB 70/15/15 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from en3150_a03.data import extract_zip_safely, prepare_splits, verify_splits  # noqa: E402

SOURCE_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
SOURCE_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"


def verify_archive(archive: Path) -> None:
    """Check the downloaded ZIP against Zenodo's published MD5."""
    digest = hashlib.md5()  # noqa: S324 — checksum verification, not cryptography
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_MD5:
        raise ValueError(f"Incorrect EuroSAT RGB archive checksum: {archive}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data/raw/EuroSAT_RGB")
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "data/raw/EuroSAT_RGB.zip")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/splits_torch_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="fetch official RGB ZIP if missing")
    parser.add_argument("--verify-only", action="store_true", help="audit existing CSVs and images")
    args = parser.parse_args(argv)

    if args.verify_only:
        result = verify_splits(args.image_root, args.split_dir)
        print(json.dumps(result, sort_keys=True))
        return result

    if not args.image_root.is_dir():
        if not args.archive.is_file():
            if not args.download:
                parser.error("RGB images/archive absent: supply --download or --archive")
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            partial = args.archive.with_name(args.archive.name + ".part")
            if partial.exists():
                raise FileExistsError(f"Remove incomplete download manually: {partial}")
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify_archive(partial)
            partial.rename(args.archive)
        verify_archive(args.archive)
        extracted = extract_zip_safely(args.archive, args.image_root.parent)
        if extracted.resolve() != args.image_root.resolve():
            raise ValueError(f"Extracted to unexpected directory: {extracted}")
    elif args.archive.is_file():
        verify_archive(args.archive)

    counts = prepare_splits(args.image_root, args.split_dir, seed=args.seed)
    audit = verify_splits(args.image_root, args.split_dir)
    result = {"counts": counts, "total": audit["total"],
              "dataset_sha256": audit["dataset_sha256"]}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
