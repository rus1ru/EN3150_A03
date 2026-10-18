"""Download EuroSAT and VERIFY the published split; never regenerate manifests.

Unlike first-time split construction, this works across PyTorch versions because
committed file IDs, rather than a particular random_split implementation, win.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from en3150_a03.data import extract_zip_safely, verify_splits
from prepare_data import SOURCE_URL, verify_archive


def prepare(image_root, archive, split_dir):
    image_root, archive, split_dir = map(Path, (image_root, archive, split_dir))
    if image_root.name != "EuroSAT_RGB":
        raise ValueError("Image root must end in EuroSAT_RGB, matching the official archive")
    if not image_root.exists():
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".zip.part")
            try:
                with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                verify_archive(partial)
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        verify_archive(archive)
        extract_zip_safely(archive, image_root.parent)
    return verify_splits(image_root, split_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/EuroSAT_RGB")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/EuroSAT_RGB.zip")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits_torch_v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.image_root, args.archive, args.split_dir), indent=2))

if __name__ == "__main__":
    main()
