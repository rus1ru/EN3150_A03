# Dataset

**No dataset is included or downloaded.** Proposed source: EuroSAT RGB — https://github.com/phelber/eurosat . Confirm the source/download location, version and licensing before preparation.

Put the extracted images in `data/raw/` (ignored by Git). Implement `scripts/prepare_data.py` to validate images and labels and generate `data/splits/train.csv`, `val.csv`, and `test.csv` with image IDs/paths and labels. Keep these manifests in Git so every model uses the same 70/15/15 split. Document any rejected images and class ordering here.
