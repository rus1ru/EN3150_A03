# EuroSAT RGB dataset and shared split

**Canonical split: `data/splits_torch_v1/`.** Every model must use these committed CSVs. `data/splits/` is an unused scaffold directory; its earlier local files are ignored. Never create an independent split for another experiment.

## Source and provenance

- Official dataset record: https://zenodo.org/records/7711810 (EuroSAT_RGB.zip, RGB JPEGs).
- Published ZIP MD5: `f46e308c4d50d4bf32fedad2d3d62f3b` (verified before extraction).
- Original project: https://github.com/phelber/eurosat . Cite the EuroSAT paper and dataset record in the report. Review the Sentinel data terms given in the record.
- Observed: 27,000 RGB images (64 × 64), 10 classes. Images and ZIP are **not** committed to Git.
- Dataset content SHA-256 (ordered relative paths plus image bytes): `548a976e7d3f714d69f2a93f1abaf6261494537faa33af10f762ad7614644dd1`.

## Set up on each collaborator's machine

From the project root, install a Python 3.12 environment for the dataset pipeline:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision Pillow
.venv/bin/python scripts/prepare_data.py --download --split-dir data/splits_torch_v1
.venv/bin/python scripts/prepare_data.py --verify-only --split-dir data/splits_torch_v1
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_splits.py -v
```

The first preparation downloads the official ZIP if absent, checks its checksum, extracts safely, and validates images. If downloaded manually, put `EuroSAT_RGB.zip` in `data/raw/` and omit `--download`. If images already exist, preparation verifies them and the committed manifests. An incorrect archive, image, or split fails instead of silently making a different split. Keep the ZIP and extracted images under `data/raw/`.

## Exact split contract

`torch.utils.data.random_split` splits **each alphabetically sorted class separately**, using one `torch.Generator().manual_seed(42)` across the classes. Seed, PyTorch version (`2.14.1+cpu` for the committed manifests), class order, per-class counts, dataset fingerprint, and CSV checksums are recorded in `data/splits_torch_v1/metadata.json`. The seed is only for constructing the split; the **committed CSVs**, not regenerated random indices, are the source of truth for every collaborator.

| Split | Images | Share |
| --- | ---: | ---: |
| Train | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Test | 4,050 | 15% |

Each CSV has `path,label,label_index` columns; paths are **relative to `data/raw/EuroSAT_RGB`** and portable across machines. The files are class-stratified and mutually disjoint. The test set must not be used for tuning, checkpoint choice, or architecture selection. This is an image-level random split, **not a geographically independent split**; adjacent satellite tiles may be correlated.

## Use in model training

```python
from pathlib import Path
from torchvision.transforms import ToTensor
from en3150_a03.data import get_dataloaders

loaders = get_dataloaders(
    image_root=Path("data/raw/EuroSAT_RGB"),
    split_dir=Path("data/splits_torch_v1"),
    batch_size=64,
    transform=ToTensor(),
)
images, labels = next(iter(loaders["train"]))
```

The loader audits the split and uses the same class-index mapping for all models. It shuffles only training data. For different train versus validation/test transforms, pass `train_transform=` and `eval_transform=`. Keep stochastic augmentation on training only; any normalization needed for pretrained weights belongs in the model-specific transform, not in the split itself.

## Collaboration rules

- Commit the small CSV/JSON manifests and data pipeline; do **not** commit raw images, ZIPs, or large checkpoints.
- Work on feature branches and submit PRs to `main`; review any proposed change to the split as a new version, not an overwrite.
- A fresh clone should verify the committed split after downloading EuroSAT. If verification fails, stop rather than training with divergent images.
