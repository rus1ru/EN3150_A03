# EN3150 Assignment 03 — Resource-Constrained CNN

A shared project for comparing two custom CNNs (standard and depthwise-separable) with two fine-tuned lightweight pretrained networks on low-resolution imagery.

**Status:** EuroSAT RGB dataset preparation, frozen PyTorch split, and shared `DataLoader`s are implemented and tested. Models, training/evaluation commands, and experiment results are **not implemented yet**. Other configuration values are starting points.

## Assignment requirements

- Use EuroSAT RGB for image classification; it is not CIFAR-10.
- Images must be at most 64 × 64 pixels. Save a reproducible 70% train / 15% validation / 15% test split.
- Design Model A with standard convolutions and max-pooling; design Model B with depthwise-separable convolutions and no more than 100,000 trainable parameters. Document and calculate both parameter counts.
- Choose and justify a primary optimizer and learning rate; compare with SGD and SGD with momentum.
- Train both custom models for at least 20 epochs. Report loss curves, test accuracy, precision, recall, and confusion matrices.
- Fine-tune two lightweight pretrained models on the **same split** and compare accuracy, memory, and compute.
- Submit a PDF report and runnable code separately. See [`docs/task_checklist.pdf`](docs/task_checklist.pdf) for the full checklist.

## Dataset and collaboration

EuroSAT RGB was downloaded from the official Zenodo record and verified against its published archive checksum. It contains **27,000 images** in 10 classes. The canonical, class-stratified split uses seeded PyTorch `random_split` per class: **18,900 train / 4,050 validation / 4,050 test**. The small CSV/JSON manifests are committed under `data/splits_torch_v1/`; raw images and ZIPs stay local in `data/raw/`. The old `data/splits/` scaffold is not used. See [`data/README.md`](data/README.md) for source, setup, class mapping, and the exact split contract.

In a fresh clone, from the project root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision Pillow
.venv/bin/python scripts/prepare_data.py --download --split-dir data/splits_torch_v1
.venv/bin/python scripts/prepare_data.py --verify-only --split-dir data/splits_torch_v1
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_splits.py -v
```

Each teammate downloads their own images, verifies the committed split, and trains on the **same file IDs**. Never regenerate a split per model. PRs changing these manifests need explicit group review.

## Layout

- `configs/base.yaml`: common seed, paths, image size, and split proportions.
- `configs/models/`: separate configuration for each of the four models.
- `configs/optimizers/`: separate optimizer settings; learning rates remain undecided.
- `src/en3150_a03/models/`: one Python module per model. Shared data/training/evaluation code is in `src/en3150_a03/`.
- `scripts/`: planned CLI entry points for preparing data, training one model, evaluating checkpoints, and generating report assets.
- `data/splits_torch_v1/`: committed PyTorch-generated image IDs, labels, class order, checksums, and split metadata.
- `results/<model>/<optimizer>/`: per-run metrics, history, and checkpoints (checkpoints are ignored by Git). Use distinct run IDs within these folders if repeating experiments.
- `tests/`: passing split/data tests; model-shape, parameter-count and metric tests remain placeholders.
- `report/`: final report source and generated figures/tables.
- `docs/`: task checklist and experiment notes.

## Workflow to implement

1. Dataset and shared PyTorch split are in place; agree on model-specific preprocessing, optimizer hyperparameters, and experiment settings.
2. `scripts/prepare_data.py` creates/verifies `data/splits_torch_v1/{train,val,test}.csv`. Do not replace the published split; make a new version if it must change.
3. Implement models independently in `src/en3150_a03/models/` and verify input/output shapes and parameter counts.
4. Implement the shared training/evaluation pipeline and separate model + optimizer configurations. Select checkpoints using validation data only.
5. Train each model independently. Save its configuration, checkpoint, epoch times, training/validation history, and test metrics in its own results folder.
6. Generate tables and plots from saved metrics, then write and verify the report.

**Dataset preparation and PyTorch DataLoaders work. Training/evaluation commands do not work yet.** Other script entry points intentionally stop with “not implemented”; the model and metric test files remain placeholders.

## Collaboration and reproducibility

Use feature branches and pull requests; commit work regularly. All runs must use the same saved split files and metric definitions. Do not tune on the held-out test set. Track small metrics and manifests, but not raw images or large checkpoints. Record software versions and hardware alongside measured times. The GitHub repository is private by default; invite collaborators and the examiner if they need direct access.
