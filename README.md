# EN3150 Assignment 03 — Resource-Constrained CNN

A shared project for comparing two custom CNNs (standard and depthwise-separable) with two fine-tuned lightweight pretrained networks on low-resolution imagery.

**Status:** The frozen EuroSAT RGB split and all four model experiments are implemented. Model B has 20-epoch Adam, SGD, and SGD-with-momentum results. The final group PDF report remains to be written.

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

## Model A: compact standard CNN

`src/en3150_a03/models/model_a.py` implements three repeated standard-convolution blocks for 64×64 RGB input. Each block uses a 3×3 `Conv2d` with stride 1 and padding 1, ReLU, then 2×2 max pooling with stride 2. Channel/spatial sizes are `3×64×64 → 16×32×32 → 32×16×16 → 64×8×8`. Adaptive average pooling reduces the final map to 64 values, followed by fully connected layers `64 → 32 → 10` (10 EuroSAT classes).

| Layer | Parameter calculation | Trainable parameters |
| --- | --- | ---: |
| Conv2d, 3→16, 3×3 + bias | 16 × (3 × 3 × 3 + 1) | 448 |
| Conv2d, 16→32, 3×3 + bias | 32 × (16 × 3 × 3 + 1) | 4,640 |
| Conv2d, 32→64, 3×3 + bias | 64 × (32 × 3 × 3 + 1) | 18,496 |
| Global average pool | No learned parameters | 0 |
| Linear, 64→32 + bias | 64 × 32 + 32 | 2,080 |
| Linear, 32→10 + bias | 32 × 10 + 10 | 330 |
| **Total** |  | **25,994** |

At four bytes per FP32 weight, the estimated weights-only footprint is `25,994 × 4 / 1,024 = 101.54 KiB` (103.98 decimal kB). The actual saved `.pt` size, including checkpoint metadata, is recorded separately. ReLU is used because it is a simple max-with-zero operation supported efficiently by common CPU, DSP, and quantized inference kernels; unlike sigmoid/tanh, it does not require an exponential and is less prone to saturating small gradients. In-place ReLUs can reuse activation storage. Max pooling and global average pooling reduce later activation and dense-layer cost.

## Train and compare Model A optimizers

With EuroSAT already prepared, run the following from the project root:

```powershell
python scripts/run_model_a_experiments.py --epochs 20 --batch-size 64
```

The runner requires at least 20 epochs; use a larger value for longer runs. It selects the checkpoint with the lowest validation loss and evaluates that checkpoint on the held-out test set exactly once per optimizer. The same canonical split, batch size, seed, model initialization seed, and loss are used for each comparison. No test data is used for checkpoint selection.

| Optimizer | Learning rate | Momentum | Role and rationale |
| --- | ---: | ---: | --- |
| Adam | 0.001 | — | Primary: per-parameter adaptive steps are a robust starting point for training a compact CNN without a learning-rate schedule. |
| SGD | 0.01 | — | Plain-SGD reference; uses the same fixed learning rate throughout. |
| SGD | 0.01 | 0.9 | Momentum reference; accumulates a smoothed gradient direction, which can accelerate progress in consistently descending directions and damp batch-to-batch oscillation. Excessive momentum can overshoot and make validation performance less stable; compare the recorded curves and test metrics rather than assuming it wins. |

All runs use cross-entropy loss, no weight decay, and no scheduler. The selected checkpoint rule, optimizer settings, and split are saved with each result.

The run creates `results/model_a/<run-id>/experiment_report.md`, a single shareable report with the architecture/parameter calculations and hardware/optimizer justifications, the measured comparison table, per-optimizer confusion matrices and class scores, plus links to the loss curves. The same directory contains experiment settings/environment and, for each optimizer, `best_model.pt`, `training_history.csv`, `loss_curves.png`, `training_summary.json`, and `test_metrics.json`, plus machine-readable `comparison.csv` and `comparison.json`. Test metrics include accuracy, macro and per-class precision/recall, confusion matrix, trainable parameter count, estimated FP32 weight size, actual checkpoint size, best validation epoch, and mean training-plus-validation seconds per epoch. Device defaults to CUDA when available and otherwise CPU. Set `--device cpu`/`--device cuda`, or use `--num-workers` to configure the input pipeline. Keep the test split for final reporting only.

To train just one optimizer, use `scripts/train_model_a.py --optimizer adam --output-dir results/model_a/adam_single --epochs 20`. To evaluate an existing checkpoint independently, use `scripts/evaluate_model_a.py --checkpoint <path-to-best_model.pt>`.

## Layout

- `configs/base.yaml`: common seed, paths, image size, and split proportions.
- `configs/models/`: separate configuration for each of the four models.
- `configs/optimizers/`: documented Adam, SGD, and momentum-SGD comparison settings.
- `src/en3150_a03/models/`: one Python module per model. Model A-specific training, evaluation, and metrics live in `model_a_train.py`, `model_a_evaluate.py`, and `model_a_metrics.py`; shared dataset/split functionality remains in `data.py`.
- `scripts/`: CLI entry points for data preparation, Model A-specific training/optimizer comparison (`train_model_a.py`, `run_model_a_experiments.py`), checkpoint evaluation (`evaluate_model_a.py`), and report figures.
- `data/splits_torch_v1/`: committed PyTorch-generated image IDs, labels, class order, checksums, and split metadata.
- `results/<model>/<optimizer>/`: per-run metrics, history, and checkpoints (checkpoints are ignored by Git). Use distinct run IDs within these folders if repeating experiments.
- `tests/`: Model A tests use `test_model_a_*.py`; shared dataset/split tests remain in `test_splits.py`.
- `report/`: final report source and generated figures/tables.
- `docs/`: task checklist and experiment notes.

## EfficientNet-B0 (independent runnable component)

The EfficientNet-B0 implementation is in `src/en3150_a03/models/efficientnet_b0.py`.
It replaces the classifier with 10 outputs, fine-tunes all weights, keeps images at
64×64, and applies ImageNet normalization for pretrained weights. Training uses
only the committed split IDs; the best checkpoint is selected by validation accuracy.
The test split is evaluated once after training, not used to choose hyperparameters.

```bash
# From the project root, install CUDA PyTorch into the existing .venv (NVIDIA GPU).
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 "torch==2.4.1+cu121" "torchvision==0.19.1+cu121"
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# The first run downloads torchvision's ImageNet B0 weights.
.venv/bin/python scripts/train_efficientnet_b0.py --epochs 10 --batch-size 32
# Quick pipeline check without a weight download or meaningful benchmark metrics:
.venv/bin/python scripts/train_efficientnet_b0.py --scratch --epochs 1 --batch-size 2 --max-batches 1 --output /tmp/en3150-efficientnet-smoke
PYTHONPATH=src .venv/bin/python -m unittest tests/test_efficientnet_b0.py -v
```

The default output is `results/efficientnet_b0/adam/{best.pt,metrics.json}`.
Use a new `--output` path per experiment to avoid replacing a checkpoint.
The values in `configs/models/efficientnet_b0.yaml` are starting choices, not
validated performance claims. The training script selects CUDA automatically when available; otherwise it uses CPU.
Model B is implemented and trained; see [`docs/model_b.md`](docs/model_b.md) and the [four-model comparison](results/comparisons/all_models_20261018/comparison.md).

## Saved four-model results

The linked [comparison table](results/comparisons/all_models_20261018/comparison.md) reads recorded test metrics and profiles at 64×64 resolution; it does not rerun training. Representative results are:

| Model and run | Test accuracy | Trainable parameters | Conv/linear MACs per image |
| --- | ---: | ---: | ---: |
| [Model A, Adam](results/model_a/modelA_20261011/adam/test_metrics.json) | 75.80% | 25,994 | 11,209,024 |
| [Model B, Adam](results/model_b/modelB_gpu_01/adam/test_metrics.json) | 59.14% | 5,589 | 1,579,328 |
| [MobileNetV2, pretrained](results/mobilenet_v2/run_01/test_metrics.json) | 97.65% | 2,236,682 | 24,461,312 |
| [EfficientNet-B0, pretrained](results/efficientnet_b0/adam/run_20261009_193426/metrics.json) | 97.85% | 4,020,358 | 31,979,392 |

Model B was also trained with plain SGD (18.99%) and SGD with momentum (74.20%). These are single-run scores; do not select a preferred optimizer from held-out test accuracy. The A/B architecture comparison holds widths and preprocessing constant, but pretrained networks use different training recipes. Epoch times were measured on different machines, so they do **not** establish relative hardware speed. Model A and EfficientNet-B0 saved metadata do not include manifest hashes; see the [comparison caveats](results/comparisons/all_models_20261018/comparison.md).

## Remaining workflow

1. Dataset and shared PyTorch split are in place; Model A settings are documented above. Agree on remaining model-specific preprocessing and experiment settings.
2. `scripts/prepare_data.py` creates/verifies `data/splits_torch_v1/{train,val,test}.csv`. Do not replace the published split; make a new version if it must change.
3. The Model A and Model B architectures, parameter counts and 20-epoch optimizer runs are saved in their respective `results/` folders. MobileNetV2 and EfficientNet-B0 are also trained.
4. Review the [four-model comparison](results/comparisons/all_models_20261018/comparison.md); model accuracies reflect different training recipes and hardware timings are not directly comparable.
5. Write, compile, and verify the final PDF report in `report/`, with citations and the group's required submission details.

**Code and saved model results are present; the final group PDF remains unfinished.**

## Collaboration and reproducibility

Use feature branches and pull requests; commit work regularly. All runs must use the same saved split files and metric definitions. Do not tune on the held-out test set. Track small metrics and manifests, but not raw images or large checkpoints. Record software versions and hardware alongside measured times. The GitHub repository is private by default; invite collaborators and the examiner if they need direct access.
