# EN3150 Assignment 03 — Resource-Constrained CNN

A shared project for comparing two custom CNNs (standard and depthwise-separable) with two fine-tuned lightweight pretrained networks on low-resolution imagery.

**Status:** All four model implementations and saved results are present. Current Model A and Model B v2 have verified 40-epoch runs; the report is compiled under `report/`. Group details and final human review remain pending. The full-pipeline notebook provides review and opt-in fresh training.

## Full-pipeline notebook

Open [`EN3150_A03_Full_Pipeline.ipynb`](EN3150_A03_Full_Pipeline.ipynb). It is standalone: data download/verification, split creation, custom architectures, training, evaluation, plots and comparison are defined inline. Only standard Python/ML dependencies are required; no repository imports or scripts are used. Set `TRAIN = True` and allow data download if needed for fresh experiments. It creates its own saved split and new output directories; it does not silently reuse historical report scores or claim identical historical split membership.

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

## Model A: resource-conscious standard CNN

`src/en3150_a03/models/model_a.py` implements four standard-convolution blocks for 64×64 RGB input. Each block uses a 3×3 `Conv2d` with stride 1 and padding 1, ReLU, then 2×2 max pooling with stride 2. Channel/spatial sizes are `3×64×64 → 32×32×32 → 64×16×16 → 96×8×8 → 112×4×4`. Adaptive average pooling reduces the final map to 112 values, followed by fully connected layers `112 → 192 → 10` (10 EuroSAT classes). The wider network is around 200,000 parameters, while pooling each block and global average pooling constrain feature-map and classifier costs.

| Layer | Parameter calculation | Trainable parameters |
| --- | --- | ---: |
| Conv2d, 3→32, 3×3 + bias | 32 × (3 × 3 × 3 + 1) | 896 |
| Conv2d, 32→64, 3×3 + bias | 64 × (32 × 3 × 3 + 1) | 18,496 |
| Conv2d, 64→96, 3×3 + bias | 96 × (64 × 3 × 3 + 1) | 55,392 |
| Conv2d, 96→112, 3×3 + bias | 112 × (96 × 3 × 3 + 1) | 96,880 |
| Global average pool | No learned parameters | 0 |
| Linear, 112→192 + bias | 112 × 192 + 192 | 21,696 |
| Linear, 192→10 + bias | 192 × 10 + 10 | 1,930 |
| **Total** |  | **195,290** |

At four bytes per FP32 weight, the estimated weights-only footprint is `195,290 × 4 / 1,024 = 762.85 KiB` (781.16 decimal kB). The actual saved `.pt` size, including checkpoint metadata, is recorded separately. ReLU is used because it is a simple max-with-zero operation supported efficiently by common CPU, DSP, and quantized inference kernels; unlike sigmoid/tanh, it does not require an exponential and is less prone to saturating small gradients. In-place ReLUs can reuse activation storage. Pooling after every convolution halves the spatial dimensions before the next wider layer; adaptive global average pooling keeps the classifier small rather than flattening the full 112×4×4 activation map.

## Train and compare Model A optimizers

With EuroSAT already prepared, run the following from the project root:

```powershell
python scripts/run_model_a_experiments.py --epochs 20 --batch-size 64
```

The runner requires at least 20 epochs; use a larger value for longer runs. It selects the checkpoint with the lowest validation loss and evaluates that checkpoint on the held-out test set exactly once per optimizer. The same canonical split, batch size, seed, model initialization seed, and loss are used for each comparison. No test data is used for checkpoint selection.

Model A training and evaluation default to the first CUDA GPU (`cuda:0`), and fail rather than silently falling back to CPU. Before running, check that this Python environment has a CUDA-enabled PyTorch build:

```powershell
py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU visible')"
```

`torch.version.cuda` must not be `None`, and `torch.cuda.is_available()` must be `True`. Install a CUDA-enabled PyTorch build into the same Python environment used to launch the experiment; choose the Windows/Pip/Python/CUDA command from [PyTorch's installation selector](https://pytorch.org/get-started/locally/). If multiple CUDA GPUs are visible, choose one with `--device cuda:1` (or another available index). Use `--device cpu` only to explicitly run on the CPU.

| Optimizer | Learning rate | Momentum | Role and rationale |
| --- | ---: | ---: | --- |
| Adam | 0.001 | — | Primary: per-parameter adaptive steps are a robust starting point for this resource-constrained CNN without a learning-rate schedule. |
| SGD | 0.01 | — | Plain-SGD reference; uses the same fixed learning rate throughout. |
| SGD | 0.01 | 0.9 | Momentum reference; accumulates a smoothed gradient direction, which can accelerate progress in consistently descending directions and damp batch-to-batch oscillation. Excessive momentum can overshoot and make validation performance less stable; compare the recorded curves and test metrics rather than assuming it wins. |

All runs use cross-entropy loss, no weight decay, and no scheduler. The selected checkpoint rule, optimizer settings, and split are saved with each result.

The run creates `results/model_a/<run-id>/experiment_report.md`, a single shareable report with the architecture/parameter calculations and hardware/optimizer justifications, the measured comparison table, per-optimizer confusion matrices and class scores, plus links to the loss curves. The same directory contains experiment settings/environment and, for each optimizer, `best_model.pt`, `training_history.csv`, `loss_curves.png`, `training_summary.json`, and `test_metrics.json`, plus machine-readable `comparison.csv` and `comparison.json`. Test metrics include accuracy, macro and per-class precision/recall, confusion matrix, trainable parameter count, estimated FP32 weight size, actual checkpoint size, best validation epoch, and mean training-plus-validation seconds per epoch. Training and evaluation default to `cuda:0` and stop with an explicit error if CUDA is unavailable; use `--device cpu` to opt into CPU execution. A CUDA-enabled PyTorch build and working NVIDIA driver are required. Use `--device cuda:1` to select another visible GPU, or use `--num-workers` to configure the input pipeline. Keep the test split for final reporting only. Use a new run ID after architecture changes; existing result folders are historical and are not overwritten.

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

## Controlled single-seed A/B study (completed)

New matched experiment: `results/controlled_ab/controlled_ab_seed42_20261019_215110/`. Six fixed recipes (Model A/B × Adam/SGD/momentum SGD), seed 42, 40 epochs, batch 64, train-only shared normalization and horizontal/vertical flips. Plain cross-entropy, FP32 (AMP/TF32 off), no weight decay/scheduler. Adam LR 0.001; SGD/momentum LR 0.01; momentum 0.9. No LR search. All six finish training before any test evaluation; checkpoint selection uses minimum validation loss. This compares complete architectures, not convolution type alone. Single-seed and prior-test-reuse limitations remain. Existing report/results are unchanged. All six runs completed and were checked: 40 epochs each, full 4,050-image test matrices, local best checkpoints, and saved plots. See [controlled results](results/controlled_ab/controlled_ab_seed42_20261019_215110/comparison.md). Test accuracy: Adam A/B 95.48%/96.69%; SGD 83.51%/93.06%; momentum SGD 95.95%/97.14%. The report now uses this controlled study for its A/B tables, figures and discussion; pretrained results remain unmatched references.

## Historical unmatched 40-epoch experiments (superseded for A/B reporting)

The confirmed Model A (195,290 parameters) and Model B v2 (96,538 parameters) completed **40 epochs per optimizer** on the RTX 4050. All six runs have local validation-best checkpoints, complete histories, test metrics, matrix CSV/PNG files, and loss curves. The [40-epoch comparison](results/comparisons/40epochs_20261019/comparison.md) includes both pretrained models as references.

| Optimizer | Model A test accuracy | Model B v2 test accuracy |
| --- | ---: | ---: |
| Adam / AdamW respectively | 92.42% | 97.83% |
| SGD | 69.26% | 94.79% |
| SGD with momentum | 92.94% | 97.58% |

Model A uses Adam/SGD/momentum SGD, batch size 64, plain ToTensor inputs, and no scheduler. Model B uses AdamW/SGD/momentum SGD, batch size 128, augmentation, normalization, and warmup/cosine scheduling. Equal epoch counts do not make these recipes identical. Model A checkpoints were selected by minimum validation loss, as were Model B checkpoints; test was evaluated after each fixed training recipe completed. Prior test scores had already been observed before choosing this new epoch budget, so this is adaptive test reuse, not a fresh untouched final evaluation.

- Model A: `results/model_a/modelA_40ep_20261019_095058/`.
- Model B: `results/model_b/modelB_40ep_{adamw,sgd,sgd_momentum}_20261019_095058/`.
- Run inventory: [`results/experiment_40epochs_20261019_095058.json`](results/experiment_40epochs_20261019_095058.json).
- Local logs: `results/run_logs/20261019_095058_40epochs.log` (ignored).
- Checkpoints are local and excluded from Git; preserve them separately for submission.

## Current completed results and remaining work

Use the [current four-model comparison](results/comparisons/current_new_model_a/comparison.md). It includes the latest GitHub Model A, final Model B v2, MobileNetV2, and EfficientNet-B0.

| Model | Reported optimizer | Epochs | Test accuracy | Parameters |
| --- | --- | ---: | ---: | ---: |
| Current Model A | SGD with momentum | 20 | 87.88% | 195,290 |
| Model B v2 | AdamW | 60 | 98.12% | 96,538 |
| MobileNetV2 | Two-stage Adam | 20 | 97.65% | 2,236,682 |
| EfficientNet-B0 | Adam | 10 | 97.85% | 4,020,358 |

Current Model A also scored 85.33% with Adam and 46.91% with plain SGD. Historical 25,994-parameter Model A runs (including the local 60-epoch AdamW experiment) are not results for this architecture and must not be mixed with it. The local AdamW training-code changes have been replaced by the latest GitHub Model A implementation.

The current Model A's three matrix CSV/PNG pairs are saved beside `results/model_a/modelA_cuda_run/{adam,sgd,sgd_momentum}/test_metrics.json`. Their class order and all 4,050 test counts are validated when rendered. Older Model A matrices describe only their own historical architectures.

Saved confusion matrices and epoch histories support the reported metrics. Different training recipes, epoch budgets and hardware mean this is not an isolated convolution-type or inference-speed comparison. Single-seed test scores do not establish statistical superiority. See the comparison for provenance limitations.

EfficientNet-B0 is implemented in `src/en3150_a03/models/efficientnet_b0.py`; run `scripts/train_efficientnet_b0.py` in the project environment. Its existing `metrics.json`, matrix CSV/PNG, and `training_curves.png` are under `results/efficientnet_b0/adam/run_20261009_193426/`. That history records training/validation loss and validation accuracy, not training accuracy.

Interrupted experiment files and loose logs have been moved to ignored `local_archive/`; completed historical results and checkpoints are preserved. Do not cite the interrupted attempt as a completed run.

Remaining work:
1. Review the compiled draft `report/report.pdf` and its source `report/report.tex`; fill in names, index numbers and group number, confirm assignment formatting, and review citations/interpretation before submission.
2. Preserve the six new Model A/B checkpoints and EfficientNet-B0 checkpoint separately; obtain MobileNetV2 weights from its owner. Historical runs still require their original weights if independent reproduction is needed. Weights are excluded from Git.
3. Review and push the local integration branch; these local commits are not yet on GitHub.

## Collaboration and reproducibility

Use feature branches and pull requests; commit work regularly. All runs must use the same saved split files and metric definitions. Do not tune on the held-out test set. Track small metrics and manifests, but not raw images or large checkpoints. Record software versions and hardware alongside measured times. The GitHub repository is private by default; invite collaborators and the examiner if they need direct access.
