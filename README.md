# EN3150 Assignment 03 — Resource-Constrained CNN

A shared project for comparing two custom CNNs (standard and depthwise-separable) with two fine-tuned lightweight pretrained networks on low-resolution imagery.

**Status:** repository scaffold only. The data loader, model implementations, training commands, tests, and experiment results are **not implemented yet**. Configuration values are starting points, not validated experimental choices.

## Assignment requirements

- Choose an image-classification dataset other than CIFAR-10. EuroSAT RGB is the proposed dataset; confirm this with the group.
- Images must be at most 64 × 64 pixels. Save a reproducible 70% train / 15% validation / 15% test split.
- Design Model A with standard convolutions and max-pooling; design Model B with depthwise-separable convolutions and no more than 100,000 trainable parameters. Document and calculate both parameter counts.
- Choose and justify a primary optimizer and learning rate; compare with SGD and SGD with momentum.
- Train both custom models for at least 20 epochs. Report loss curves, test accuracy, precision, recall, and confusion matrices.
- Fine-tune two lightweight pretrained models on the **same split** and compare accuracy, memory, and compute.
- Submit a PDF report and runnable code separately. See [`docs/task_checklist.pdf`](docs/task_checklist.pdf) for the full checklist.

## Proposed dataset

EuroSAT project: https://github.com/phelber/eurosat . Use its RGB image release if the group approves this dataset. Put extracted images under `data/raw/`; raw images are intentionally excluded from Git. Record the actual download source and preparation steps in [`data/README.md`](data/README.md). No dataset has been downloaded by this scaffold.

## Layout

- `configs/base.yaml`: common seed, paths, image size, and split proportions.
- `configs/models/`: separate configuration for each of the four models.
- `configs/optimizers/`: separate optimizer settings; learning rates remain undecided.
- `src/en3150_a03/models/`: one Python module per model. Shared data/training/evaluation code is in `src/en3150_a03/`.
- `scripts/`: planned CLI entry points for preparing data, training one model, evaluating checkpoints, and generating report assets.
- `data/splits/`: saved split manifests **to be generated**. Do not create different splits for different models.
- `results/<model>/<optimizer>/`: per-run metrics, history, and checkpoints (checkpoints are ignored by Git). Use distinct run IDs within these folders if repeating experiments.
- `tests/`: planned tests for split integrity, model shapes, the Model B parameter cap, and metric definitions.
- `report/`: final report source and generated figures/tables.
- `docs/`: task checklist and experiment notes.

## Workflow to implement

1. Confirm framework (this scaffold reserves a PyTorch-oriented layout), dataset, optimizer hyperparameters, and class ordering.
2. Implement `scripts/prepare_data.py`; write `data/splits/{train,val,test}.csv`. Validate ratios and absence of overlap.
3. Implement models independently in `src/en3150_a03/models/` and verify input/output shapes and parameter counts.
4. Implement the shared training/evaluation pipeline and separate model + optimizer configurations. Select checkpoints using validation data only.
5. Train each model independently. Save its configuration, checkpoint, epoch times, training/validation history, and test metrics in its own results folder.
6. Generate tables and plots from saved metrics, then write and verify the report.

**No training command works yet.** Script entry points intentionally stop with a clear “not implemented” message until the group adds code. The test files are likewise placeholders, not passing tests.

## Collaboration and reproducibility

Use feature branches and pull requests; commit work regularly. All runs must use the same saved split files and metric definitions. Do not tune on the held-out test set. Track small metrics and manifests, but not raw images or large checkpoints. Record software versions and hardware alongside measured times. The GitHub repository is private by default; invite collaborators and the examiner if they need direct access.
