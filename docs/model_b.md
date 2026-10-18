# Model B contribution: depthwise-separable CNN

Model B implementation and training results by [KavijaK](https://github.com/KavijaK). Model B reads the shared data loader and Model A's optimizer/evaluation helpers without editing them. Its original pull request left teammate implementations, results, README, and canonical split manifests unchanged; later integration updated project documentation and the combined comparison.

## What is implemented and what still needs to run

The implementation includes the network, checked parameter/MAC accounting, Adam/SGD/momentum comparison runner, epoch-boundary resume checkpoints, loss/accuracy curves, test confusion matrices, per-class precision/recall, machine-readable metrics, and generated experiment reports. `notebooks/train_model_b_colab.ipynb` runs this code on Colab GPU. Model B was trained for 20 epochs per optimizer on a Tesla T4 using the frozen EuroSAT RGB split. The [saved experiment report](../results/model_b/modelB_gpu_01/experiment_report.md) records test accuracy of 59.14% (Adam), 18.99% (plain SGD), and 74.20% (SGD with momentum). The declared primary optimizer is Adam; the momentum result is an observed comparison, not grounds to retune on the test set. Tests on tiny synthetic data validate execution but are not assignment scores.

Model A, MobileNetV2, and EfficientNet-B0 also have saved results. The [four-model comparison](../results/comparisons/all_models_20261018/comparison.md) combines their reported metrics with Model B without retraining them. The final group PDF report and its required submission details remain unfinished.

## Architecture: isolate the cost of convolution factorization

For the primary A/B comparison, retain Model A's exact channel widths, pooling locations, ReLU placement and classifier. Replace each standard convolution with a depthwise 3x3 convolution followed by a pointwise 1x1 convolution. There is no nonlinearity between the depthwise and pointwise operators; ReLU follows pointwise mixing. The depthwise bias is omitted because its effect can be absorbed by the subsequent pointwise bias.

This is a custom matched experiment using established operations, not a claim to have invented depthwise convolution or a reproduction of MobileNet. Its contribution is a reproducible, tightly controlled EuroSAT accuracy/memory/compute comparison. Smaller weight count alone does not establish faster hardware execution or better accuracy.

All convolutions have stride 1; depthwise convolutions have padding 1 and `groups=C_in`; pointwise convolutions have padding 0 and groups 1. Each max-pool is 2x2 with stride 2. Shapes below exclude batch dimension. The detailed executed layer list is saved in `architecture_profiles.json` on every experiment run.

| Operation | Input -> output | Kernel / groups | Trainable parameters | MACs per image |
| --- | --- | --- | ---: | ---: |
| Depthwise 1 | 3x64x64 -> 3x64x64 | 3x3 / 3 | 3x9 = 27 | 110,592 |
| Pointwise 1 + bias | 3x64x64 -> 16x64x64 | 1x1 / 1 | 3x16+16 = 64 | 196,608 |
| ReLU, max pool | 16x64x64 -> 16x32x32 | 2x2 pool | 0 | excluded |
| Depthwise 2 | 16x32x32 -> 16x32x32 | 3x3 / 16 | 16x9 = 144 | 147,456 |
| Pointwise 2 + bias | 16x32x32 -> 32x32x32 | 1x1 / 1 | 16x32+32 = 544 | 524,288 |
| ReLU, max pool | 32x32x32 -> 32x16x16 | 2x2 pool | 0 | excluded |
| Depthwise 3 | 32x16x16 -> 32x16x16 | 3x3 / 32 | 32x9 = 288 | 73,728 |
| Pointwise 3 + bias | 32x16x16 -> 64x16x16 | 1x1 / 1 | 32x64+64 = 2,112 | 524,288 |
| ReLU, max pool | 64x16x16 -> 64x8x8 | 2x2 pool | 0 | excluded |
| Global average pool, flatten | 64x8x8 -> 64 | 8x8 effective | 0 | excluded |
| Linear + bias, ReLU | 64 -> 32 | - | 64x32+32 = 2,080 | 2,048 |
| Linear + bias, logits | 32 -> 10 | - | 32x10+10 = 330 | 320 |
| **Total** | | | **5,589** | **1,579,328** |

For a depthwise/pointwise block, parameter count is `9*C_in + C_in*C_out + C_out`, and MAC count is `H*W*(9*C_in + C_in*C_out)`. Linear MAC count is `N_in*N_out`. One multiply-accumulate is one MAC, approximately two FLOPs. Bias additions, ReLU, pooling, memory traffic and launch overhead are excluded consistently for all profiled models.

Model A has 25,994 parameters and 11,209,024 Conv/Linear MACs at 64x64. Model B therefore reduces parameter storage by **78.50%** and counted MACs by **85.91%**. Its FP32 parameter estimate is `5,589*4/1024 = 21.832 KiB` (22.356 decimal kB), far below the 100,000-parameter cap. This excludes activations, optimizer state and serialization overhead. Reports also save actual weights-only state_dict bytes and best-checkpoint size separately.

ReLU computes a max-with-zero and avoids exponential evaluation. Global pooling avoids a 4,096-value flattened head. Depthwise factorization restricts spatial expressiveness: with one spatial filter per input channel, it cannot represent every full-convolution kernel. A reduction in capacity can affect accuracy. Peak runtime memory and deployment latency are not measured by parameter/MAC accounting.

## Training contract

- Use the committed `data/splits_torch_v1/` files: 18,900 train, 4,050 validation, 4,050 test images. Never create a new split for Model B.
- Native RGB 64x64, `ToTensor()` to [0,1], no normalization or augmentation, matching Model A's custom-model recipe.
- Seed 42, batch size 64, cross-entropy, fixed rates, no weight decay, scheduler or early stopping. Each run lasts at least 20 epochs.
- Adam lr=0.001 is the predeclared primary choice; adaptive updates provide a practical baseline for this small CNN. Plain SGD lr=0.01 and SGD with momentum 0.9 at lr=0.01 are comparison settings inherited from Model A.
- Every optimizer starts from the same model initialization and shuffle seed. GPU cuDNN benchmarking is disabled and deterministic cuDNN mode enabled; bitwise equivalence across PyTorch versions/devices is not promised.
- Select the earliest checkpoint with minimum validation cross-entropy. Finish training all requested optimizers before evaluating their best checkpoints on test data. Do not select architectures, hyperparameters or a winning optimizer using test accuracy.
- Use Model A's evaluator unchanged for accuracy, macro precision/recall and per-class precision/recall. Confusion rows are true classes and columns predicted classes. Zero-denominator precision/recall are zero, matching the team implementation.
- A 95% Wilson accuracy interval is supplementary: it assumes independent images and does not measure variation across seeds or account for nearby correlated satellite tiles.

Momentum maintains a decaying accumulation of gradients, which can accelerate persistent descent directions and smooth noisy updates. It can also overshoot. Compare saved validation loss/accuracy histories and convergence with plain SGD rather than assuming a gain. These fixed settings make a consistent comparison with Model A; they are not equally tuned optimizer optima.

## Run on Colab GPU

Open `notebooks/train_model_b_colab.ipynb` through Colab's **Upload notebook** option, choose a GPU runtime, then run cells in order. The delivered `EN3150_A03_ModelB_Code.zip` is an archive of the local committed branch, with the team's code dependencies and frozen manifests included. Upload that ZIP when the notebook prompts. No GitHub credentials are needed. This is useful while the feature branch is only local.

The notebook mounts Drive for per-epoch checkpoints and small result files, downloads images to local Colab disk, verifies the frozen split without regenerating it, runs all three optimizers for 20 epochs, generates an A/B/MobileNet integration comparison from the models present in its uploaded code archive, and downloads a small results ZIP. The saved notebook-era comparison predates EfficientNet-B0 integration; use the separate command above for all four models. Checkpoints remain on Drive and are excluded from the integration ZIP.

After a disconnect, select the same GPU hardware and package versions, reopen the notebook, upload the same code ZIP, choose the same `RUN_ID`, and rerun setup/data cells. The training cell automatically passes `--resume` when its experiment config exists. A partially processed epoch is redone from its last completed boundary. Interrupted setup with incomplete extracted images should be repaired by deleting only the incomplete local `data/raw/EuroSAT_RGB` directory and rerunning data setup; the ZIP is checksummed before extraction.

Resume requires the same settings, source hashes, classes, dataset, manifests, device type, GPU model, PyTorch/torchvision versions, CPU thread count and platform. If hardware changes, start a separate run ID; mixed-device times should not be averaged. Drive writes use temporary files and rename to reduce partial-file risk, but Drive must remain mounted and available. This is single-process recovery, not multi-worker/distributed training recovery.

## Run from the repository

```bash
python -m pip install -e .
python scripts/prepare_model_b_data.py
python scripts/run_model_b_experiments.py --run-id modelB_gpu_01 --device cuda --epochs 20 --batch-size 64
# Continue an interrupted run with the same code/settings:
python scripts/run_model_b_experiments.py --run-id modelB_gpu_01 --device cuda --epochs 20 --batch-size 64 --resume
# One optimizer only (still >=20 epochs; saves a different, explicit run):
python scripts/train_model_b.py --run-id modelB_adam_01 --device cuda
# Read all four models' saved metrics and write a new, separate comparison:
python scripts/compare_model_b.py --model-b-run results/model_b/modelB_gpu_01 --efficientnet-run results/efficientnet_b0/adam/run_20261009_193426 --output-dir results/comparisons/new_run
```

Choose `--device cpu` for CPU execution. Paths may be overridden through CLI flags. Existing run IDs cannot be overwritten silently. The data setup helper verifies official archive MD5 and the frozen image/split fingerprint; it never calls `prepare_splits`, which would compare historical PyTorch-version metadata when reconstructing manifests.

Each run writes `experiment_config.json` (source hashes, data provenance, environment), `architecture_profiles.json`, `comparison.{json,csv}`, `experiment_report.md`, and an optimizer subdirectory for each requested optimizer. Optimizer folders contain `best_model.pt`, `last_model.pt`, `model_weights.pt`, per-epoch history, curves, training summary, test metrics, confusion CSV/PNG and per-class scores CSV. All model weight files are already covered by the repository's ignore rules. Exported paths in new reports are relative so they remain usable outside Colab.

The integration helper reads Model A, MobileNetV2, and (when supplied) EfficientNet-B0 results without changing them. To include the saved EfficientNet-B0 run, pass `--efficientnet-run results/efficientnet_b0/adam/run_20261009_193426` or use that default path from this checkout; choose a fresh `--output-dir` each time. Timings from different machines must not be presented as an architecture speedup. The pretrained models use ImageNet initialization/normalization and augmentation, so compare full training recipes when discussing accuracy. Model A's historical metadata records the dataset fingerprint but not manifest hashes. EfficientNet-B0 records class order and split path, but not the dataset fingerprint or manifest hashes; its exact data provenance cannot be independently established from saved run metadata alone.

## Verification

```bash
python -m pytest -q
```

Model B tests cover the exact parameter/MAC calculations, grouped-convolution semantics, gradients/output shapes, CLI guards, recovery equivalence, checkpoint/config mismatch rejection, tiny end-to-end experiments for all optimizers, frozen-manifest preservation, cross-PyTorch-version data verification, and read-only integration with actual teammate artifact formats. These checks validate the implementation, not EuroSAT accuracy or Colab GPU execution.

## Sources for the final group report

- Howard et al. (2017), [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861): depthwise-separable factorization and efficiency motivation. Model B is not the published MobileNet architecture.
- Lin, Chen and Yan (2014), [Network In Network](https://arxiv.org/abs/1312.4400): global average pooling.
- Helber et al. (2019), [EuroSAT project and paper references](https://github.com/phelber/EuroSAT), and [official RGB archive](https://zenodo.org/records/7711810): dataset provenance. Published benchmark accuracy is not a result of this implementation.
- [PyTorch Conv2d reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): grouped/depthwise convolution semantics.

The final group PDF report still needs interpretation across all four models, group names/index numbers, citations, and required submission formatting. Model B GPU and EfficientNet-B0 results are saved in this repository.

## Local validation record

On 2026-10-18, Model B was trained on EuroSAT RGB for 20 epochs per optimizer on a Tesla T4 with CUDA PyTorch 2.11.0+cu130; configuration, dataset/manifest hashes, histories, and test metrics are saved under `results/model_b/modelB_gpu_01/`. The 27 repository tests passed in the PR review environment; tiny synthetic-data tests and CPU resume checks validate implementation behavior, not EuroSAT performance. Saved results do not measure deployment latency or statistical variation across seeds.
