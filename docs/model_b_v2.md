# Model B v2: accuracy-focused depthwise CNN

Contribution: KavijaK. This is an additional Model B experiment; the original model, trained results, shared split and teammates' files are unchanged.

## Run in Colab

1. Upload `notebooks/train_model_b_v2_colab.ipynb` to Colab and choose a T4 GPU runtime.
2. Run setup and upload `EN3150_A03_ModelB_v2_Code.zip` when prompted. Mount Drive when asked.
3. Run the default `modelB_v2_adamw_01` experiment for **60 epochs**. Training checkpoints are saved every epoch to `MyDrive/EN3150_A03/model_b/<RUN_ID>/`.
4. Review validation. When the recipe is fixed, set `EVALUATE_FINAL_TEST = True` and run the final-test cell.
5. Run the report/export cell. Extract the downloaded integration ZIP into the repository, review and commit only the new run directory. Keep checkpoint files on Drive.

For interruptions: use the same ZIP and settings, set `RESUME = True`, and rerun setup/training. Resume preserves completed epochs and checks the source, data, split and runtime versions. Start a fresh runtime and new run ID when changing code. For out-of-memory errors, start a new run with batch size 64. The archive excludes the raw dataset and trained checkpoints; setup downloads and verifies the official data.

**Status:** architecture and training/evaluation/export paths tested locally, including a synthetic full run. No EuroSAT GPU training was performed for v2 here; no improved accuracy is claimed yet. The delivered notebook contains no outputs.

## Why change the architecture?

The original Model B has 5,589 parameters. Its train and validation curves are close, Adam is still improving at epoch 20, and plain SGD barely reduces cross-entropy. This suggests insufficient capacity and/or optimization, rather than a large generalization gap. It does not identify one cause conclusively.

V2 uses a pointwise RGB expansion before the first depthwise spatial filter, seven expanded bottleneck blocks, batch normalization, four residual connections and two SE channel gates. Linear projections preserve information before residual addition. Global average pooling retains a small classification head; GAP was already present in v1.

| Stage | Output shape | Design |
| --- | --- | --- |
| Stem | 24 × 32 × 32 | RGB pointwise lift; depthwise 3×3, stride 2 |
| Blocks 1–3 | 32 × 32 × 32 | Widths 24, 32, 32; expansion ratio 2 |
| Blocks 4–5 | 48 × 16 × 16 | Strided depthwise block; residual block with SE |
| Blocks 6–7 | 80 × 8 × 8 | Strided depthwise block; residual block with SE |
| Head | 192 × 8 × 8 → 10 | Pointwise, GAP, dropout 0.1, linear logits |

All 3×3 spatial convolutions are depthwise. Hidden activations use ReLU6; channel gates use hard sigmoid. There are no pretrained weights. The custom configuration combines published techniques; it is not a claim of a new convolution or attention method.

| Architecture | Trainable parameters | FP32 weights (KiB) | Conv/Linear MACs |
| --- | ---: | ---: | ---: |
| Original Model B | 5,589 | 21.83 | 1,579,328 |
| Model A | 25,994 | 101.54 | 11,209,024 |
| Model B v2 | **96,538** | **377.10** | **21,358,976** |
| V2 without attention | 87,546 | 341.98 | 21,350,272 |

Batch size one, RGB 64×64, one MAC is approximately two FLOPs. Counts include all Conv2d and Linear modules, including attention's 1×1 convolutions; they exclude normalization, pooling, nonlinearities, gate multiplication, residual additions and memory traffic. FP32 estimates exclude buffers, activations and optimizer memory. V2 intentionally spends more memory/compute than v1 and Model A while remaining under the assignment's 100,000-parameter cap. Operation counts are not runtime measurements.

## Training recipe and evaluation

- Default AdamW: peak learning rate 0.003, weight decay 0.0001, excluding biases and batch-normalization parameters from decay.
- Five-epoch linear warmup, cosine decay to 0.00001; 60 epochs, batch size 128, seed 42.
- Per-channel normalization measured from **train.csv only**.
- Training-only horizontal/vertical flips, quarter-turn rotations, reflected crops and mild color jitter.
- Cross-entropy with label smoothing 0.05 for training; ordinary cross-entropy for validation/test. Dropout 0.1, gradient clipping at 5, CUDA mixed precision.
- Best checkpoint selected by minimum validation loss. Training never evaluates the test loader. Final test is an explicit step, and saved test results are reused.
- Source hashes, dataset fingerprint, manifest hashes, settings and runtime recorded with the run. Resume includes optimizer/scaler states and epoch seeds. CPU recovery is tested; numerical identity across different GPU platforms is not promised.

The immutable 70/15/15 split is reused. The image-level split does not establish geographical independence. V2 changes both architecture and recipe, so a better score cannot be attributed to attention alone or presented as the original controlled A/B comparison. Keep v1 results for that comparison.

## Additional experiments

Run the default AdamW experiment first. For the assignment's Adam, SGD and momentum-SGD comparison, select each optimizer in a new run while keeping the architecture, split, seed, augmentation and epoch budget fixed. Defaults are Adam 0.003, SGD 0.05 and momentum-SGD 0.05 with momentum 0.9; disclose these different learning rates. AdamW is an additional optimizer, not a substitute for the required Adam result.

Example from the repository root:

```sh
python scripts/train_model_b_v2.py --run-id modelB_v2_adam_01 --optimizer adam --device cuda
python scripts/train_model_b_v2.py --run-id modelB_v2_sgd_01 --optimizer sgd --device cuda
python scripts/train_model_b_v2.py --run-id modelB_v2_sgdm_01 --optimizer sgd_momentum --device cuda
python scripts/train_model_b_v2.py --run-id modelB_v2_no_se_01 --no-attention --device cuda
```

The no-attention run keeps widths, depth and training unchanged but has fewer parameters; it tests the attention module's contribution, not a parameter-matched substitute. Use validation to choose configurations. If making a stronger comparative claim, repeat the fixed selected recipe with seeds 42, 43 and 44 and report mean/spread; do not select a seed by test accuracy.

Final evaluation and report generation:

```sh
python scripts/train_model_b_v2.py --run-id modelB_v2_adamw_01 --evaluate-only --device cuda
python scripts/report_model_b_v2.py --run-dir results/model_b/modelB_v2_adamw_01 --output-dir results/model_b/modelB_v2_adamw_01/report
```

Report output must be a new directory. Colab supplies Drive paths automatically. Direct commands default to the repository's `results/model_b/` directory. The YAML configuration is a descriptive specification; CLI arguments control execution.

## Paper foundations

- Howard et al., [MobileNets](https://arxiv.org/abs/1704.04861): depthwise separable convolutions.
- Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381): expanded inverted residuals and linear bottlenecks.
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507): channel recalibration.
- Howard et al., [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244): efficient hard gating.
- Lin et al., [Network In Network](https://arxiv.org/abs/1312.4400): global average pooling classification head.
- Ioffe and Szegedy, [Batch Normalization](https://arxiv.org/abs/1502.03167): normalized intermediate activations.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW.

These papers support the design choices, not an unmeasured accuracy claim for this implementation.
