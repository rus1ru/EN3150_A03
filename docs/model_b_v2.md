# Final Model B: accuracy-focused depthwise CNN

Contribution: KavijaK. This is the final Model B implementation. The `v2` filenames and training source are retained to match the completed run's checkpoint/source hashes. Shared data and teammates' code/results remain unchanged. Superseded Model B workflows/results are available in Git history.

## Run in Colab

1. Upload `notebooks/train_model_b_v2_colab.ipynb` to Colab and choose a T4 GPU runtime.
2. Run setup and upload `EN3150_A03_ModelB_v2_Code.zip` when prompted. Mount Drive when asked.
3. Configure a new run from the remaining-optimizer instructions below, then train for **60 epochs**. The default AdamW recipe is already complete. Training checkpoints are saved every epoch to `MyDrive/EN3150_A03/model_b/<RUN_ID>/`.
4. Review validation. When the recipe is fixed, set `EVALUATE_FINAL_TEST = True` and run the final-test cell.
5. Run the report/export cell. Extract the downloaded integration ZIP into the repository, review and commit only the new run directory. Keep checkpoint files on Drive.

For interruptions: use the same ZIP and settings, set `RESUME = True`, and rerun setup/training. Resume preserves completed epochs and checks the source, data, split and runtime versions. Start a fresh runtime and new run ID when changing code. For out-of-memory errors, start a new run with batch size 64. The archive excludes the raw dataset and trained checkpoints; setup downloads and verifies the official data.

## Verified result

The user completed 60 epochs with AdamW on a Colab Tesla T4. The selected checkpoint is epoch 59 (validation accuracy **97.58%**, validation loss **0.10745**). Final test accuracy is **98.1235%** (3,974/4,050 images), macro precision **98.1205%**, and macro recall **97.9911%**. Mean training-plus-validation time was **25.11 seconds per epoch** on that run.

The original exported metrics, history, curves, confusion matrix, configuration and architecture profile are saved under [`results/model_b/modelB_v2_adamw_01`](../results/model_b/modelB_v2_adamw_01/). Source hashes and split hashes match; accuracy/precision/recall were independently checked against the saved confusion matrix. The checkpoint itself remains in Google Drive, as intended by the export workflow.

- [Final experiment report](../results/model_b/modelB_v2_adamw_01/report/experiment_report.md)
- [Read-only comparison with teammates' results](../results/model_b/final_comparison/comparison.md)

The primary AdamW experiment is complete. **Required remaining Model B work: train and evaluate plain SGD and momentum SGD on this same final architecture.** Do not rerun the completed AdamW experiment just to reproduce the already saved scores.

## Architecture

The model uses a pointwise RGB expansion before the first depthwise spatial filter, seven expanded bottleneck blocks, batch normalization, four residual connections and two SE channel gates. Linear projections preserve information before residual addition. Global average pooling keeps the classification head small.

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
| Model A | 25,994 | 101.54 | 11,209,024 |
| Model B v2 | **96,538** | **377.10** | **21,358,976** |
| V2 without attention | 87,546 | 341.98 | 21,350,272 |

Batch size one, RGB 64×64, one MAC is approximately two FLOPs. Counts include all Conv2d and Linear modules, including attention's 1×1 convolutions; they exclude normalization, pooling, nonlinearities, gate multiplication, residual additions and memory traffic. FP32 estimates exclude buffers, activations and optimizer memory. The final Model B uses more memory/compute than Model A while remaining under the assignment's 100,000-parameter cap. Operation counts are not runtime measurements.

## Training recipe and evaluation

- Default AdamW: peak learning rate 0.003, weight decay 0.0001, excluding biases and batch-normalization parameters from decay.
- Five-epoch linear warmup, cosine decay to 0.00001; 60 epochs, batch size 128, seed 42.
- Per-channel normalization measured from **train.csv only**.
- Training-only horizontal/vertical flips, quarter-turn rotations, reflected crops and mild color jitter.
- Cross-entropy with label smoothing 0.05 for training; ordinary cross-entropy for validation/test. Dropout 0.1, gradient clipping at 5, CUDA mixed precision.
- Best checkpoint selected by minimum validation loss. Training never evaluates the test loader. Final test is an explicit step, and saved test results are reused.
- Source hashes, dataset fingerprint, manifest hashes, settings and runtime recorded with the run. Resume includes optimizer/scaler states and epoch seeds. CPU recovery is tested; numerical identity across different GPU platforms is not promised.

The immutable 70/15/15 split is reused. The image-level split does not establish geographical independence. Model A and final Model B differ in architecture and training recipe. Their accuracy difference cannot be attributed to attention or depthwise convolution alone.

## Remaining optimizer comparison

The assignment checklist requires a chosen primary optimizer compared with plain SGD and SGD with momentum. AdamW is the documented primary optimizer. Run SGD and momentum SGD with new run IDs, keeping the final architecture, split, seed, batch size, augmentation and 60-epoch budget fixed. Both default to peak learning rate 0.05; momentum SGD uses momentum 0.9. Disclose these settings versus AdamW's 0.003; these are documented recipes, not equally tuned optimizer optima. A separate Adam run is optional.

In the Colab settings cell, use `RUN_ID = "modelB_v2_sgd_01"` and `OPTIMIZER = "sgd"`, then repeat with `RUN_ID = "modelB_v2_sgdm_01"` and `OPTIMIZER = "sgd_momentum"`. Keep `RESUME = False` for each new run. After each completes, run the final-test and export cells. These produce the two remaining result ZIPs. Use a fresh runtime and the final code archive if uploading the cleaned code; do not mix source versions in an existing runtime.

Example from the repository root:

```sh
python scripts/train_model_b_v2.py --run-id modelB_v2_sgd_01 --optimizer sgd --device cuda
python scripts/train_model_b_v2.py --run-id modelB_v2_sgdm_01 --optimizer sgd_momentum --device cuda
python scripts/train_model_b_v2.py --run-id modelB_v2_no_se_01 --no-attention --device cuda
```

The no-attention run is optional research, not an outstanding assignment requirement. It keeps widths, depth and training unchanged but has fewer parameters; it tests the attention module's contribution, not a parameter-matched substitute. Use validation to choose configurations. If making a stronger comparative claim, repeat the fixed selected recipe with seeds 42, 43 and 44 and report mean/spread; do not select a seed by test accuracy.

Final evaluation and report generation:

```sh
python scripts/train_model_b_v2.py --run-id modelB_v2_adamw_01 --evaluate-only --device cuda
python scripts/report_model_b_v2.py --run-dir results/model_b/modelB_v2_adamw_01 --output-dir results/model_b/modelB_v2_adamw_01/report
```

The report command refreshes only its derived Markdown report. After importing the SGD and momentum-SGD ZIPs, regenerate the comparison into a new directory with all three `--model-b-run` paths:

```sh
python scripts/compare_model_b.py --model-b-run results/model_b/modelB_v2_adamw_01 results/model_b/modelB_v2_sgd_01 results/model_b/modelB_v2_sgdm_01 --output-dir results/model_b/complete_optimizer_comparison
```

Colab supplies Drive paths automatically. Direct commands default to the repository's `results/model_b/` directory. The YAML configuration is a descriptive specification; CLI arguments control execution.

## Paper foundations

- Howard et al., [MobileNets](https://arxiv.org/abs/1704.04861): depthwise separable convolutions.
- Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381): expanded inverted residuals and linear bottlenecks.
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507): channel recalibration.
- Howard et al., [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244): efficient hard gating.
- Lin et al., [Network In Network](https://arxiv.org/abs/1312.4400): global average pooling classification head.
- Ioffe and Szegedy, [Batch Normalization](https://arxiv.org/abs/1502.03167): normalized intermediate activations.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW.

The papers support the design choices. The reported accuracy is from the saved local experiment artifacts, not the papers.
