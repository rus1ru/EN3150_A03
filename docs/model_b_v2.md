# Final Model B: accuracy-focused depthwise CNN

Contribution: KavijaK. This is the final Model B implementation. The `v2` filenames and training source are retained to match the completed run's checkpoint/source hashes. Shared data and teammates' code/results remain unchanged. Superseded Model B workflows/results are available in Git history.

## Run in Colab

1. Upload `notebooks/train_model_b_v2_colab.ipynb` to Colab and choose a T4 GPU runtime.
2. Run setup and upload `EN3150_A03_ModelB_v2_Code.zip` when prompted. Mount Drive when asked.
3. All required Model B experiments are complete. To reproduce an experiment, choose its optimizer below and a new run ID, then train for **60 epochs**. Training checkpoints are saved every epoch to `MyDrive/EN3150_A03/model_b/<RUN_ID>/`.
4. Review validation. When the recipe is fixed, set `EVALUATE_FINAL_TEST = True` and run the final-test cell.
5. Run the report/export cell. Extract the downloaded integration ZIP into the repository, review and commit only the new run directory. Keep checkpoint files on Drive.

For interruptions: use the same ZIP and settings, set `RESUME = True`, and rerun setup/training. Resume preserves completed epochs and checks the source, data, split and runtime versions. Start a fresh runtime and new run ID when changing code. For out-of-memory errors, start a new run with batch size 64. The archive excludes the raw dataset and trained checkpoints; setup downloads and verifies the official data.

## Verified final results

All three optimizer experiments completed **60 epochs** on a Colab Tesla T4, using the same final architecture, 70/15/15 split, seed 42, batch size 128, augmentation, normalization and epoch budget. AdamW was the predeclared primary optimizer. Checkpoints were selected independently by minimum validation cross-entropy, then evaluated on the 4,050-image test split.

| Optimizer | Peak LR | Selected epoch | Validation accuracy | Test accuracy | Macro precision | Macro recall | Mean epoch (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW (primary) | 0.003 | 59 | 97.58% | 98.12% | 98.12% | 97.99% | 25.11 |
| SGD | 0.05 | 57 | 95.36% | 96.00% | 95.98% | 95.83% | 24.73 |
| SGD + momentum | 0.05 | 59 | 97.83% | 97.85% | 97.80% | 97.77% | 23.89 |

The original exported metrics, histories, curves, confusion matrices, settings and architecture profiles are preserved. Source hashes and split hashes match the repository. Accuracy, macro precision and macro recall were independently recomputed from each saved confusion matrix. The downloaded primary archive is **AdamW**, not ordinary Adam.

- [AdamW (primary): complete run and report](../results/model_b/modelB_v2_adamw_01/report/experiment_report.md)
- [SGD: complete run and report](../results/model_b/modelB_v2_sgd_01/report/experiment_report.md)
- [SGD + momentum: complete run and report](../results/model_b/modelB_v2_sgdm_01/report/experiment_report.md)
- [Final comparison with teammates' saved results](../results/model_b/final_comparison/comparison.md)

**Model B is complete from this contribution's side:** architecture, primary optimizer, SGD and momentum comparison, histories/curves, test metrics, per-class scores, confusion matrices, parameter/memory/compute accounting, reproducible code and reports are included. No additional mandatory Model B training is outstanding. Checkpoints/weights remain in Google Drive and are intentionally excluded from Git; retain them for submission or independent checkpoint evaluation.

## Optimizer interpretation

At the shared 0.05 peak learning rate, momentum SGD improved test accuracy from 96.00% to 97.85%: **1.85 percentage points**, or 75 additional correct predictions. It first reached 95% validation accuracy at epoch **17**, compared with **38** for plain SGD (AdamW: epoch **20**). This first-crossing statistic describes convergence on these curves; it is not a guarantee of sustained accuracy or a multi-seed estimate.

AdamW recorded 3,974/4,050 correct predictions, plain SGD 3,888/4,050, and momentum SGD 3,963/4,050. AdamW's test advantage over momentum SGD is only **0.27 percentage points** (11 images). Momentum SGD had slightly higher validation accuracy at its selected checkpoint (97.83% versus 97.58%), while their selected validation losses were very close (0.10759 versus 0.10745). Keep AdamW as the predeclared primary recipe; do not claim statistical superiority from this single seed or select a winner by test accuracy.

All runs share the same architecture and data recipe; optimizer behavior differs, including AdamW's decoupled weight decay versus SGD's coupled decay and different peak learning rates. These are documented settings, not equally tuned optimizer optima. Epoch times are recorded on T4 but are not deployment/inference benchmarks.

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

## Reproduce the completed optimizer comparison

AdamW uses peak learning rate 0.003. Plain SGD and momentum SGD use 0.05; momentum is 0.9 for the latter. Keep the 60-epoch budget and other saved settings. Use **new run IDs** when repeating an experiment; the three committed result directories are completed records.

```sh
python scripts/train_model_b_v2.py --run-id modelB_adamw_repeat --optimizer adamw --device cuda
python scripts/train_model_b_v2.py --run-id modelB_sgd_repeat --optimizer sgd --device cuda
python scripts/train_model_b_v2.py --run-id modelB_sgdm_repeat --optimizer sgd_momentum --device cuda
```

After each repeated run, review validation before using `--evaluate-only` for final test evaluation. The notebook mounts Drive and exports the small integration files. In a fresh runtime, use the same source archive for checkpoint recovery; do not mix source versions mid-run. The YAML is a descriptive specification; CLI options control execution.

To refresh a run's derived report without changing its metrics:

```sh
python scripts/report_model_b_v2.py --run-dir results/model_b/modelB_v2_adamw_01 --output-dir results/model_b/modelB_v2_adamw_01/report
```

To rebuild the comparison in a new output directory:

```sh
python scripts/compare_model_b.py --model-b-run results/model_b/modelB_v2_adamw_01 results/model_b/modelB_v2_sgd_01 results/model_b/modelB_v2_sgdm_01 --output-dir results/model_b/comparison_rebuilt
```

Additional Adam, attention-ablation or multi-seed experiments are optional. An attention ablation is needed only to make a specific causal claim about that module, and repeat seeds are needed for stronger statistical claims. They are not missing from the required Model B optimizer comparison.

## Paper foundations

- Howard et al., [MobileNets](https://arxiv.org/abs/1704.04861): depthwise separable convolutions.
- Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381): expanded inverted residuals and linear bottlenecks.
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507): channel recalibration.
- Howard et al., [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244): efficient hard gating.
- Lin et al., [Network In Network](https://arxiv.org/abs/1312.4400): global average pooling classification head.
- Ioffe and Szegedy, [Batch Normalization](https://arxiv.org/abs/1502.03167): normalized intermediate activations.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW.

The papers support the design choices. The reported accuracy is from the saved local experiment artifacts, not the papers.
