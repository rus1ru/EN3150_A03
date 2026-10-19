# Model B v2 experiment

Architecture: custom inverted-residual depthwise CNN; channel attention=True.
Parameters: **96,538**; FP32 weights: **377.10 KiB**; Conv/Linear MACs: **21,358,976** at 64x64.

Optimizer: sgd_momentum; peak learning rate: 0.05; completed epochs: 40/40; seed: 42.
Selected epoch: 36; validation accuracy: **97.38%**; validation cross-entropy: 0.11770.

Checkpoint selection uses minimum validation loss. Normalization uses training images only. Training augmentation enabled: True. When enabled, it uses flips, quarter-turns, reflected crops and mild color jitter. Label smoothing applies to training; validation/test metrics use ordinary targets.

## Saved final result

| Model/recipe | Parameters | FP32 weights (KiB) | MACs | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Model B v2 / sgd_momentum | 96,538 | 377.10 | 21,358,976 | 97.58% |

Macro precision: 0.9758; macro recall: 0.9748.

## Interpretation

The final architecture uses expanded features, batch normalization, linear projections and residual connections. SE gates model channel importance; global average pooling produces the classifier input. An accuracy comparison against Model A changes both architecture and recipe, so a gain cannot be attributed to attention alone. Use the no-attention ablation with the same training recipe to test that component.

Batch-one Conv2d and Linear only; one multiply-accumulate = one MAC (approximately two FLOPs). Bias adds, activations, pooling, memory traffic and framework overhead excluded. This is operation count, not latency. Also excludes BN, residual additions and SE elementwise gates. Includes SE Conv2d MACs.

This model targets accuracy under the 100k limit and uses more memory/compute than Model A. FP32 weights exclude activations and optimizer state. Timing across different computers does not establish speedup. The frozen split is image-level, not geographically independent. One seed does not establish statistical superiority.

## Sources

- MobileNets: https://arxiv.org/abs/1704.04861
- MobileNetV2: https://arxiv.org/abs/1801.04381
- Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507
- MobileNetV3 hard gating: https://arxiv.org/abs/1905.02244
- Network In Network/global average pooling: https://arxiv.org/abs/1312.4400
- Batch Normalization: https://arxiv.org/abs/1502.03167
- AdamW: https://arxiv.org/abs/1711.05101

## Run artifacts

- [loss_curves.png](../loss_curves.png)
- [confusion_matrix.png](../confusion_matrix.png)
- [per_class_metrics.csv](../per_class_metrics.csv)
- [test_metrics.json](../test_metrics.json)
- [training_history.csv](../training_history.csv)
- [experiment_config.json](../experiment_config.json)
- [architecture_profile.json](../architecture_profile.json)
