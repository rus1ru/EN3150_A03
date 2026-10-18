# Model B integration comparison

Metrics are read from saved teammate runs. No model is selected by test performance.

| Model | Optimizer | Accuracy | Parameters | FP32 weights (KiB) | Conv/Linear MACs | Mean epoch (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ModelA | adam | 0.7580 | 25,994 | 101.54 | 11,209,024 | 93.60 |
| ModelA | sgd | 0.3837 | 25,994 | 101.54 | 11,209,024 | 58.32 |
| ModelA | sgd_momentum | 0.8168 | 25,994 | 101.54 | 11,209,024 | 31.11 |
| ModelB | adam | 0.5914 | 5,589 | 21.83 | 1,579,328 | 11.61 |
| ModelB | sgd | 0.1899 | 5,589 | 21.83 | 1,579,328 | 11.58 |
| ModelB | sgd_momentum | 0.7420 | 5,589 | 21.83 | 1,579,328 | 11.40 |
| MobileNetV2 | adam_two_stage | 0.9765 | 2,236,682 | 8737.04 | 24,461,312 | unavailable |

Batch-one Conv2d and Linear only; one multiply-accumulate = one MAC (approximately two FLOPs). Bias adds, activations, pooling, memory traffic and framework overhead excluded. This is operation count, not latency.

Epoch times come from different devices and runs: they cannot establish architecture speedups. Model A/B use ToTensor without augmentation; MobileNetV2 uses pretrained weights, flips and ImageNet normalization with a different batch size. Accuracy therefore compares complete training recipes, while A/B provides the controlled convolution comparison. FP32 weights exclude runtime activations and optimizer memory.

EfficientNet-B0 is still a placeholder in this checkout and has no results to include. The group must complete its second pretrained network before claiming the final comparison is complete.

Model A's existing results identify the dataset fingerprint and class order but do not save split manifest hashes. Its exact image split provenance cannot be independently proven from that run metadata.
