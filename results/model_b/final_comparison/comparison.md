# Final Model B integration comparison

Read from saved runs. Checkpoints were selected using validation loss; this table does not select an optimizer by test performance.

| Model | Optimizer | Test accuracy | Macro precision | Macro recall | Parameters | FP32 weights (KiB) | Conv/Linear MACs | Mean epoch (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModelA | adam | 0.7580 | 0.7532 | 0.7491 | 25,994 | 101.54 | 11,209,024 | 93.60 |
| ModelA | sgd | 0.3837 | 0.4214 | 0.3886 | 25,994 | 101.54 | 11,209,024 | 58.32 |
| ModelA | sgd_momentum | 0.8168 | 0.8106 | 0.8089 | 25,994 | 101.54 | 11,209,024 | 31.11 |
| ModelB | adamw | 0.9812 | 0.9812 | 0.9799 | 96,538 | 377.10 | 21,358,976 | 25.11 |
| ModelB | sgd | 0.9600 | 0.9598 | 0.9583 | 96,538 | 377.10 | 21,358,976 | 24.73 |
| ModelB | sgd_momentum | 0.9785 | 0.9780 | 0.9777 | 96,538 | 377.10 | 21,358,976 | 23.89 |
| MobileNetV2 | adam_two_stage | 0.9765 | 0.9764 | 0.9746 | 2,236,682 | 8737.04 | 24,461,312 | unavailable |

**Final Model B optimizer comparison is complete:** the primary AdamW run and both SGD baselines are included.

Operation counts cover batch-one Conv2d and Linear layers at 64×64, including SE convolutions. One MAC is approximately two FLOPs. Bias adds, activations, pooling, batch normalization, residual additions, SE gate multiplication, memory traffic and framework overhead are excluded. These counts are not latency.

Model A uses unaugmented ToTensor inputs. Final Model B uses train-only normalization, augmentation, a primary AdamW optimizer, a longer schedule and a larger custom architecture; MobileNetV2 uses pretrained weights and ImageNet preprocessing. Accuracy compares complete recipes, not an isolated convolution change. FP32 weights exclude runtime activations and optimizer state. Epoch times from different devices/runs cannot establish architecture speedups.

Model A's saved metadata identifies the dataset and class order but lacks split manifest hashes, so that run's exact image split cannot be independently proven. Model B and MobileNetV2 manifest hashes match the canonical split.

EfficientNet-B0 has no saved test metrics in this checkout. The second pretrained network and final group report remain group-level work. One seed on an image-level split does not establish statistical superiority or geographic generalization.
