# Final Model B integration comparison

Read from saved runs. Checkpoints were selected using validation loss; this table does not select an optimizer by test performance.

| Model | Optimizer | Test accuracy | Macro precision | Macro recall | Parameters | FP32 weights (KiB) | Conv/Linear MACs | Mean epoch (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModelA | adam | 0.9242 | 0.9241 | 0.9204 | 195,290 | 762.85 | 42,785,664 | 5.98 |
| ModelA | sgd | 0.6926 | 0.6895 | 0.6828 | 195,290 | 762.85 | 42,785,664 | 6.48 |
| ModelA | sgd_momentum | 0.9294 | 0.9267 | 0.9276 | 195,290 | 762.85 | 42,785,664 | 6.12 |
| ModelB | adamw | 0.9783 | 0.9777 | 0.9778 | 96,538 | 377.10 | 21,358,976 | 14.00 |
| ModelB | sgd | 0.9479 | 0.9472 | 0.9468 | 96,538 | 377.10 | 21,358,976 | 13.99 |
| ModelB | sgd_momentum | 0.9758 | 0.9758 | 0.9748 | 96,538 | 377.10 | 21,358,976 | 14.23 |
| MobileNetV2 | adam_two_stage | 0.9765 | 0.9764 | 0.9746 | 2,236,682 | 8737.04 | 24,461,312 | unavailable |
| EfficientNet-B0 | adam | 0.9785 | 0.9786 | 0.9776 | 4,020,358 | 15704.52 | 31,979,392 | 16.79 |

**Final Model B optimizer comparison is complete:** the primary AdamW run and both SGD baselines are included.

Operation counts cover batch-one Conv2d and Linear layers at 64×64, including SE convolutions. One MAC is approximately two FLOPs. Bias adds, activations, pooling, batch normalization, residual additions, SE gate multiplication, memory traffic and framework overhead are excluded. These counts are not latency.

Model A uses unaugmented ToTensor inputs. Final Model B uses train-only normalization, augmentation, a primary AdamW optimizer, a longer schedule and a larger custom architecture; MobileNetV2 and EfficientNet-B0 use pretrained weights and ImageNet preprocessing. Accuracy compares complete recipes, not an isolated convolution change. FP32 weights exclude runtime activations and optimizer state. Epoch times from different devices/runs cannot establish architecture speedups.

Model A's saved metadata identifies the dataset and class order but lacks split manifest hashes, so that run's exact image split cannot be independently proven. Model B and MobileNetV2 manifest hashes match the canonical split.

EfficientNet-B0 records class order and split path, but not manifest hashes or dataset fingerprint; its exact split provenance cannot be independently established from saved run metadata. The final group PDF remains unfinished. One seed on an image-level split does not establish statistical superiority or geographic generalization.
