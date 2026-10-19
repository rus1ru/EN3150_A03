# Controlled A/B study

Single seed; matched recipes. No learning-rate search. Prior test reuse remains a limitation.

| Model | Optimizer | Test accuracy | Macro precision | Macro recall |
|---|---|---:|---:|---:|
| model_a | adam | 95.4815% | 95.3303% | 95.3889% |
| model_a | sgd | 83.5062% | 82.9765% | 82.8422% |
| model_a | sgd_momentum | 95.9506% | 95.9267% | 95.8911% |
| model_b | adam | 96.6914% | 96.7018% | 96.5467% |
| model_b | sgd | 93.0617% | 92.8644% | 92.8822% |
| model_b | sgd_momentum | 97.1358% | 97.0710% | 97.0356% |
