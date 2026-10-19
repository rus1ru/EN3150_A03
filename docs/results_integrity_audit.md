# Saved EuroSAT experiment integrity audit

Scope: read-only audit of `results/model_a/modelA_cuda_run`, the three `results/model_b/modelB_v2_*_01` 60-epoch runs, `results/mobilenet_v2/run_01`, and `results/efficientnet_b0/adam/run_20261009_193426`. Unfinished/new 40-epoch runs were excluded. No training, source/config edits, installations, or git mutations were performed.

## Verified findings (no numerical integrity error found)

| Saved run | Epochs | Selected epoch | Test accuracy |
|---|---:|---:|---:|
| ModelA CUDA / SGD | 20 | 19 | 46.9136% |
| ModelA CUDA / SGD momentum | 20 | 20 | 87.8765% |
| ModelA CUDA / Adam | 20 | 18 | 85.3333% |
| ModelBV2 / SGD | 60 | 57 | 96.0000% |
| ModelBV2 / SGD momentum | 60 | 59 | 97.8519% |
| ModelBV2 / AdamW | 60 | 59 | 98.1235% |
| MobileNetV2 | 20 | 19 | 97.6543% |
| EfficientNet-B0 | 10 | 9 | 97.8519% |

- All eight histories have the declared number of epochs. Their selected epochs agree with the minimum validation loss, except EfficientNet-B0 correctly uses maximum validation accuracy (epoch 9, not minimum-loss epoch 10). ModelA/ModelBV2 summaries agree with the histories.
- All eight JSON confusion matrices exactly match their CSV matrices. Every matrix contains 4,050 samples with the expected class supports; independently recomputed accuracy, macro precision, and macro recall match the JSON values. PNG images were not independently rendered/visually verified.
- `results/comparisons/current_new_model_a/comparison.csv` carries these values and the correct ModelA CUDA provenance (195,290 parameters). The older `all_models_v2_20261018/comparison.csv` intentionally points at `modelA_20261011` (25,994 parameters); do not substitute it for the current ModelA comparison.
- Native project validation (`PYTHONPATH=src .venv/bin/python`, calling `verify_splits`) passed: 27,000 source images, 18,900/4,050/4,050 train/val/test, dataset SHA-256 `548a976e7d3f714d69f2a93f1abaf6261494537faa33af10f762ad7614644dd1`. Canonical split file checksums match metadata; manifests have no repeated or intersecting paths.
- All six ModelBV2 saved source hashes match current files; its three configs and metrics share the verified dataset and manifest hashes. MobileNetV2 also records matching split hashes. ModelA CUDA records the same dataset hash but no manifest/code hash. EfficientNet records the split directory but not a dataset/manifest/code hash.
- EfficientNet `train.log` epoch records agree with `metrics.json`; its existing `best.pt` was read without executing inference and confirms epoch 9 and validation accuracy 0.9780246913580247.

## Errors versus limitations

**Confirmed numerical errors:** none found in the scoped saved histories, selection summaries, or confusion-matrix metrics.

**Artifact/reproducibility gaps:** ModelA CUDA's three referenced `best_model.pt` files, all three ModelBV2 `best_model.pt`/`model_weights.pt` files, and MobileNetV2's referenced `best_model.pt` are absent from their saved run folders. Therefore these are internally consistent result artifacts, not locally checkpoint-reproducible experiments. Windows absolute paths in ModelA/MobileNet settings are provenance from the original machines, not valid current-machine paths. EfficientNet is the only scoped run with its selected checkpoint present. No predictions/logits are saved for independently reconstructing metrics from samples. ModelBV2 has null git revision/dirty-state fields, though source hashes mitigate that gap. Existing EfficientNet provenance cannot prove its historical split bytes merely from today's directory.

**Comparison limitations:** budgets differ (20/60/20/10 epochs), pretrained versus scratch training differs, hardware/software differ (ModelA Windows RTX4060 laptop, ModelBV2 Linux T4, MobileNet CPU, EfficientNet CUDA), and checkpoint selection rules differ. Accuracy comparisons are descriptive, not controlled equal-budget experiments; raw epoch-time rankings are not hardware-normalized. Each is one seed. ModelA SGD's large late validation-loss movement is observed behavior, not a matrix/summary inconsistency. EfficientNet epoch-3/4 high validation loss despite high accuracy can reflect a few confidently wrong predictions; no saved logits allow deeper diagnosis.

## Test-leakage review

- No direct test-training leakage found in current code: `model_a_train.py:95,110` chooses checkpoints on validation loss; `run_model_a_experiments.py:238` evaluates test after loading the chosen checkpoint. `mobilenet_train.py:182-204` does likewise. `train_efficientnet_b0.py:101-122` chooses validation accuracy then evaluates test once.
- ModelBV2 is strongest operationally: `model_b_v2_data.py:19-29` estimates normalization from train.csv only; its training loaders exclude test by default; `model_b_v2_train.py:161-194` requires completed training and explicit final evaluation, checks source/data contracts, and retains existing test results rather than reevaluating. MobileNet/EfficientNet use fixed ImageNet normalization, not test-derived statistics.
- Remaining risk: saved test results have already been exposed while architectures/hyperparameters and fresh runs are being developed. Choosing a new architecture/budget because it wins this same test set creates adaptive test-set reuse even if each training script only evaluates once. Do not describe the reused test set as a pristine blind final test; report the exploratory reuse or reserve a new untouched evaluation set.
- Split validation proves path-disjoint random per-class splits, not spatial independence or absence of visually near-duplicate satellite tiles across splits. No geographic holdout/near-duplicate analysis was performed. This is a generalization limitation, not proof of leakage.
