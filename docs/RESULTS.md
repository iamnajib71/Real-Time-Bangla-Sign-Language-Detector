# Reproduction results

These are new measurements. They are not the 2022 thesis figures, and the original preprint has not yet been updated on ScienceOpen.

| Experiment | Dataset | Classes | Train / validation / test | Test accuracy | Macro F1 |
|---|---|---:|---|---:|---:|
| original | Thesis Gray | 36 | 709 / 148 / 148 | 3.38% | 0.0018 |
| mobilenet | Thesis Gray | 36 | 709 / 148 / 148 | 52.70% | 0.5072 |
| ku-mobilenet | KU-BdSL USLD | 30 | 1020 / 240 / 240 | 99.58% | 0.9958 |

All experiments use seed 2026 and 12 epochs. The two thesis experiments use the **identical saved split**. The public-data experiment is separate: its grouped 30-class label space differs from the 36-class thesis data. Do not interpret these rows as a single three-way model ranking.

The original CNN uses the architecture in `v2022-thesis:ishara.py`, including the historical shear/zoom/flip augmentation. The transfer baseline uses frozen ImageNet MobileNetV2 features and trains only a new classifier, without geometric augmentation. It is an established baseline, not a new architecture or controlled architecture-only ablation.

## Evidence

Each experiment directory has inventory, split, label order, training history, per-class CSV, confusion matrix CSV/PNG, every test prediction, model SHA256 and runtime metadata. Accuracy is computed once from argmax predictions on the untouched test partition. Undefined per-class metrics are set to zero; support is shown. Training accuracy is diagnostic logging and is not the reported test metric.

22 software tests passed locally. See `reports/tests.xml`. The CI workflow runs on CPU; remote workflow status must be checked on GitHub separately. The synthetic fixture model proves plumbing, not sign recognition quality.

## Interpretation

This is a small-data feasibility study. A seeded image-level split is reproducible, but it is not a signer-independent benchmark. Exact preprocessed duplicates are removed before splitting; near-duplicate captures, shared backgrounds and repeated signers can still inflate estimates. There is no confidence calibration, external unseen-signer test, natural signing video benchmark, latency benchmark or accessibility user study.

The original architecture's outcome is retained even if it performs poorly. There was no search over seeds, hyperparameters or test outcomes. The new data run is a separate current-project result and must not be substituted for a thesis measurement.


Bounded sentence experiment: **6/10 supported sentences correct; 8/10 background clips rejected; 10 accepted captions with 60% precision** on dataset signer ID U19; background types occur in training. The prior baseline outcome was observed, so this is exploratory. [Protocol and vocabulary](SENTENCES.md). Neural weights/configuration are included as optional release assets.
