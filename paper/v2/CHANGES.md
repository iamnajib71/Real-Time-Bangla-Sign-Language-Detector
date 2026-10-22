# Changes from version 1

- v1 PDF copied byte-for-byte and kept unchanged. The revision does not alter historical claims.
- Reconciled the surviving 128x128 CNN implementation with the paper's conflicting 64x64, sparse-autoencoder, depth and MATLAB descriptions. Only the surviving CNN was retrained.
- Added fixed seed 2026, explicit image-level train/validation/test manifests, duplicate/conflict checks, environment versions and model hashes.
- Added reproduction run accuracy, full per-class metrics, confusion matrices, complete predictions and a same-split frozen MobileNetV2 comparison.
- Added a separate licensed public KU-BdSL USLD experiment. Its grouped labels are not mixed with the thesis classes.
- Added error examples, recorded-file software validation, data/licence/consent caveats and accessibility limitations.
- Added versioned code/model availability and verified DOI references. No new venue or peer-review claim.
- Author action: review and upload as a new version of the same ScienceOpen preprint. The draft has no assigned v2 DOI; do not invent one.

Every table is generated from reports/*/metrics.json and per_class.csv. paper/v2/numbers.json records the exact source values used by the manuscript generator. Examples originate from the saved test predictions.

- Added the separate signer-ID-disjoint bounded sentence experiment, false accepts, test-reuse disclosure, call companion and optional temporal neural model. No general-translation claim.

- Patch release: verified the real video-file selection flow, added a media-clock sampling regression test, and added pipeline, validation-history and temporal confusion figures. The original v2.0.0 release/tag remains unchanged.
