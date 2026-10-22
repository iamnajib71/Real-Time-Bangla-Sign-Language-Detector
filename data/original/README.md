# Historical thesis dataset

`Gray`, `Resized`, and `Otsu_again` were supplied in the 2022 repository. They are representations of the same collection, not independent datasets. The reproduction run reads **Gray only**, applies the shared 128x128 Otsu preprocessing, and creates a new deterministic train/validation/test split. The old train/test folders are retained only for audit.

The original paper describes camera-collected hand crops and constrained backgrounds. Signer identifiers, capture sessions, consent records and independently verified class annotations are not present in the repository. Do not claim a verified single-signer count or generalisation to unseen signers. The old GUI label mapping is retained as provenance, with class 6's duplicate `উ` corrected to `ও`; expert label validation remains outstanding.

The software MIT licence does not establish additional rights or participant consent for these historical data. Prefer the separately licensed KU-BdSL experiment for redistribution and future development. Never pool corresponding Gray/Resized/Otsu images across splits. See reports/original for the inventory and per-image split.
