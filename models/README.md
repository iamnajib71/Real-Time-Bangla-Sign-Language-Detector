# Models are release assets

Model binaries are excluded from the current Git tree. The 2022 file remains in the immutable `v2022-thesis` history; it has an uncertain output-label contract and is not used for current evaluation.

Download the model and matching `.labels.json` from [v2.0.0](https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/tag/v2.0.0), or train locally with the commands in the README. Keep both files together. Each experiment records its model SHA256 under `reports/<experiment>/model.json`. Do not interchange label maps.
