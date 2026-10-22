Modern rebuild of the 2022 thesis project. The original state is preserved at v2022-thesis; history has not been rewritten.

Reproduction results: original CNN 3.38% on 148 thesis test images; frozen MobileNetV2 52.70% on the same split; separate KU-BdSL USLD MobileNetV2 experiment 99.58% on 240 images. These are image-level tests, not unseen-signer benchmarks. See reports for every prediction, class metric and model hash.

CLI image, folder, recorded-video and camera inputs; PyQt5 desktop client; 28 passing automated tests in CPU CI including fixture-video end-to-end inference; CPU CI workflow; 30-second held-out image demo; failure audit; unchanged v1 preprint and a v2 manuscript draft for author review/upload.

Download a .keras file with its matching .labels.json. Default: ku-mobilenet.keras. Model hashes are in SHA256SUMS.txt and the committed reports. legacy-model1.h5 is archived for provenance only: its historic output mapping is uncertain. The paper PDF is a draft, not a new published version.

Camera preview and Bengali letter-to-text composition are included. The separate motion-template feasibility test recognised 1/4 supported phrases in another tutorial and rejected 4/8 unsupported phrases; it remains experimental, with full errors reported.


Bounded sentence experiment: **6/10 supported sentences correct; 8/10 background clips rejected; 10 accepted captions with 60% precision** on dataset signer ID U19; background types occur in training. The prior baseline outcome was observed, so this is exploratory. [Protocol and vocabulary](SENTENCES.md). Neural weights/configuration are included as optional release assets.

- Patch release: verified the real video-file selection flow, added a media-clock sampling regression test, and added pipeline, validation-history and temporal confusion figures. The original v2.0.0 release/tag remains unchanged.
