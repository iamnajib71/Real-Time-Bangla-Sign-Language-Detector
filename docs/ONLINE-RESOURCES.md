# Public Bengali sign resources

The new public-data experiment uses **KU-BdSL v4, USLD only**, without mixing its label space with the thesis dataset.

- Dataset: Abdullah Al Jaid Jim, Ibrahim Rafi, Md. Zahid Akon, Uzzal Biswas, Abdullah-Al Nahid (2023), [KU-BdSL v4](https://doi.org/10.17632/scpvm2nbkm.4). CC BY 4.0. The publisher documents written consent for v4. The downloader verifies each publisher SHA256.
- Dataset paper: the same authors, [KU-BdSL: An open dataset for Bengali sign language recognition](https://doi.org/10.1016/j.dib.2023.109797), Data in Brief 51, 109797 (2023).
- Alternatives for future experiments: [BdSL47](https://doi.org/10.5061/dryad.1vhhmgqwk), and [BDSL49 v6](https://doi.org/10.17632/k5yk4j8z8s.6). These are resources, not datasets used to produce this project's current results.
- Bengali rendering: [Noto Sans Bengali](https://github.com/google/fonts/tree/main/ofl/notosansbengali), SIL Open Font License. Local font and licence in assets/fonts.
- Transfer baseline: Sandler et al., [MobileNetV2](https://doi.org/10.48550/arXiv.1801.04381). This established baseline is not a novel architecture.

## Dataset contract

The publisher describes 1,500 USLD images in 30 classes, with some classes grouping multiple Bengali consonants. Keep these group labels: do not silently split them into independent letter classes. Folder names encode decimal Unicode points (including hyphen-separated groups). The repository download manifest records the actual file counts and hashes. The split manifest records every usable example and excluded duplicate or conflicting input.

The source describes 39 participants and smartphone capture on flat surfaces under varying brightness. Individual signer IDs are not supplied in the class-folder layout used here. Consequently the measured split is image-level; it cannot establish unseen-signer accuracy. Exact preprocessed duplicates are grouped before splitting; near duplicates may remain.

## Redistribution and privacy

Data is not covered by the software MIT licence. KU-BdSL retains CC BY 4.0. Demo thumbnails are resized derivatives and carry dataset attribution. No unlicensed video or third-party face clip is used. The demo is a labelled slideshow of held-out hand images, not evidence of continuous sign recognition. Real Bangla Sign Language includes motion, grammar and facial expression beyond these isolated static classes.
