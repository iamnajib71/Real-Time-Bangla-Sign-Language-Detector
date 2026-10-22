# ScienceOpen revision form: copy-ready information

This is for the existing ScienceOpen preprint, not ScienceDirect. Use the revision workflow for DOI 10.14293/S2199-1006.1.SOR-.PPUF56Q.v1. Do not create a separate publication record or invent a version-2 DOI. The platform assigns version metadata. The author reviews and submits; nothing has been uploaded automatically.

## Content

Upload the revised clean manuscript: `paper/v2/main.pdf` (10 pages). This file is distinct from the earlier manuscript attached to release v2.0.1. The code and model release remains v2.0.1 because the present changes are editorial.

## Title

Bangla Sign Language Gesture Recognition: Static Image Classification and Bounded Sentence Recognition

## Subtitle

Leave blank; the complete title is supplied above.

## Authors and contact

Preserve the author list of the existing preprint. For Nazmul's entry:

- Given name: Nazmul
- Surname: Hassan
- ORCID: https://orcid.org/0000-0001-6116-017X
- Affiliation: North South University, Dhaka, Bangladesh
- Corresponding author email: iam.najib@yahoo.com
- Language: English

Use ScienceOpen's institutional lookup for the ROR identifier. Add a department only if it accurately reflects the research affiliation; do not infer a current appointment from the thesis affiliation.

## Abstract (163 words; matches the PDF)

This study evaluates static Bangla hand-sign classification and exploratory recognition of a bounded sentence vocabulary, extending an undergraduate thesis system. The surviving Python architecture is retrained once, while a frozen MobileNetV2 classifier provides a modest comparison. A separate experiment uses the openly licensed KU-BdSL USLD collection; its grouped label space is kept distinct from the historical dataset. Evaluation uses fixed seeded partitions, content audits, class metrics, confusion matrices and prediction-level evidence. The historical CNN obtains 3.38% on 148 held-out images; the frozen comparison obtains 52.70% on the same split. The separate public-data model obtains 99.58% on 240 held-out images. These image-level tests do not establish unseen-signer performance. The software accepts images and recorded files as well as cameras, and a fixture-video test verifies the pipeline without camera hardware. A separate exploratory ten-sentence classifier recognises 6 of ten supported test clips and rejects 8 of ten background clips. The results support constrained recognition experiments; continuous translation and performance with new signers require further evaluation.

## Revision notes (public; paste the following)

1. Clarified the implemented CNN architecture and documented differences between the surviving code and the original method description. The original preprint remains unchanged and citable.
2. Added fixed train/validation/test partitions, a recorded random seed, duplicate and label-conflict audits, pinned dependencies, model checksums and prediction-level evidence.
3. Added new measurements on the historical dataset: 3.38% test accuracy for the reconstructed CNN and 52.70% for frozen MobileNetV2 on the same 148-image test split. These measurements are separate from the original preprint's accuracy statements.
4. Added a separate KU-BdSL USLD experiment, with 99.58% accuracy on 240 test images. Its different collection and grouped label space are explicitly distinguished from the historical task; the split is image-level, not a verified unseen-signer evaluation.
5. Added per-class metrics, numbered confusion matrices, validation curves, a pipeline diagram and correct/incorrect examples.
6. Added an exploratory ten-sentence temporal classifier: 6/10 supported test captions correct and 8/10 background clips rejected. The manuscript discloses the earlier observation of this test signer, background-type overlap and incorrect captions in tutorial-video integration checks.
7. Added recorded-video and camera interfaces, software tests, and code/model availability links. Clarified data permissions, consent limitations, accessibility evaluation needs and the scope of the recognition claims.
8. Revised the abstract and academic presentation, removed obsolete unsupported performance wording, and standardised headings, table/figure captions and citation formatting.

These are author-initiated revisions; no formal reviewer or editor comments were supplied for this revision.

## Keywords (one per line; maximum 10)

Bangla Sign Language
Sign language recognition
Computer vision
Convolutional neural networks
Transfer learning
MobileNetV2
Temporal modelling
Accessibility
Reproducible research
Human-computer interaction

## Disciplines

Choose the closest available entries for Computer Science / Artificial Intelligence / Computer Vision. The existing Neural & Evolutionary Computing entry is also consistent with the neural models. These are subject descriptions, not a claim that the exact labels exist in the menu.

## AI usage declaration (internal field)

Generative AI-assisted coding, analysis tooling and manuscript editing were used. Select the Yes option. Complete the required author review and validation before affirming the form's responsibility statement. Do not select No. This submission declaration is separate from the paper's running text and from Git commit authorship.

## Data availability

The repository option is the closest fit for the reproducibility records. Include the following qualification wherever the form permits free text; one unrestricted-data claim would not accurately describe all three sources.

Code, split manifests, model metadata, per-class metrics and prediction-level outputs are available at https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector. Model files, label maps and checksums are archived in release https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/tag/v2.0.1. KU-BdSL version 4 is available from Mendeley Data at https://doi.org/10.17632/scpvm2nbkm.4 under CC BY 4.0. Historical thesis records are preserved in the repository history; original consent records are unavailable. The bounded sentence experiment uses source clips from https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1; reuse permissions are unresolved, and the source clips are not redistributed. The MIT code licence does not apply to third-party data.

## Funder

Enter only a confirmed grant or funding body. No verified funding information has been supplied; leave this optional field blank unless the author can confirm it. Do not invent a grant or a no-funding declaration.

## Reference DOIs (one per line)

10.14293/S2199-1006.1.SOR-.PPUF56Q.v1
10.1016/j.dib.2023.109797
10.17632/scpvm2nbkm.4
10.48550/arXiv.1801.04381

These match the four references in the manuscript. Remove the unrelated example DOI entries already visible in the pasted form.

## Thumbnail

An optional suitable image is the project-authored pipeline diagram in `paper/v2/figures/pipeline.png`. Do not use unlicensed tutorial faces. The pasted form warns that choosing a thumbnail can clear text fields, so choose it before pasting metadata.

## PDF for tracked changes

The clean manuscript is not a tracked-changes PDF. This task produced a clean revised PDF and textual revision notes, not a marked comparison against v1. If the field is mandatory, a separate marked comparison is still needed before submission; do not present the clean PDF or CHANGES.md as tracked changes.

## Submission declarations

Review these personally. Because this is an existing preprint revision, the statement that the article has never been posted elsewhere must be interpreted within the correct revision workflow; do not assert that the original preprint never existed. Preserve co-author permissions where applicable. The preprint remains non-peer-reviewed. The CC BY 4.0 submission agreement covers material you have authority to license and does not grant new rights over third-party source videos.

## Rebuilding the professional manuscript

After running any report-driven manuscript generators, run `python scripts/polish_manuscript.py` last. Compile `paper/v2/main.tex` using Tectonic or upload its self-contained source folder to Overleaf. Run `python scripts/verify_publication.py` to compare manuscript values against the saved model outputs. This edit changes presentation and wording, not model predictions or the original v1 document.
