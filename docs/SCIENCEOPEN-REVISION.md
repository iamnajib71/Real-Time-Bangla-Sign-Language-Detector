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
- Affiliation: Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh
- ROR: https://ror.org/05wdbfp45
- Corresponding author email: iam.najib@yahoo.com
- Language: English

Affiliation: Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh (research affiliation from the original thesis). ROR: **https://ror.org/05wdbfp45**. If the field accepts only the identifier, enter **05wdbfp45**. Use the university record for ROR; ORCID identifies the author, not the institution. The official record is saved in `docs/editorial-sources.json`. This does not assert a current university appointment.

## Abstract (163 words; matches the PDF)

This study evaluates static Bangla hand-sign classification and exploratory recognition of a bounded sentence vocabulary, extending an undergraduate thesis system. The surviving Python architecture is retrained once, while a frozen MobileNetV2 classifier provides a modest comparison. A separate experiment uses the openly licensed KU-BdSL USLD collection; its grouped label space is kept distinct from the historical dataset. Evaluation uses fixed seeded partitions, content audits, class metrics, confusion matrices and prediction-level evidence. The historical CNN obtains 3.38% on 148 held-out images; the frozen comparison obtains 52.70% on the same split. The separate public-data model obtains 99.58% on 240 held-out images. These image-level tests do not establish unseen-signer performance. The software accepts images and recorded files as well as cameras, and a fixture-video test verifies the pipeline without camera hardware. A separate exploratory ten-sentence classifier recognises 6 of ten supported test clips and rejects 8 of ten background clips. The results support constrained recognition experiments; continuous translation and performance with new signers require further evaluation.

## Revision notes (public; paste the following)

Thank you for the editorial screening comments. The affiliation now includes North South University's verified ROR identifier, https://ror.org/05wdbfp45. Section 2 has been expanded to cover recognition versus continuous signing and translation, Bangla datasets and recent temporal models, and representation and accessibility considerations. The bibliography has increased from 4 to 14 references, with one self-reference. The added sources have verified DOIs; arXiv sources are labelled as preprints. The review distinguishes task definitions, data and evaluation protocols and avoids direct accuracy rankings across incompatible studies. Otsu preprocessing and MediaPipe hand tracking now have explicit methodological citations. The measured results, data partitions, model weights and predictions remain unchanged. A clean manuscript and a marked comparison with the previously prepared submission are included. The original preprint remains preserved unchanged.

The complete point-by-point response is in `paper/v2/RESPONSE-TO-EDITOR.md`.

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

Answer the platform's tool-use disclosure question accurately. Complete the required author review and validation before affirming the form's responsibility statement. Do not select No. This submission declaration is separate from the paper's running text and from Git commit authorship.

## Data availability

The repository option is the closest fit for the reproducibility records. Include the following qualification wherever the form permits free text; one unrestricted-data claim would not accurately describe all three sources.

Code, split manifests, model metadata, per-class metrics and prediction-level outputs are available at https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector. Model files, label maps and checksums are archived in release https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/tag/v2.0.1. KU-BdSL version 4 is available from Mendeley Data at https://doi.org/10.17632/scpvm2nbkm.4 under CC BY 4.0. Historical thesis records are preserved in the repository history; original consent records are unavailable. The bounded sentence experiment uses source clips from https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1; reuse permissions are unresolved, and the source clips are not redistributed. The MIT code licence does not apply to third-party data.

## Funder

Enter only a confirmed grant or funding body. No verified funding information has been supplied; leave this optional field blank unless the author can confirm it. Do not invent a grant or a no-funding declaration.

## Reference DOIs (one per line; manuscript order)

10.1145/3308561.3353774
10.14293/S2199-1006.1.SOR-.PPUF56Q.v1
10.1109/cvpr.2018.00812
10.1109/cvpr42600.2020.01004
10.1016/j.ipm.2024.103774
10.1016/j.dib.2023.109797
10.17632/scpvm2nbkm.4
10.48550/arxiv.2408.10518
10.48550/arxiv.2503.02360
10.1038/s41598-025-30856-y
10.1109/iccit68739.2025.11491213
10.48550/arXiv.1801.04381
10.48550/arxiv.2006.10214
10.1109/tsmc.1979.4310076

These match all 14 references in the revised manuscript. Replace the old reference metadata list in the form; do not retain unrelated example DOIs.

## Thumbnail

Upload `assets/scienceopen-thumbnail.png`, a dedicated 1536 x 1024 PNG cover. This is an illustrative cover generated with the built-in image tool, not an experimental figure or an authenticated sign diagram. The generation prompt is saved in `docs/thumbnail-prompt.txt`. It uses no source-video faces or accuracy claims. The project-authored pipeline in `paper/v2/figures/pipeline.png` is also available as an alternative. The pasted form warns that choosing a thumbnail can clear text fields, so choose it before pasting metadata.

## PDF for tracked changes

Upload `paper/v2/tracked-changes.pdf` (21 pages, approximately 0.59 MB, below the 10 MB limit). It compares the revised paper with the previously prepared 10-page submission preserved at `paper/submissions/initial-revision/main.pdf`. Confirm that baseline matches the file you uploaded. Blue underlining marks additions and red strikethrough marks removals; the markup is embedded in page content. Source checksums and the baseline path are in `paper/v2/tracked-changes.json`. The cover explains PDF text extraction limitations; graphics remain visible in each version but are not compared pixel by pixel. The original ScienceOpen v1 remains unchanged in `paper/v1/`. Use the clean `paper/v2/main.pdf` in the main Content field.

Rebuild the comparison with `python scripts/build_tracked_changes.py --baseline paper/submissions/initial-revision/main.pdf --baseline-label "Previously prepared ScienceOpen submission, before the editorial ROR and literature request"` (requires PyMuPDF).

## Submission declarations

Review these personally. Because this is an existing preprint revision, the statement that the article has never been posted elsewhere must be interpreted within the correct revision workflow; do not assert that the original preprint never existed. Preserve co-author permissions where applicable. The preprint remains non-peer-reviewed. The CC BY 4.0 submission agreement covers material you have authority to license and does not grant new rights over third-party source videos.

## Rebuilding the professional manuscript

After running any report-driven manuscript generators, run `python scripts/polish_manuscript.py` last. Compile `paper/v2/main.tex` using Tectonic or upload its self-contained source folder to Overleaf. Run `python scripts/verify_publication.py` to compare manuscript values against the saved model outputs. This edit changes presentation and wording, not model predictions or the original v1 document.

## Resubmission following the editorial email

Log in to ScienceOpen, open the existing article and choose **Resubmit**, as directed in the email. Update the affiliation ROR field, upload the clean and marked PDFs to their respective fields, and replace the reference DOI list. Paste the updated revision notes or the point-by-point response. The title and abstract remain unchanged. This package prepares a resubmission; no ScienceOpen upload or metadata edit has been performed automatically, and peer review has not been claimed complete.

The post-screening `paper/v2/main.tex` is the authoritative manuscript source. Earlier report-driven generators predate this literature revision; do not overwrite the expanded review with their older template. The source folder remains self-contained for Overleaf. `scripts/polish_manuscript.py` preserves the editorial version when its verified ROR and expanded bibliography are present.
