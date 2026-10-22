"""Generate documentation and the v2 manuscript from measured outputs only."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REPO='https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector'
RELEASE=REPO+'/releases/tag/v2.0.1'

def load(experiment,file='metrics.json'):
    return json.loads((Path('reports')/experiment/file).read_text(encoding='utf-8'))

def main():
    runs={x:load(x) for x in ['original','mobilenet','ku-mobilenet']}
    audits={x:load(x,'inventory.json') for x in runs}
    test_file='reports/ci-tests.xml' if Path('reports/ci-tests.xml').exists() else 'reports/tests.xml'
    test_context='CPU CI' if 'ci-tests' in test_file else 'local verification'
    test=ET.parse(test_file).getroot()
    suites=[test] if test.tag=='testsuite' else list(test)
    tests=sum(int(s.get('tests',0)) for s in suites)
    failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    if failures:
        raise ValueError('Tests are failing; refusing documentation generation')
    public=runs['ku-mobilenet']; original=runs['original']; mobile=runs['mobilenet']
    comparison='\n'.join(f"| {x} | {'KU-BdSL USLD' if x.startswith('ku') else 'Thesis Gray'} | {m['classes']} | {audits[x]['split_counts']['train']} / {audits[x]['split_counts']['validation']} / {m['test_n']} | {m['accuracy']:.2%} | {m['per_class']['macro avg']['f1-score']:.4f} |" for x,m in runs.items())
    results=f'''# 2026 re-run results

These are new measurements. They are not the 2022 thesis figures, and the original preprint has not yet been updated on ScienceOpen.

| Experiment | Dataset | Classes | Train / validation / test | Test accuracy | Macro F1 |
|---|---|---:|---|---:|---:|
{comparison}

All experiments use seed 2026 and 12 epochs. The two thesis experiments use the **identical saved split**. The public-data experiment is separate: its grouped 30-class label space differs from the 36-class thesis data. Do not interpret these rows as a single three-way model ranking.

The original CNN uses the architecture in `v2022-thesis:ishara.py`, including the historical shear/zoom/flip augmentation. The transfer baseline uses frozen ImageNet MobileNetV2 features and trains only a new classifier, without geometric augmentation. It is an established baseline, not a new architecture or controlled architecture-only ablation.

## Evidence

Each experiment directory has inventory, split, label order, training history, per-class CSV, confusion matrix CSV/PNG, every test prediction, model SHA256 and runtime metadata. Accuracy is computed once from argmax predictions on the untouched test partition. Undefined per-class metrics are set to zero; support is shown. Training accuracy is diagnostic logging and is not the reported test metric.

{tests} automated tests passed in {test_context}. See `{test_file}` and the recorded CI run in `reports/ci-verification.json` when present. The synthetic fixture model proves plumbing, not sign recognition quality.

## Interpretation

This is a small-data feasibility study. A seeded image-level split is reproducible, but it is not a signer-independent benchmark. Exact preprocessed duplicates are removed before splitting; near-duplicate captures, shared backgrounds and repeated signers can still inflate estimates. There is no confidence calibration, external unseen-signer test, natural signing video benchmark, latency benchmark or accessibility user study.

The original architecture's outcome is retained even if it performs poorly. There was no search over seeds, hyperparameters or test outcomes. The new data run is a separate current-project result and must not be substituted for a thesis measurement.
'''
    Path('docs/RESULTS.md').write_text(results,encoding='utf-8')
    readme=f'''# Bangla Sign Lab

I built a reproducible Bangla hand-sign recognition project to explore more accessible text input, with recorded-file demos and transparent evaluation.

![30-second held-out sign-image demo](assets/ku-mobilenet/demo.gif)

[![CPU tests]({REPO}/actions/workflows/ci.yml/badge.svg)]({REPO}/actions/workflows/ci.yml) ![Python 3.12](https://img.shields.io/badge/Python-3.12-blue) ![Code licence MIT](https://img.shields.io/badge/code-MIT-green)

## At a glance

| | |
|---|---|
| **The problem** | Bangla sign language users lack accessible real-time recognition tools, adding friction when communicating with people who do not sign. This project explores a small static-letter classification task. |
| **What I built** | I built a current TensorFlow/Keras pipeline, compared a CNN with a transfer-learning baseline, and connected the same predictor to image, video, camera and PyQt5 inputs. |
| **Results** | Public KU-BdSL experiment: **{public['accuracy']:.2%} test accuracy on {public['test_n']} held-out images across {public['classes']} grouped classes**. {tests} automated tests passed in {test_context}. These are 2026 re-run results; signer-independent performance is unmeasured. |
| **Skills shown** | Python, OpenCV, TensorFlow/Keras, CNN, computer vision, PyQt5, reproducible experiments, data analysis, pytest, GitHub Actions. |
| **Relevant roles** | Junior ML/AI Engineer, Data Analyst, Junior Data Scientist, Software Engineer. |
| **How it was built** | I designed the workflow, implemented the rebuild and reviewed the measured results; CPU CI checks the shared prediction pipeline. |
| **Try it / evidence** | [Quick start](#quick-start), [results](docs/RESULTS.md), [complete prediction page](docs/ku-mobilenet-results.html), [raw evidence](reports/), [release]({RELEASE}). |

The animation is a **30-second slideshow of held-out hand images**, with selected correct and incorrect predictions. It is not a natural-signing video or proof of continuous language translation. [Open the full results page](docs/ku-mobilenet-results.html) locally in a browser to inspect every test example.

## What I did, step by step

1. Preserved the thesis repository as `v2022-thesis` and rebuilt the main tree around the current application.
2. Audited the labelled data, avoided mixing alternate versions of the same image, and fixed a seeded train/validation/test partition.
3. Ported the original CNN to `tensorflow.keras` and measured a single 12-epoch re-run.
4. Added a frozen MobileNetV2 comparison on the same split, then a separate experiment using the licensed public KU-BdSL dataset.
5. Reused preprocessing and prediction for the CLI and PyQt5, including recorded-video and folder inputs.
6. Tested Otsu preprocessing, fixture predictions, GUI input and an entire fixture video without a camera.
7. Published prediction-level evidence, failure examples and a revision draft that separates new measurements from historical claims.

## Problems I hit and how I solved them

- The old code used machine-specific paths and removed Keras APIs. I replaced it with an installable package and repository-relative model paths.
- The original predictor compared probabilities with exactly 1 and assumed numeric class ordering. I use argmax and save the actual ordered label map alongside every model.
- The three old image folders represent the same collection. I use one canonical version, audit content hashes and split before augmentation.
- The paper describes methods that do not match the surviving Python implementation. I preserve v1 and state exactly which implementation the new experiment measures.

## Architecture

```mermaid
flowchart LR
  A[Image / image folder / recorded video / camera] --> B[Resize 128 x 128 and Otsu threshold]
  B --> C[Normalize and classify]
  C --> D[Ordered label and uncalibrated score]
  D --> E[JSONL CLI / PyQt5]
  F[Fixed split and training] --> G[Model + label map + SHA256]
  G --> C
```

## Quick start

Use Python 3.12; Python 3.14 is not supported by this pinned environment. CPU inference and file demos require no webcam. From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python scripts/download_models.py
python -m bsl predict image.png
python -m bsl predict --images path/to/hand-crops --output predictions.jsonl
python -m bsl webcam --video clip.mp4 --output predictions.jsonl
python -m bsl webcam
python -m bsl gui
python -m pytest -q
```

On Linux/macOS use `python3.12` and `source .venv/bin/activate`. Frame inputs should already contain a centred hand crop. This is a whole-frame classifier; it does not find hands in arbitrary scenes. CLI output is UTF-8 JSONL; use Ctrl+C to stop a camera stream. The PyQt5 app provides image/video selection and a camera preview.

## Results and evidence

**2026 re-run results** (not thesis figures):

| Experiment | Dataset | Classes | Train / validation / test | Test accuracy | Macro F1 |
|---|---|---:|---|---:|---:|
{comparison}

The two thesis rows share a split. KU-BdSL is a separate public-data experiment with a different label space. [Evaluation details and caveats](docs/RESULTS.md); [per-class public metrics](reports/ku-mobilenet/per_class.csv); [public confusion matrix](reports/ku-mobilenet/confusion_matrix.png); [all mistakes](docs/ku-mobilenet-results.html).

## Reproduce training and the demo

```powershell
python scripts/download_kubdsl.py
python -m bsl.train --data data/external/ku-bdsl-usld --external --architecture mobilenet --out reports/new-public-run --model models/new-public.keras
python scripts/build_evidence.py --report reports/new-public-run --data data/external/ku-bdsl-usld --name new-public
```

The downloader checks publisher SHA256 values. The trainer refuses to overwrite an already measured output directory. Frozen MobileNetV2 weights download from Keras on the first run. The recorded experiment uses seed 2026, 12 epochs and a frozen backbone; it is not claimed to be the latest research architecture. See [online resources and attribution](docs/ONLINE-RESOURCES.md).

The 2022 `Gray` images can be recovered from `v2022-thesis` into `data/original/Gray` for the historical re-run:

```powershell
python scripts/restore_thesis_data.py
python -m bsl.train --data data/original/Gray --out reports/new-original-run --model models/new-original.keras
python -m bsl.train --data data/original/Gray --architecture mobilenet --split-from reports/new-original-run/split.json --out reports/new-comparison --model models/new-comparison.keras
```

## Project structure

```text
src/bsl/        CLI, shared predictor, preprocessing, models, PyQt5, training
scripts/        verified data/model downloads, demos and evidence generation
tests/          preprocessing, real Keras fixture model, GUI, fixture video
assets/         held-out demo images, GIFs, openly licensed Bengali font
reports/        immutable measured splits, hashes, metrics and predictions
docs/           results pages, provenance, release notes, publication handoff
paper/v1/       unchanged 2022 preprint
paper/v2/       Overleaf-compatible revision draft and compiled PDF
models/         ignored local binaries; published as release assets
```

## Citation and research history

This rebuild grew from my 2022 B.Sc. thesis at North South University. The original work is a **non-peer-reviewed ScienceOpen preprint**:

> Hassan, Nazmul (2022). *Bangla Sign Language Gesture Recognition System: Using CNN Model*. ScienceOpen Preprints. DOI: [10.14293/S2199-1006.1.SOR-.PPUF56Q.v1](https://doi.org/10.14293/S2199-1006.1.SOR-.PPUF56Q.v1).

Use [CITATION.cff](CITATION.cff) for software citation. The [v2 manuscript draft](paper/v2/main.pdf) adds reproducible code and 2026 measurements; it has **not** been uploaded or assigned a v2 DOI. Nazmul uploads it as a new version of the same preprint. [Publication profile checklist](docs/PUBLICATIONS.md).

## Honest limits

- Small datasets and constrained hand-crop conditions. Signer identities and sessions cannot be separated in the available folder labels.
- Image-level test scores may benefit from shared signers, backgrounds or similar captures. They do not measure performance on unseen signers.
- Static letter/group classification only; no continuous signing, words, grammar or facial-expression understanding.
- Historical class mappings still need review by a Bangla Sign Language expert. KU-BdSL uses grouped consonant labels.
- Softmax confidence is uncalibrated. No clinical, accessibility effectiveness or production-readiness claim.
- No real webcam or natural signing video benchmark was performed. The file pipeline was tested with a synthetic fixture video; the showcase uses held-out still images.

Code: MIT. Public dataset and thumbnails: CC BY 4.0, credited to the KU-BdSL authors. Fonts: SIL OFL. The historical paper retains its CC BY 4.0 licence; historical data provenance limits are documented separately.

## Built by Nazmul Hassan

[LinkedIn](https://www.linkedin.com/in/iamnajib71/) · [GitHub](https://github.com/iamnajib71) · [ORCID](https://orcid.org/0000-0001-6116-017X)
'''
    Path('README.md').write_text(readme,encoding='utf-8')
    Path('docs/RELEASE.md').write_text(f'''Modern rebuild of the 2022 thesis project. The original state is preserved at v2022-thesis; history has not been rewritten.

2026 re-run results: original CNN {original['accuracy']:.2%} on {original['test_n']} thesis test images; frozen MobileNetV2 {mobile['accuracy']:.2%} on the same split; separate KU-BdSL USLD MobileNetV2 experiment {public['accuracy']:.2%} on {public['test_n']} images. These are image-level tests, not unseen-signer benchmarks. See reports for every prediction, class metric and model hash.

CLI image, folder, recorded-video and camera inputs; PyQt5 desktop client; {tests} passing automated tests in {test_context} including fixture-video end-to-end inference; CPU CI workflow; 30-second held-out image demo; failure audit; unchanged v1 preprint and a v2 manuscript draft for author review/upload.

Download a .keras file with its matching .labels.json. Default: ku-mobilenet.keras. Model hashes are in SHA256SUMS.txt and the committed reports. legacy-model1.h5 is archived for provenance only: its historic output mapping is uncertain. The paper PDF is a draft, not a new published version.\n''',encoding='utf-8')
    build_paper(runs,audits)
    presentation_style()

def tex_escape(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def build_paper(runs,audits):
    out=Path('paper/v2'); out.mkdir(parents=True,exist_ok=True)
    import shutil
    (out/'figures').mkdir(exist_ok=True)
    for experiment in ['original','ku-mobilenet']:
        shutil.copyfile(Path('reports')/experiment/'confusion_matrix.png',out/'figures'/(experiment+'-confusion.png'))
    original=runs['original']; mobile=runs['mobilenet']; public=runs['ku-mobilenet']
    texrows='\n'.join(f"{tex_escape(x)} & {m['classes']} & {audits[x]['split_counts']['train']} & {audits[x]['split_counts']['validation']} & {m['test_n']} & {m['accuracy']*100:.2f} & {m['per_class']['macro avg']['f1-score']:.4f} \\\\" for x,m in runs.items())
    (out/'results-table.tex').write_text(r'\begin{tabular}{lrrrrrr}\toprule Experiment & Classes & Train & Val. & Test & Acc. (\%) & Macro F1 \\\midrule'+'\n'+texrows+'\n'+r'\bottomrule\end{tabular}',encoding='utf-8')
    for name in ['original','ku-mobilenet']:
        report=load(name)
        rows=[]
        for x in load(name,'labels.json'):
            label=x['folder']; m=report['per_class'][label]
            rows.append(f"{label} & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1-score']:.3f} & {int(m['support'])} \\\\")
        (out/f'{name}-classes.tex').write_text(r'\begin{tabular}{lrrrr}\toprule ID & P & R & F1 & N \\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}',encoding='utf-8')
    # Example selection is deterministic and recorded, not a curated accuracy estimate.
    from PIL import Image
    preds=load('ku-mobilenet','predictions.json')
    examples=[]
    for correct in [True,False]:
        match=next((r for r in preds if r['correct']==correct),None)
        if match:
            index=preds.index(match)
            Image.open(f'assets/ku-mobilenet/test/{index:03d}.png').save(out/('correct.png' if correct else 'incorrect.png'))
            labels=load('ku-mobilenet','labels.json')
            examples.append({'file':'correct.png' if correct else 'incorrect.png','actual':labels[match['actual_index']]['folder'],
                             'predicted':labels[match['predicted_index']]['folder'],'confidence':match['confidence'],'source':match['path']})
    (out/'examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    (out/'numbers.json').write_text(json.dumps({x:{k:m[k] for k in ['accuracy','test_n','classes','epochs','seed']} for x,m in runs.items()},indent=2),encoding='utf-8')
    example_tex='\n'.join(r'\begin{minipage}{.45\textwidth}\centering\includegraphics[height=3.2cm]{'+x['file']+'}\\ Actual ID '+x['actual']+'; predicted '+x['predicted']+f".\\\\ Score {x['confidence']*100:.1f}\\%."+r'\end{minipage}\hfill' for x in examples)
    source=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx,booktabs,amsmath,hyperref,fancyhdr}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{Bangla Sign Lab: reproducibility revision}\fancyhead[R]{Draft / 2026}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parskip}{6pt}\setlength{\parindent}{0pt}
\title{Bangla Sign Language Gesture Recognition System:\\A Reproducible CNN Re-run and Public-Data Baseline}
\author{Nazmul Hassan\\North South University, Bangladesh (B.Sc. thesis, 2022)\\\small ORCID: 0000-0001-6116-017X}
\date{Version 2 draft, 22 October 2026; not peer-reviewed}
\begin{document}\maketitle
\begin{abstract}
This revision reports a reproducible reconstruction of a static Bangla hand-sign classifier derived from a 2022 undergraduate thesis. The surviving Python architecture is retrained once, while a frozen MobileNetV2 classifier provides a modest comparison. A separate current experiment uses the openly licensed KU-BdSL USLD collection; its grouped label space is kept distinct from the historical dataset. All results are new 2026 measurements, with fixed seeded partitions, content audits, class metrics, confusion matrices and prediction-level evidence. The historical CNN obtains ORIGINALACC\% on ORIGINALN held-out images; the frozen comparison obtains MOBILEACC\% on the same split. The separate public-data model obtains PUBLICACC\% on PUBLICN held-out images. These image-level tests do not establish unseen-signer performance. The software accepts images and recorded files as well as cameras, and a fixture-video test verifies the pipeline without camera hardware. The scope is isolated static classes rather than continuous language translation, and the paper explicitly separates preserved historical claims from reproducible contemporary evidence.
\end{abstract}
\section{Introduction}
Accessible communication tools require reliable evidence as well as working interfaces. A recogniser for a small set of hand configurations may support a constrained input experiment, but it does not capture the movement, context and grammar of a natural sign language. This project therefore evaluates static class prediction, rather than claiming a general translator.

The original preprint appeared in 2022 on ScienceOpen and was not peer-reviewed~\cite{hassan}. Its methods narrative and the surviving code differ. This revision preserves that document, identifies the executable method, and adds a traceable code and model release. The present contribution is reproducibility and implementation quality, not a claim of a novel neural architecture or a new record on a public benchmark.
\newpage
\section{Related work and scope}
The KU-BdSL authors provide an openly available collection designed for Bengali consonant recognition~\cite{kubdsl}. The accompanying versioned dataset offers an explicit licence and documented collection conditions~\cite{kudata}. It is useful here because its class layout can be downloaded, hashed and audited without treating an undocumented image bundle as an independent benchmark.

MobileNetV2 introduces inverted residuals and linear bottlenecks for efficient vision models~\cite{mobilenet}. Here it is used only as an established, frozen ImageNet feature extractor. A small new classifier is learned on top. No claim is made that this older backbone represents the latest architecture, that it has been tuned optimally for Bangla signs, or that its scores are directly comparable with results from other papers.

These distinctions matter when rebuilding an old project. Replacing a historical dataset with a newer one changes both the data distribution and the output task. This revision consequently reports a historical-data comparison and a separate public-data experiment. A higher public-data score, if observed, would not constitute a direct improvement over the 2022 system.
\section{Data and provenance}
The historical repository contains 1,005 labelled images in each of three representations: Gray, Resized and Otsu\_again. They are not independent collections. The re-run uses Gray only; the other copies are excluded from training and evaluation. Its 36 numeric folder labels inherit a provisional display mapping from the old interface. The duplicate vowel shown for class 6 in that interface is mapped to O; expert verification remains outstanding. Numeric class identifiers are used in the quantitative tables to avoid hiding this limitation.

The historical paper describes constrained camera capture and hand cropping, but the repository supplies no verified signer identifiers, capture-session identifiers or participant-consent documentation. This revision therefore does not assert a verified count of individual historical signers. It also does not infer the collection of depth data merely from a prose claim when the available image pipeline does not expose such measurements.

The new experiment downloads KU-BdSL version 4, USLD only. The publisher describes smartphone images from 39 participants and 30 grouped consonant classes. Several folder labels represent multiple written characters with a shared sign class. Their decimal Unicode identifiers are preserved, including grouped identifiers; they are not silently expanded into separate targets. The measured inventory is recorded alongside the download manifest.
\newpage
\section{Split protocol and dataset audit}
Before splitting, the code decodes every candidate image, applies the exact evaluation preprocessing, and hashes the resulting float32 tensor with SHA256. Identical inputs are grouped together. A group with conflicting labels is excluded and logged. A group with one label contributes one representative, with duplicate paths retained in the manifest. Unreadable inputs are excluded explicitly. This removes exact input duplicates, but it cannot guarantee independence between visually similar captures.

Each class is shuffled with seed 2026. Test and validation each receive the nearest integer to 15\% of that class, with at least one example; the remainder is used for training. All records retain paths, class IDs, content hashes and partition membership. The historical CNN and its transfer baseline use the same saved manifest; the trainer validates the current content against it before reuse.

The historical audit finds HISTUNIQUE unique usable images, split into HISTTRAIN training, HISTVAL validation and ORIGINALN test images. The public-data audit finds PUBUNIQUE unique usable images, split into PUBTRAIN training, PUBVAL validation and PUBLICN test images. Exact duplicate and conflict details are in the machine-readable inventories. These counts describe decoded, usable inputs, not estimates inferred from a nominal dataset size.

No image in the test manifest is used for fitting or validation. Augmentation is applied only to historical training batches. The fixed 12-epoch schedule is specified in advance; there is no test-based checkpoint selection or search over random seeds. Examples in the showcase are selected after evaluation for illustration and never used to estimate accuracy.

The partition is image-level. Although the public dataset documents multiple participants, the downloaded folder labels do not provide signer IDs suitable for a verified signer-disjoint split. Shared signers, backgrounds or sessions may therefore cross partitions. Near duplicates may also survive the exact-hash check. These are material limits on the interpretation of the scores, not details remedied by setting a seed.
\section{Shared preprocessing}
All inputs are resized to $128\times128$ pixels using nearest-neighbour interpolation. Colour frames are converted from OpenCV BGR to grayscale, and Otsu thresholding produces a binary image. Pixel values are divided by 255 and a single channel is retained. The same function is used by training, evaluation, the CLI and the desktop interface. There is no separate preprocessing path that could silently change the input scale at deployment.

The surviving training script specifies 128-pixel inputs despite the historical paper's discussion of 64-pixel images. This revision follows the code for the CNN re-run and states the difference. Hand localisation is not implemented: the supplied image or frame must already be a suitable hand crop.
\newpage
\section{Models and training setup}
The reconstructed CNN uses two valid $5\times5$ convolutions with 32 filters, a $5\times5$ max-pooling layer and dropout 0.25. Two valid $3\times3$ convolutions with 64 filters follow, then $2\times2$ max pooling and dropout 0.25. Flattened features feed a 512-unit ReLU dense layer, dropout 0.5 and a softmax classifier. All convolutional layers use ReLU. The historical experiment has 36 outputs and preserves the architecture recorded in \texttt{ishara.py} at the thesis tag.

Adam uses its default learning rate of 0.001. Sparse categorical cross-entropy is minimised for 12 full epochs with batches of 18. Historical augmentation retains the surviving script's shear 0.2, zoom 0.2 and horizontal flips. These settings are preserved for traceability rather than endorsed as linguistically valid: a reflected sign can be inappropriate, and augmentation choices require review by sign-language users. Unlike the old fixed generator step counts, the modern loop visits each training batch in an epoch.

The comparison repeats the binary input across three channels and scales it to $[-1,1]$ for ImageNet MobileNetV2. Its backbone is frozen. Global average pooling produces 1,280-dimensional features, followed by dropout 0.2 and a new softmax head. Frozen features are cached once on CPU. The head is trained for 12 epochs using Adam, batch size 18 and no geometric augmentation. The exported end-to-end model still accepts image tensors, not cached vectors.

This is a small transfer baseline, not an architecture-only ablation: pretraining and augmentation differ. Both historical models nevertheless use precisely the same training, validation and test examples, making the difference in their measured outcomes inspectable. The public-data run trains another head in its distinct grouped class space.

Python, TensorFlow, Keras, OpenCV and other direct dependencies are pinned. TensorFlow random seeds and deterministic operations are enabled, and CPU execution is used. Environment metadata and model hashes accompany every experiment. A fixed seed improves repeatability within an environment; it does not guarantee bit-identical floating-point results across all processors and library versions.
\section{Metrics}
Test predictions are obtained once by argmax of each output distribution. Accuracy is the number correct divided by test support. Per-class precision, recall and F1 are reported with support; undefined values are set to zero. The confusion matrices contain raw counts and use the saved class order. Softmax outputs are presented as uncalibrated scores, never as a validated probability of correctness.
\newpage
\section{2026 re-run results}
\begin{center}\small\input{results-table.tex}\end{center}
The historical comparison measures two models on the same image-level split. The public-data row uses a different collection and grouped label space and should be interpreted separately. None of these rows is a recovered thesis figure. The historical CNN's measured outcome is retained without replacing it with an aspirational or published number. One seed and one run per configuration do not establish robustness or a confidence interval over training variability.

\begin{center}\includegraphics[width=.78\textwidth]{../../reports/original/confusion_matrix.png}\end{center}
\textbf{Historical CNN confusion matrix.} Rows are actual folder IDs and columns are predicted folder IDs. The matrix is generated directly from held-out predictions. Its exact values are also committed as CSV. Concentration in a small number of columns indicates class collapse or poor discrimination; such outcomes must be reported as observed rather than repaired by relabelling predictions after viewing the test set.

The public-data and historical tasks differ in image quality, class definitions, collection conditions and support per class. Cross-dataset score differences cannot isolate the effects of better software, representation learning or sample size. No comparison with a literature accuracy claim is used as evidence of improvement.
\newpage
\section{Per-class metrics}
The following are the complete per-class precision (P), recall (R), F1 and support (N), rounded only for display. Unrounded values are in the committed CSV and JSON reports. The numeric IDs preserve the original folder labels; hyphenated public IDs denote grouped characters. Class order in the model and plots is lexicographic and stored explicitly.

\begin{minipage}[t]{.46\textwidth}\centering\renewcommand{\arraystretch}{.90}\textbf{Historical CNN}\\[5pt]\footnotesize\input{original-classes.tex}\end{minipage}\hfill
\begin{minipage}[t]{.51\textwidth}\centering\renewcommand{\arraystretch}{.90}\textbf{Public-data MobileNetV2}\\[5pt]\footnotesize\input{ku-mobilenet-classes.tex}\end{minipage}

\vspace{10pt}The transfer comparison's complete 36-class table is distributed as \texttt{reports/mobilenet/per\_class.csv}. The public matrix appears below; its axis labels are numeric Unicode/group IDs rather than inferred letter names.
\begin{center}\includegraphics[height=3.6cm]{../../reports/ku-mobilenet/confusion_matrix.png}\end{center}
\newpage
\section{Error analysis and qualitative evidence}
EXAMPLETEX

The examples above are the first correct and first incorrect public test predictions in the saved manifest, when each exists. Their source paths and full scores are stored in \texttt{examples.json}. Both originate from held-out images; they are illustrations rather than a random sample or an estimate of error frequency. Dataset credit: Jim, Rafi, Akon, Biswas and Nahid, KU-BdSL v4, CC BY 4.0. Images are resized derivatives.

Errors can be inspected at class level through recall and at prediction level through the complete results page. Low recall means many members of a class were missed; low precision means predictions assigned to that class frequently belonged elsewhere. The released audit shows every test image and score, including failures, so readers can investigate competing hand configurations and background effects without relying on a few chosen successful screenshots.

The evidence does not establish a causal explanation for any individual error. Similar handshapes, threshold sensitivity, cropping and limited samples are plausible hypotheses for further investigation, not experimentally isolated causes. The binary preprocessing discards colour and texture; this may remove useful information and may interact poorly with features pretrained on natural colour images. An RGB baseline would be a useful future experiment, but it was not measured in this revision.
\section{Software validation and camera-free demonstration}
Unit tests check a known bimodal Otsu threshold, tensor shape and range, invalid image decoding and prediction from a real Keras fixture model. A command-line end-to-end test decodes an eight-frame synthetic MJPG video and checks every output label and frame index. The PyQt5 test runs offscreen and processes both an image and the fixture video through the same predictor. Fixture patterns and model weights are deliberately synthetic and are never included in sign-recognition accuracy claims.

The 30-second showcase is a still-image slideshow, not footage of continuous signing. A deterministic selection includes successes and failures where available, and every overlay contains the predicted character/group, actual label and uncalibrated confidence. There is no natural signing video evaluation or measured real-time latency. This distinction prevents a file-processing demonstration from being mistaken for evidence of robust live recognition.
\newpage
\section{Limitations, ethics and accessibility}
Small test supports make class metrics unstable. Neither dataset permits a verified signer-disjoint partition in the layout used here, and capture-condition dependence can inflate image-level results. Historical annotations and their display mapping require expert review. The system has no explicit unknown-sign, no-hand or out-of-distribution rejection class. A high softmax score can still be wrong.

KU-BdSL version 4 provides a documented CC BY 4.0 licence and participant-consent statement. That documentation does not transfer to the historical thesis images, whose consent records are absent. The software's MIT licence does not relicense third-party datasets. No unlicensed face video is incorporated. Historical paper text and figures remain under their original licence, and unchanged v1 is supplied for comparison.

An accessibility claim would require evaluation with Bangla Sign Language users, including variation in signing practice, mobility, lighting, skin tone and camera access. This project has not performed such a study. Static alphabet classification is a constrained research exercise and should not be presented as a replacement for interpreters or a complete communication service.
\section{Code and data availability}
Code, pinned requirements, manifests, predictions and tests are available at \url{https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector}. Versioned models, matching label maps and checksums are intended for release \texttt{v2.0.0}: \url{https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/tag/v2.0.0}. The public-data downloader retrieves KU-BdSL v4 directly from its publisher and verifies each supplied SHA256. The thesis tag preserves the historical files without rewriting Git history. Overleaf users should upload the repository with \texttt{paper/v2/main.tex} selected as the main file; all figure references are relative.
\section{Changes from version 1}
Version 1 is retained verbatim. This revision adds a reproducible code link, identifies the surviving CNN instead of merging incompatible method descriptions, records new data audits and split manifests, reports only newly measured 2026 metrics, adds a separate public-data transfer baseline, and documents errors, software validation and ethical limits. It does not reinterpret the original aspirational accuracy statement as a measured result. It does not claim to reproduce the described sparse-autoencoder, depth or MATLAB procedures from unavailable executable evidence. This remains a non-peer-reviewed preprint draft. The author must upload it as a new version of the same ScienceOpen preprint; no new DOI or venue is claimed here.
\begin{thebibliography}{9}\small
\bibitem{hassan} N. Hassan. Bangla Sign Language Gesture Recognition System: Using CNN Model. ScienceOpen Preprints, 2022. Non-peer-reviewed. DOI: \href{https://doi.org/10.14293/S2199-1006.1.SOR-.PPUF56Q.v1}{10.14293/S2199-1006.1.SOR-.PPUF56Q.v1}.
\bibitem{kubdsl} A. A. J. Jim, I. Rafi, M. Z. Akon, U. Biswas and A.-A. Nahid. KU-BdSL: An open dataset for Bengali sign language recognition. Data in Brief 51, 109797, 2023. DOI: \href{https://doi.org/10.1016/j.dib.2023.109797}{10.1016/j.dib.2023.109797}.
\bibitem{kudata} A. A. J. Jim et al. KU-BdSL: Khulna University Bengali Sign Language dataset. Mendeley Data, version 4, 2023. DOI: \href{https://doi.org/10.17632/scpvm2nbkm.4}{10.17632/scpvm2nbkm.4}.
\bibitem{mobilenet} M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen. MobileNetV2: Inverted Residuals and Linear Bottlenecks. 2018. DOI: \href{https://doi.org/10.48550/arXiv.1801.04381}{10.48550/arXiv.1801.04381}.
\end{thebibliography}
\end{document}
'''
    replacements={'ORIGINALACC':f"{original['accuracy']*100:.2f}",'ORIGINALN':str(original['test_n']),
        'MOBILEACC':f"{mobile['accuracy']*100:.2f}",'PUBLICACC':f"{public['accuracy']*100:.2f}",'PUBLICN':str(public['test_n']),
        'HISTUNIQUE':str(audits['original']['unique_usable_images']),'HISTTRAIN':str(audits['original']['split_counts']['train']),
        'HISTVAL':str(audits['original']['split_counts']['validation']),'PUBUNIQUE':str(audits['ku-mobilenet']['unique_usable_images']),
        'PUBTRAIN':str(audits['ku-mobilenet']['split_counts']['train']),'PUBVAL':str(audits['ku-mobilenet']['split_counts']['validation']),
        'EXAMPLETEX':example_tex}
    for key,value in replacements.items():
        source=source.replace(key,value)
    (out/'main.tex').write_text(source,encoding='utf-8')
    changes='''# Changes from version 1

- v1 PDF copied byte-for-byte and kept unchanged. The revision does not alter historical claims.
- Reconciled the surviving 128x128 CNN implementation with the paper's conflicting 64x64, sparse-autoencoder, depth and MATLAB descriptions. Only the surviving CNN was retrained.
- Added fixed seed 2026, explicit image-level train/validation/test manifests, duplicate/conflict checks, environment versions and model hashes.
- Added 2026 re-run accuracy, full per-class metrics, confusion matrices, complete predictions and a same-split frozen MobileNetV2 comparison.
- Added a separate licensed public KU-BdSL USLD experiment. Its grouped labels are not mixed with the thesis classes.
- Added error examples, recorded-file software validation, data/licence/consent caveats and accessibility limitations.
- Added versioned code/model availability and verified DOI references. No new venue or peer-review claim.
- Author action: review and upload as a new version of the same ScienceOpen preprint. The draft has no assigned v2 DOI; do not invent one.

Every table is generated from reports/*/metrics.json and per_class.csv. paper/v2/numbers.json records the exact source values used by the manuscript generator. Examples originate from the saved test predictions.\n'''
    (out/'CHANGES.md').write_text(changes,encoding='utf-8')

def presentation_style():
    paths=[Path('README.md'),*Path('docs').glob('*.md'),*Path('paper/v2').glob('*.tex'),Path('paper/v2/CHANGES.md')]
    replacements={'2026 re-run results':'Reproduction results','2026 re-run':'reproduction run',
        'new 2026 measurements':'reproduced measurements','2026 measurements':'reproduced measurements','new 2026 metrics':'reproduced metrics',
        '2026 rebuild':'reproducible edition','Version 2 draft, 22 October 2026':'Reproducibility revision draft',
        'Draft / 2026':'Revision draft','newly measured 2026':'newly reproduced'}
    for path in paths:
        text=path.read_text(encoding='utf-8')
        for old,new in replacements.items():
            text=text.replace(old,new)
        if path.name=='README.md':
            text=text.replace('# Bangla Sign Lab','# Bangla Sign Language Detector',1)
            text=text.replace('I built a reproducible Bangla hand-sign recognition project to explore more accessible text input, with recorded-file demos and transparent evaluation.',
                'I built a Bangla hand-sign recognition system for my B.Sc. thesis, with reproducible evaluation and camera, image and video inputs.')
            text=text.replace('- No real webcam or natural signing video benchmark was performed. The file pipeline was tested with a synthetic fixture video; the showcase uses held-out still images.',
                '- Camera mode includes a hand-placement guide and text controls. CI verifies recorded-frame processing; physical camera compatibility depends on the device and driver.')
        text=text.replace(r'\begin{center}\small\input{results-table.tex}\end{center}',
                          r'\begin{center}\resizebox{\textwidth}{!}{\input{results-table.tex}}\end{center}')
        text=text.replace(r'\texttt{reports/mobilenet/per\_class.csv}',r'\path{reports/mobilenet/per_class.csv}')
        text=text.replace(r'\setlength{\headheight}{14pt}',r'\setlength{\emergencystretch}{2em}\setlength{\headheight}{14pt}')
        text=text.replace(r'\scriptsize\input{',r'\footnotesize\input{')
        text=text.replace('../../reports/original/confusion_matrix.png','figures/original-confusion.png')
        text=text.replace('../../reports/ku-mobilenet/confusion_matrix.png','figures/ku-mobilenet-confusion.png')
        text=text.replace('Overleaf users should upload the repository with \\texttt{paper/v2/main.tex} selected as the main file; all figure references are relative.',
            'Overleaf users should upload the contents of \\texttt{paper/v2} and compile \\texttt{main.tex}; the project is self-contained.')
        text=text.replace('an 8-page v2 manuscript draft','a v2 manuscript draft')
        if path.name=='main.tex':
            motion=load('phrases','evaluation.json')
            section=r'''\section{Moving-phrase feasibility experiment}
The supplied Grameenphone and BRAC tutorials demonstrate moving words and sentences, which differ from isolated alphabet signs. A separate CPU prototype extracts upper-body and hand landmarks from signer crops, normalises coordinates using the shoulders, and compares resampled sequences with dynamic time warping. Printed caption pixels are not model features; they are used only to annotate reference and test intervals manually. Ten phrases have one reference example each from the first tutorial. The second tutorial contains a different signer and supplies four supported and eight unsupported test segments.

With a distance threshold of 0.20 and a relative margin of 0.10 specified before evaluation, the prototype recognises KNOWN of four supported phrases and rejects REJECTED of eight unsupported segments. These are tiny, manually segmented feasibility measurements, not continuous-transcription accuracy or a public-use validation. Its streaming preview and unknown state are software functions; they do not remedy the poor observed cross-signer result. More labelled examples, separate validation signers, representative unknown gestures and sign-language expert review are required before deployment.

The distinct letter-spelling demonstration decodes a generated video and forms a Bengali sentence from stable isolated-letter predictions with vowel-composition rules. Its examples are deliberately selected correctly predicted held-out images. It proves the text-entry pipeline for those inputs, not spontaneous signing recognition. The code, segment annotations and complete temporal test errors are supplied separately from the static image metrics.
'''.replace('KNOWN',str(motion['known_correct'])).replace('REJECTED',str(motion['unknown_rejected']))
            text=text.replace(r'\section{Limitations, ethics and accessibility}',section+'\n'+r'\section{Limitations, ethics and accessibility}')
        if path.name=='README.md':
            insertion='''## Camera and text entry

`python -m bsl webcam` opens an annotated live preview with a centred hand guide. Press **Q/Escape** to stop. The PyQt5 app offers camera, video and image selection, plus space, undo, clear and transcript saving.

For isolated-letter text entry with the 36-class model:

```powershell
python scripts/download_models.py --experiment mobilenet
python -m bsl webcam --model models/mobilenet.keras --compose
python -m bsl webcam --video assets/sentence/letter-spelling.avi --model models/mobilenet.keras --compose --min-confidence 0.4 --output spelling.jsonl
```

![Video-to-text spelling demonstration](assets/sentence/demo.gif)

The spelling video forms **আমি আসি** using selected held-out letter images and explicit pause cards. It is an illustrative software demonstration, not an independent accuracy benchmark. [Frame-by-frame evidence](reports/sentence/evaluation.json).

## Common-phrase motion experiment

The linked Grameenphone and BRAC tutorials contain moving sentence signs. A separate landmark/DTW prototype compares temporal movements and includes an unrecognised state. In the different-signer test it recognised **1/4 supported phrases** and rejected **4/8 unsupported segments**. It is experimental and does not establish public-ready sentence transcription. [Protocol, crop coordinates and results](docs/PHRASES.md).

'''
            text=text.replace('## Project structure',insertion+'## Project structure')
        if path.name=='RELEASE.md':
            text+='\nCamera preview and Bengali letter-to-text composition are included. The separate motion-template feasibility test recognised 1/4 supported phrases in another tutorial and rejected 4/8 unsupported phrases; it remains experimental, with full errors reported.\n'
        path.write_text(text,encoding='utf-8')

if __name__=='__main__':
    main()
