"""Append measured temporal evidence and call workflow to generated documentation."""
import json
from pathlib import Path

root=Path('reports/sentences-background')
report=json.loads((root/'test-results.json').read_text(encoding='utf-8'))
config=json.loads((root/'frozen-config.json').read_text(encoding='utf-8'))
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
m=report['test']
evidence=f"{m['known_correct']}/{m['known_n']} supported sentences correct; {m['unknown_rejected']}/{m['unknown_n']} background clips rejected; {m['accepted_n']} accepted captions with {m['accepted_precision']:.0%} precision"
vocab='\n'.join('| '+row[0]+' | '+row[1]+' |' for row in manifest['selected'])
doc=f'''# Bounded sentence recognition

The call companion is a usable local camera/video interface. This recognizer is a separate ten-sentence experiment, not unrestricted conversation translation.

## Measured outcome

Test signer ID U19: **{evidence}**. Two unsupported clips were falsely accepted. Read [all predictions](../reports/sentences-background/test-results.json), [per-class metrics](../reports/sentences-background/per_class.csv), and [confusion matrix](../reports/sentences-background/confusion_matrix.png). Softmax scores are uncalibrated, not probabilities of correct translation.

The original tutorial baseline remains 1/4 supported and 4/8 unsupported rejected. An initial conservative sentence baseline rejected every U19 clip (0/10 supported correct). The background-trained model is a follow-up experiment on a different vocabulary and dataset; these are not directly comparable accuracy rows. U19 was excluded from training and validation, but its earlier baseline outcome had already been observed. A fresh independent test is still needed.

## Data and protocol

[Ban-Sign-Sent-9K-V1](https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1), source revision `{manifest['source_revision']}`, supplies sentence annotations and five encoded user IDs. Ten supported sentences were selected for text diversity before recognition outcomes, with ten additional background sentence types. A byte-range ZIP reader retrieved only the selected 100 clips and verified CRC and SHA256; it did not download the full archive. The inspected card has no explicit licence. No raw footage, face frames, or recorded landmark template banks are redistributed. Dataset identities and consent were not independently verified; obtain appropriate permission before redistribution or deployment. Neural weights and evaluation records are separate from source footage.

U4/U5/U6: 60 training clips (30 supported, 30 background). U7: 20 validation clips. U19: 20 test clips. There is only one clip per sentence per signer ID. The same background sentence TYPES occur in training, validation and test; rejection measures signer transfer on these types, not arbitrary unseen utterances. Class selection is narrow and food/family focused.

MediaPipe pose/hands at 8 sampled frames/second; shoulder-normalized coordinates; hand reassociation; interpolation; motion/shape features; rest trimming; 32 time steps of 104 features. GRU(64), dropout 0.3, Dense(32, ReLU), 11-way softmax including background; Adam 0.001; 80 epochs; batch 8; four temporal-warp/noise augmentations per original clip; seed 2026. Validation alone selected epoch {config['selected_epoch']}, backend `{config['backend']}` and score threshold {config['threshold']}. DTW was also considered on validation. Test tracking failures remain in the denominator.

## Run

```powershell
python -m pip install -r requirements-motion.txt
python scripts/download_motion_models.py
python scripts/download_sentence_model.py
python -m bsl.sentence --video your-complete-sentence.mp4 --output sentence.json
python -m bsl.sentence --camera 0
```

The video CLI classifies one completed, trimmed clip. The GUI offers rolling suggestions; confirm a suggestion before adding it to the transcript. Automatic sentence boundaries, unrestricted vocabulary, actual call latency and accessibility effectiveness have not been validated. A high-confidence wrong suggestion is possible. The caption window, typing/corrections and recorded-file inputs work independently of recognition quality.

## Vocabulary

| Dataset sentence ID | Caption |
|---|---|
{vocab}

## Reproduce

Run `scripts/inspect_sentence_dataset.py`, `scripts/fetch_sentence_subset.py --include-background-training`, then `scripts/train_sentence_model.py --background` in a new output directory. The training script refuses to overwrite an existing final test report. `scripts/summarize_sentences.py` creates class metrics and a confusion matrix without fitting anything. `scripts/verify_sentence_export.py` re-decodes one accepted and one rejected test clip through the exported model; this is an implementation consistency check, not a new accuracy estimate.
'''
Path('docs/SENTENCES.md').write_text(doc,encoding='utf-8')
readme=Path('README.md').read_text(encoding='utf-8')
section=f'''## Call companion and sentence vocabulary

![Desktop caption companion](assets/app/caption-companion.png)

Run `python -m bsl gui` for the camera/video preview, editable Bengali transcript and separate caption window. Share that window in your call app. Screenshot text is manually entered to illustrate the controls.

The optional temporal model supports ten complete sentences: **{evidence}** on a separate dataset signer ID. This small exploratory test does not establish general-call transcription. Suggestions require confirmation. [Setup, exact vocabulary and full errors](docs/SENTENCES.md) · [Call workflow](docs/CALL-COMPANION.md).

'''
readme=readme.replace('## Project structure',section+'## Project structure')
readme=readme.replace('- Static letter/group classification only; no continuous signing, words, grammar or facial-expression understanding.',
    '- The static classifier handles letters/groups. The separate sentence model has a fixed ten-sentence vocabulary; unrestricted signing, grammar and facial-expression understanding remain unvalidated.')
Path('README.md').write_text(readme,encoding='utf-8')
for file in ['docs/RESULTS.md','docs/RELEASE.md','docs/CALL-COMPANION.md']:
    path=Path(file); text=path.read_text(encoding='utf-8')
    text=text.split('\n\nBounded sentence experiment:')[0]
    text+='\n\nBounded sentence experiment: **'+evidence+'** on dataset signer ID U19; background types occur in training. The prior baseline outcome was observed, so this is exploratory. [Protocol and vocabulary](SENTENCES.md). Neural weights/configuration are included as optional release assets.\n'
    path.write_text(text,encoding='utf-8')
# Compact manuscript addition; every count comes directly from the frozen report.
paper=Path('paper/v2/main.tex'); text=paper.read_text(encoding='utf-8')
paragraph=r'''\subsection{Bounded sentence model}
A follow-up experiment uses 100 selected clips from the public Ban-Sign-Sent-9K-V1 repository (\url{https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1}). Ten text-diverse target sentences and ten background sentence types were selected before recognition outcomes. Encoded signer IDs U4/U5/U6 supply 60 training clips; U7 supplies 20 validation clips; U19 supplies 20 test clips. There is one clip per sentence and ID. Identity and consent were not independently verified, and the inspected dataset card has no explicit licence; source footage is not redistributed.

An 11-class GRU classifier (64 units, dropout 0.3, dense 32, softmax including background) uses 32-step motion/shape landmark sequences, Adam 0.001, batch 8, 80 epochs, temporal warp/noise augmentation and seed 2026. Validation selects the checkpoint and rejection threshold, with DTW as an alternative backend. The selected GRU obtains KC/10 supported captions correct and rejects UR/10 background clips on U19; AP of AN accepted captions are correct. The two false accepts illustrate the danger of confident captions. Background sentence types occur in training, so rejection does not establish novel-utterance detection. An earlier conservative baseline rejected all clips; its U19 outcome was already observed before this follow-up, which is exploratory rather than a wholly blind confirmation. Full clip-level predictions, class metrics and confusion counts accompany the code. Automatic sentence boundaries and general conversation translation remain unevaluated.

'''.replace('KC',str(m['known_correct'])).replace('UR',str(m['unknown_rejected'])).replace('AP',str(m['known_correct'])).replace('AN',str(m['accepted_n']))
text=text.replace(r'\section{Limitations, ethics and accessibility}',paragraph+r'\section{Limitations, ethics and accessibility}')
text=text.replace('There is no natural signing video evaluation or measured real-time latency.',
    'The static-image showcase does not measure continuous signing or real-time latency; a separate bounded completed-clip experiment is reported below.')
text=text.replace('The system has no explicit unknown-sign, no-hand or out-of-distribution rejection class.',
    'The static classifier has no explicit unknown-sign, no-hand or out-of-distribution rejection class; the separate temporal background class does not solve arbitrary unknown recognition.')
text=text.replace('are intended for release','are provided with release')
text=text.replace('The scope is isolated static classes rather than continuous language translation,',
    'A separate exploratory ten-sentence classifier recognises '+str(m['known_correct'])+' of ten supported test clips and rejects '+str(m['unknown_rejected'])+' of ten background clips. The work does not establish continuous language translation,')
paper.write_text(text,encoding='utf-8')
changes=Path('paper/v2/CHANGES.md'); changes.write_text(changes.read_text(encoding='utf-8')+'\n- Added the separate signer-ID-disjoint bounded sentence experiment, false accepts, test-reuse disclosure, call companion and optional temporal neural model. No general-translation claim.\n',encoding='utf-8')
