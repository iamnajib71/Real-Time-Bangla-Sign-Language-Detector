# Bounded sentence recognition

The call companion is a usable local camera/video interface. This recognizer is a separate ten-sentence experiment, not unrestricted conversation translation.

## Measured outcome

Test signer ID U19: **6/10 supported sentences correct; 8/10 background clips rejected; 10 accepted captions with 60% precision**. Two unsupported clips were falsely accepted. Read [all predictions](../reports/sentences-background/test-results.json), [per-class metrics](../reports/sentences-background/per_class.csv), and [confusion matrix](../reports/sentences-background/confusion_matrix.png). Softmax scores are uncalibrated, not probabilities of correct translation.

The original tutorial baseline remains 1/4 supported and 4/8 unsupported rejected. An initial conservative sentence baseline rejected every U19 clip (0/10 supported correct). The background-trained model is a follow-up experiment on a different vocabulary and dataset; these are not directly comparable accuracy rows. U19 was excluded from training and validation, but its earlier baseline outcome had already been observed. A fresh independent test is still needed.

## Data and protocol

[Ban-Sign-Sent-9K-V1](https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1), source revision `0e60ad0`, supplies sentence annotations and five encoded user IDs. Ten supported sentences were selected for text diversity before recognition outcomes, with ten additional background sentence types. A byte-range ZIP reader retrieved only the selected 100 clips and verified CRC and SHA256; it did not download the full archive. The inspected card has no explicit licence. No raw footage, face frames, or recorded landmark template banks are redistributed. Dataset identities and consent were not independently verified; obtain appropriate permission before redistribution or deployment. Neural weights and evaluation records are separate from source footage.

U4/U5/U6: 60 training clips (30 supported, 30 background). U7: 20 validation clips. U19: 20 test clips. There is only one clip per sentence per signer ID. The same background sentence TYPES occur in training, validation and test; rejection measures signer transfer on these types, not arbitrary unseen utterances. Class selection is narrow and food/family focused.

MediaPipe pose/hands at 8 sampled frames/second; shoulder-normalized coordinates; hand reassociation; interpolation; motion/shape features; rest trimming; 32 time steps of 104 features. GRU(64), dropout 0.3, Dense(32, ReLU), 11-way softmax including background; Adam 0.001; 80 epochs; batch 8; four temporal-warp/noise augmentations per original clip; seed 2026. Validation alone selected epoch 78, backend `gru` and score threshold 0.2. DTW was also considered on validation. Test tracking failures remain in the denominator.

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
| 23511 | ফল খাও। |
| 12419 | রাতে খেয়েছেন? |
| 25801 | মা চা খান না। |
| 28559 | খালু সকালে কফি খাবে। |
| 29451 | চাচী বিকালে শাক-সব্জী খেয়েছিল। |
| 28390 | দাদী দুপুরে কফি খেতেন। |
| 28908 | বাবা রাতে ডাল খাবে না। |
| 28341 | দাদা বিকালে কফি খায়নি। |
| 28754 | চাচা সকালে মাংস খেতেন না। |
| 28940 | মা দুপুরে ডাল খায়নি। |

## Reproduce

Run `scripts/inspect_sentence_dataset.py`, `scripts/fetch_sentence_subset.py --include-background-training`, then `scripts/train_sentence_model.py --background` in a new output directory. The training script refuses to overwrite an existing final test report. `scripts/summarize_sentences.py` creates class metrics and a confusion matrix without fitting anything. `scripts/verify_sentence_export.py` re-decodes one accepted and one rejected test clip through the exported model; this is an implementation consistency check, not a new accuracy estimate.
