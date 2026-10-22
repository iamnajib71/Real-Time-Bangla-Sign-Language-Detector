# Common-phrase transcription experiment

The practical target is a bounded vocabulary of useful Bangla phrases, with rejection of unsupported input. The current implementation is an experimental feasibility baseline, not a general-public translator.

## Method

Two user-supplied tutorial videos were processed locally. The Grameenphone tutorial supplies ten single-example references. The BRAC tutorial supplies twelve manually segmented test examples from a different signer: four overlapping phrases and eight unsupported phrases. Segment labels come from the tutorials' printed text and require expert review; the recognizer consumes only hand and upper-body landmark coordinates, never caption or OCR pixels.

Signer crop coordinates are normalised and recorded in `data/phrases/tutorials.json`. MediaPipe tracks seven upper-body points and both sets of 21 hand points. Coordinates are normalised using shoulder centre and width, resampled to 32 time steps and compared with dynamic time warping. Missing joints are zero-filled. The fixed distance threshold is 0.20 with minimum relative margin 0.10; this was specified before the test. The displayed distance/similarity is not a calibrated probability.

## Measured result

- Supported phrase recognition: **1/4 (25%)**.
- Unsupported segment rejection: **4/8 (50%)**.
- This is a tiny manually segmented cross-video test; automatic stream segmentation and spontaneous conversation accuracy have not been established.

Every prediction and error is in `reports/phrases/evaluation.json`. This result is retained without tuning thresholds on the test tutorial or relabelling the videos to match predictions. The static image classifier's 99.58% result must never be described as sentence-transcription accuracy.

## Run locally

Keep permitted copies of the videos at `data/external/tutorials/grameenphone.mp4` and `data/external/tutorials/brac.mp4`, or adapt the annotation paths to your own labelled reference/test recordings. The videos stay outside Git. No claim of a Creative Commons licence is made for either tutorial; neither source footage nor derived templates are included in the release.

```powershell
python -m pip install -r requirements-motion.txt
python scripts/download_motion_models.py
python -m bsl.phrase build
python -m bsl.phrase live --video data/external/tutorials/brac.mp4 --roi 0.30 0 0.85 1 --no-preview --output phrase-events.jsonl
python -m bsl.phrase live --camera 0
```

The preview emits a phrase after repeated accepted matches and includes an unrecognised state. The streaming heuristic has no validated transcription accuracy. Q/Escape stops capture; C clears accumulated text. Repeated/connected sentence signs, no-hand rejection and dialect variation need further work.

## What would make it useful to the public

Collect several consented recordings per target phrase from multiple signers. Reserve entire signers and sessions for validation/test. Add realistic unsupported signs and idle gestures. Use a temporal model and thresholds selected only on validation data; report phrase errors, false accepts and latency on untouched signers. Evaluate with Bangla Sign Language users before making an accessibility effectiveness claim. Adding a text language model must not conceal visual recognition errors.

References: [Grameenphone tutorial](https://www.youtube.com/watch?v=z08yRPKmrHk), [BRAC tutorial](https://www.youtube.com/watch?v=k6bnLa4YluI), [official MediaPipe tasks documentation](https://ai.google.dev/edge/mediapipe/solutions/vision).
