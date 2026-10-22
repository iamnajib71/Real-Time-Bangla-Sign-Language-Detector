# The user problem and app workflow

A signer joins an online call with someone who does not know Bangla Sign Language. The intended outcome is a readable conversation, not a succession of unrelated class predictions.

The desktop app supplies a local camera/video preview beside an editable transcript. A separate always-on-top caption window can be shared through a meeting app's existing window-sharing feature. It does not inject captions into Zoom, Meet or Teams automatically. Video and reference clips stay on the local machine; nothing is sent to a recognition API.

## Use the working interface

Run `python -m bsl gui`, then select **Start camera** or **Open video**. Letter mode uses the same predictor as the CLI, with a placement guide for the camera. Choose **Load model** for another model and its matching label map. Use the 36-class model for the supplied letter-spelling sentence demo.

The transcript supports direct edits, space, undo, clear, copy and UTF-8 export. **Caption window** opens a readable companion window; share that window with the other caller if desired. Sharing is a deliberate action in the meeting application, outside this app.

## Teach personal phrases

Install the optional motion dependencies and official tracking assets. Select **Teach phrase from clip**, choose a short permitted recording containing one complete phrase, and type its meaning. Repeat with independent clips. The local template bank is ignored by Git. Switch to **Personal phrases (experimental)** and test with another recording or camera.

The app displays suggestions and requires **Confirm phrase** before appending them to the transcript. This prevents an unvalidated experimental match from silently becoming a conversation caption. A single saved example is supported in software, but does not establish reliable recognition; separate recordings and signers are still required for evaluation. The user can always type a correction.

## Recognition evidence and remaining gap

The public static-image experiment scored 239/240 correct. It is not a measure of moving-phrase transcription. The initial different-signer tutorial experiment recognised 1/4 supported phrases and rejected 4/8 unsupported segments. A subsequent feature/rejection search explicitly reclassified those two videos as development data; its scores are not a new blind test. The evidence is retained under reports/phrases and reports/phrases-v2.

Reliable general-call transcription remains unsolved here. Required next evidence includes several labelled examples per phrase, independent validation/test signers, realistic unknown inputs, automatic boundary evaluation, false-caption rates and end-to-end latency. English caption translation can follow after reliable Bangla recognition. Another sign language, such as ASL, requires its own recognition data/model.

Interface reference: [Rylo Translate](https://rylo.com/sign/translate/). The design and code here are independent; Rylo's public page states that its latest sign models are in research preview. No Rylo model or service is integrated.


Bounded sentence experiment: **6/10 supported sentences correct; 8/10 background clips rejected; 10 accepted captions with 60% precision** on dataset signer ID U19; background types occur in training. The prior baseline outcome was observed, so this is exploratory. [Protocol and vocabulary](SENTENCES.md). Neural weights/configuration are included as optional release assets.
