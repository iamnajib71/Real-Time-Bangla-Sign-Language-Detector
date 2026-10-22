# From research prototype to an evaluated communication product

The intended problem is a signed conversation between a Deaf signer and a hearing person who does not sign. Success means preserving the intended message with an acceptable delay and a useful recovery path when recognition fails. A high static-image score alone does not establish that outcome.

## Evidence available now

- Static KU-BdSL image classification: 239/240 correct, image-level split. This is a separate task from sentence translation.
- Exploratory ten-sentence test: 6/10 supported sentences correct; 8/10 background clips rejected; 6/10 accepted captions correct. There is one clip per supported sentence for one test signer ID, and background sentence types occur in training. A prior baseline outcome on that signer was observed.
- The upload integration test checks real decoding, landmarks, saved weights and confirmation. It includes a known failure; it is not a new accuracy benchmark.
- Live camera input and shareable caption controls exist. Continuous conversation accuracy, new-signer reliability and actual call latency have not been established.

These results support further research, not a claim that a public communication service is ready. Preserve the saved splits, weights and failures; improvement must be measured on new evidence.

## 1. Define a useful first task with Deaf users

Recruit paid Deaf Bangla Sign Language users and qualified sign-language specialists as design partners. Identify a narrowly useful conversation setting, its required vocabulary, regional variants and acceptable correction methods. Do not assume that alphabet spelling, Bengali literacy or a hearing person's confirmation replaces natural signing. Record task completion, message fidelity, correction effort and user preference. Include people with different signing styles, devices and access needs. Set acceptance criteria with these participants before testing.

## 2. Establish a commercially usable data foundation

Record the source, version, permitted uses and consent status of every dataset and clip. The inspected Ban-Sign-Sent-9K card does not state an explicit licence; public download availability is not proof of commercial permission. Resolve usage rights and participant consent before using it in a commercial product. Tutorial clips currently remain local analysis inputs. Code licensing does not establish rights to source footage or training data.

Collect multiple examples per phrase across independent signers, sessions, camera positions, lighting and backgrounds. Obtain explicit permission for recording, training, evaluation, retention and any redistribution. Avoid putting identifiable recordings in a public repository. Include natural signing, nonmanual cues and transitions, rather than only isolated demonstrations. Use expert-reviewed sentence/gloss annotations with uncertainty and disagreement retained.

## 3. Freeze a prospective evaluation protocol

Separate signers and capture sessions between training, validation and final test. Keep near-duplicate clips and alternate crops together. Define the supported label space, unsupported inputs, boundary annotations and metrics before model selection. Choose checkpoints, calibration and rejection thresholds using validation only. Reserve a fresh final test after development; do not reclassify a repeatedly inspected test set as untouched.

The current five encoded signer IDs and narrow vocabulary are insufficient for a population claim. Use grouped cross-validation to study signer variation during development, then confirm on separately collected signers and sessions. Report counts and uncertainty, not only a best run. Keep a ledger of which data each model or threshold decision has seen.

## 4. Evaluate the actual conversation task

For completed clips, report supported recall, accepted-caption precision, false accepts and rejection coverage, with per-class counts. Add genuinely novel unsupported sentence types; rejecting a background class seen in training does not establish unknown-language detection.

For continuous recordings, measure sentence/word error rate against expert annotations, missed signs, automatic boundary errors, repeated/inserted captions and false captions per minute of ordinary nonsigning activity. Include greetings, signer changes, pauses, partial hands, motion blur, occlusion and out-of-vocabulary signing. Measure end-to-end latency from sign completion to displayed caption on representative CPU devices. Distinguish cold startup, preprocessing, inference, buffering and rendering time. Offline throughput is not the same as call latency.

Use signer/session-clustered uncertainty estimates once there are enough independent participants. Inspect failures by signer and capture condition, not just aggregate accuracy. Assess calibration on validation data; a softmax score is not a validated probability of correct meaning.

## 5. Improve models against those failures

Keep the original CNN, frozen image baseline and small temporal model as reference points. Compare temporal landmark and RGB/video models on exactly the same prospective split, with capacity, compute and preprocessing disclosed. Measure whether hand/face/body cues improve unseen-signer performance through ablations. Use more training examples and appropriate language supervision before relying on a language model to rewrite outputs. Language fluency must not conceal recognition errors or invent a signer’s intent.

English text or speech output can follow reliable recognition, with translation quality tested separately. Another sign language needs its own linguistic/data evaluation; translating Bengali text does not produce an ASL recognizer.

## 6. Product validation before public claims

Keep explicit unknown states, editable captions, undo and manual confirmation available. Clear stale suggestions when tracking is lost. Make capture and sharing deliberate and visible; avoid recording or sending video by default. Keep the interface responsive during model loading and inference, with recoverable errors for missing models, permissions and devices.

Test the complete workflow with consenting participants in realistic calls. Evaluate intelligibility, correction burden, failure recovery and whether captions help the conversation. Compare against the participants' existing communication methods. Document failure conditions and provide an accessible fallback. A deployment decision should be based on this evidence and the criteria agreed with users, not a claim based on the 99.58% static-image result.

## Next research gates

| Gate | Required artifact | Current state |
|---|---|---|
| Useful task and vocabulary | Deaf-user study protocol and agreed acceptance criteria | Not collected |
| Data rights and consent | Versioned permissions/consent register | KU-BdSL attribution documented; sentence/tutorial rights unresolved for commercial use |
| Independent evaluation | Frozen signer/session manifests and unseen-utterance negatives | Small exploratory split only |
| Robust recognition | Full errors, calibration and rejection/coverage analysis | Initial clip-level results available |
| Continuous signing | Annotated sequences, boundary errors, insertion/deletion metrics | Not validated |
| Call experience | CPU latency distributions and consenting user-task evaluation | Local controls and file tests available |

These are research requirements, not completed achievements or advertised performance targets.
