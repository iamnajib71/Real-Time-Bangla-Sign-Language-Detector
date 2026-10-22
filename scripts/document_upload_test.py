"""Document actual local file-upload evidence, including errors rather than only successes."""
import json
from pathlib import Path

rows=json.loads(Path('reports/sentences-background/video-upload-test.json').read_text(encoding='utf-8'))
table='\n'.join('| '+r['case']+' | '+r['expected_caption']+' | '+r['suggested_caption']+' | '+str(r['decoded_frames'])+' |' for r in rows)
text='''# Video-upload integration test

This test invokes the desktop app's **Open video** handler. The operating-system chooser return is automated; the decoder, source-media clock, landmark tracker, exported GRU, suggestion state and confirmation action are real. It does not upload video to Rylo or another external service.

| Case | Source meaning | Actual suggestion | Decoded frames |
|---|---|---|---:|
'''+table+'''

The first two examples were chosen from already measured test predictions: one supported success and one background false accept. This verifies implementation consistency and exposes an error; it is not a new independent accuracy estimate. The optional third case is the user's supplied BRAC tutorial greeting, 0.8-5.8 seconds, with normalized signer crop (0.30, 0, 0.85, 1) to exclude the printed text. The ten-sentence model does not support this greeting: a safe result is **Unrecognised**, not an invented translation. No thresholds or weights were changed after inspecting these upload outcomes.

The supplied BRAC greeting was falsely mapped to **দাদা বিকালে কফি খায়নি।**, a supported coffee-related sentence, rather than rejected. This is a failed out-of-vocabulary stress case, not successful greeting transcription. No model or threshold was adjusted to make this test look better.

The transcript stayed empty before confirmation in every case. Confirmation appends the suggestion; it does not make a wrong suggestion correct. Raw evidence records both states, source hashes, frame counts and observed labels. Video pixels are deliberately omitted from the public screenshots; source footage remains local because no redistribution licence was established.

![Supported video upload](../assets/app/upload-supported-correct.png)

![False match reproduced through upload](../assets/app/upload-background-false-accept.png)

**Failed unsupported-greeting test:**

![BRAC greeting falsely matched to a supported sentence](../assets/app/upload-provided-brac-hello.png)

Run `python scripts/test_video_upload.py` with the optional motion dependencies, models and local source files installed. The script uses the exact source paths in the saved report. CI instead exercises a redistributable synthetic fixture through the same file-selection handler, and checks media-clock sampling and stale-state reset.

The integration check exposed playback-clock sampling in the old GUI. The fix samples recorded frames at 8 Hz using their media timestamps, resets tracking for each new file, and classifies the final completed buffer. Camera sampling still uses a monotonic live clock. Loss of tracking clears a pending suggestion, preventing a stale caption from being confirmed. This aligns the recorded-file path with the measured CLI preprocessing.
'''
Path('docs/VIDEO-UPLOAD-TEST.md').write_text(text,encoding='utf-8')
path=Path('README.md'); text=path.read_text(encoding='utf-8')
addition='''## Video-upload test and scientific figures

The actual **Open video** flow was checked with a supported held-out clip, a false-match clip, and the supplied BRAC greeting. The test records the observed suggestions and confirms that captions require user confirmation. [Upload test, screenshots and errors](docs/VIDEO-UPLOAD-TEST.md) · [Raw evidence](reports/sentences-background/video-upload-test.json).

The manuscript includes a pipeline diagram, unsmoothed validation curves, static and temporal confusion matrices, and correct/incorrect examples. [Training curves](paper/v2/figures/validation-curves.png) · [Sentence confusion matrix](reports/sentences-background/confusion_matrix.png).

For future commercial use, [the scientific validation plan](docs/SCIENTIFIC-ROADMAP.md) sets out Deaf-user involvement, data permissions, independent signer/session tests, novel unsupported inputs, continuous-signing errors and end-to-end latency. These are requirements still to be met, not claimed achievements.

'''
addition+='''## Run and test on Windows

From the repository folder, run `setup.bat` once to install the pinned dependencies and verified model assets. Then double-click `run-app.bat`. Use **Open video**, select a complete supported signing clip, and choose **Sentence vocabulary (experimental)** for the temporal model. Use **Supported phrases** to check its ten-sentence vocabulary. Suggestions require confirmation; typing and corrections remain available.

Run `python -m bsl doctor --motion` to verify pinned dependencies, model checksums, label order and tracking assets without importing TensorFlow or opening a camera. Run `run-tests.bat` for the camera-free automated suite. The local source-video integration check is `python scripts/test_video_upload.py`; it requires the local source clips described in the test report. Those third-party videos are not bundled with the repository.

'''
text=text.replace('## Project structure',addition+'## Project structure'); path.write_text(text,encoding='utf-8')
path=Path('paper/v2/main.tex'); text=path.read_text(encoding='utf-8')
para='\nThe desktop file-selection handler was additionally exercised with a real supported test clip and a background false accept; decoded suggestions matched the saved predictions, and both required confirmation before transcript insertion. A cropped user-supplied tutorial greeting was falsely mapped to a supported coffee-related sentence instead of rejected. This failed out-of-vocabulary stress input exposes a remaining recognition risk; no weights or thresholds were adjusted after inspecting it. The chooser return is automated, while decoding, landmarks and neural inference are real. Recorded-file sampling now follows media timestamps rather than wall-clock playback speed, and tracking state resets between files. These checks verify software behaviour, not a new recognition benchmark.\n'
text=text.replace(r'\section{Moving-phrase feasibility experiment}',para+r'\section{Moving-phrase feasibility experiment}')
text=text.replace(r'\section{Limitations, ethics and accessibility}',r'\section{Limitations, ethics and accessibility}'+'\nCommercial use would require a Deaf-user-led task definition, resolved data permissions and consent, fresh signer/session-disjoint evaluation, novel unsupported utterances, continuous-signing boundary/error analysis and measured call latency. The current exploratory results do not establish these requirements. The repository records an explicit validation roadmap rather than a deployment-readiness claim.\n')
# Preserve the original release and its tag; this is an additive patch release.
text=text.replace('v2.0.0','v2.0.1'); path.write_text(text,encoding='utf-8')
for file in ['docs/RELEASE.md','paper/v2/CHANGES.md']:
    path=Path(file); text=path.read_text(encoding='utf-8')
    text+='\n- Patch release: verified the real video-file selection flow, added a media-clock sampling regression test, and added pipeline, validation-history and temporal confusion figures. The original v2.0.0 release/tag remains unchanged.\n'
    path.write_text(text,encoding='utf-8')
