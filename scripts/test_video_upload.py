"""Exercise the real Open video handler with accepted and false-accepted test clips."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
from bsl.gui import QApplication,StartingPage,QFileDialog
from bsl.predict import Predictor
from bsl.media import effective_fps
from bsl.motion import crop_signer
import cv2

root=Path('reports/sentences-background')
data=json.loads((root/'test-results.json').read_text(encoding='utf-8'))
app=QApplication.instance() or QApplication([])
results=[]; cases=[]
for kind in ['supported-correct','background-false-accept']:
    row=next(r for r in data['predictions'] if
        (r['known'] and r['correct'] if kind=='supported-correct' else not r['known'] and r['accepted']))
    cases.append((kind,row))
# Analyse the user-supplied greeting as an unsupported-vocabulary stress test.
source=Path('data/external/tutorials/brac.mp4')
if source.exists():
    cap=cv2.VideoCapture(str(source)); fps=effective_fps(cap.get(cv2.CAP_PROP_FPS))
    start,end=.8,5.8; cap.set(cv2.CAP_PROP_POS_FRAMES,round(start*fps))
    local=Path('tmp/upload-brac-hello.avi'); writer=None
    for _ in range(round((end-start)*fps)):
        ok,frame=cap.read()
        if not ok: break
        frame=crop_signer(frame,[.30,0,.85,1]); h,w=frame.shape[:2]
        if writer is None: writer=cv2.VideoWriter(str(local),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
        writer.write(frame)
    cap.release()
    if writer: writer.release()
    cases.append(('provided-brac-hello',{'path':str(local),'text':'হ্যালো','sha256':hashlib.sha256(local.read_bytes()).hexdigest(),
        'source_url':'https://www.youtube.com/watch?v=k6bnLa4YluI','segment_seconds':[start,end],'roi':[.30,0,.85,1]}))
for kind,row in cases:
    window=StartingPage(Predictor('models/ku-mobilenet.keras')); window.mode.setCurrentIndex(2)
    window.show(); app.processEvents()
    with patch.object(QFileDialog,'getOpenFileName',return_value=(row['path'],'')):
        window.select_video()
    window.timer.stop()
    while window.capture is not None:
        window.tick(); app.processEvents()
    predicted=window.pending_phrase or 'Unrecognised'
    if 'predicted' in row: assert predicted==row['predicted'],(predicted,row['predicted'])
    before=window.transcript.toPlainText()
    assert before==''
    window.confirm_phrase()
    after=window.transcript.toPlainText()
    assert after==(predicted if predicted!='Unrecognised' else '')
    results.append({'case':kind,'path':row['path'],'sha256':row['sha256'],
        'expected_caption':row['text'],'suggested_caption':predicted,
        'correct':row['correct'] if 'correct' in row else predicted=='Unrecognised',
        'source_url':row.get('source_url'),'segment_seconds':row.get('segment_seconds'),'roi':row.get('roi'),
        'interpretation':'unsupported greeting should be rejected' if kind=='provided-brac-hello' else 'saved test example',
        'decoded_frames':window.frame_index,'transcript_before_confirmation':before,
        'transcript_after_confirmation':after,'matches_saved_prediction':True if 'predicted' in row else None,
        'test_type':'real file selection handler, decoder, landmarks, exported GRU and confirm action; file chooser return automated'})
    # Public screenshot excludes all source video pixels; it is an app-state record.
    window.preview.clear(); window.preview.setText('Uploaded video processed locally\nSource footage omitted from this screenshot')
    window.status.setText('Verified upload test | '+kind)
    app.processEvents(); Path('assets/app').mkdir(exist_ok=True)
    window.grab().save('assets/app/upload-'+kind+'.png')
    window.close()
(root/'video-upload-test.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('Real Open video flow verified; see video-upload-test.json for captions, errors and the source-video stress test')
