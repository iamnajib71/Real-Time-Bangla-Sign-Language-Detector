"""Decode two real held-out clips again through the exported temporal predictor."""
import json
from pathlib import Path
from bsl.motion import extract_video
from bsl.sentence import SentenceRecognizer

root=Path('reports/sentences-background')
report=json.loads((root/'test-results.json').read_text(encoding='utf-8'))
recognizer=SentenceRecognizer()
checks=[]
for accepted in [True,False]:
    row=next(r for r in report['predictions'] if r['accepted']==accepted)
    _,features,valid=extract_video(row['path'],fps=8)
    result=recognizer(features[valid])
    assert result['label']==row['predicted'] and result['accepted']==row['accepted']
    checks.append({'path':row['path'],'expected':row['predicted'],**result,'tracked_frames':int(valid.sum())})
(root/'export-verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('Exported temporal model matched both accepted and rejected recorded clips')
