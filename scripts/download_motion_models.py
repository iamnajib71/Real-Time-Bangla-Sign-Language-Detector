"""Fetch official MediaPipe task models and validate their recorded content hashes."""
import hashlib
import json
from pathlib import Path
import requests

base=Path('models'); base.mkdir(exist_ok=True)
metadata=Path('reports/data/motion-models.json')
expected=json.loads(metadata.read_text()) if metadata.exists() else {}
urls={
 'pose_landmarker_lite.task':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
 'hand_landmarker.task':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'}
records={}
for name,url in urls.items():
    path=base/name
    if not path.exists():
        response=requests.get(url,timeout=180); response.raise_for_status(); path.write_bytes(response.content)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if name in expected and expected[name]['sha256']!=digest:
        raise ValueError('MediaPipe model changed; refusing unverified content: '+name)
    records[name]={'url':url,'sha256':digest,'source':'Google MediaPipe Tasks model distribution'}
metadata.parent.mkdir(parents=True,exist_ok=True)
metadata.write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Verified official motion tracking assets')
