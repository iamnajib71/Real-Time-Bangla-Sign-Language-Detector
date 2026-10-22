"""Download the bounded sentence model; raw tutorial/dataset videos are not bundled."""
import hashlib
import json
from pathlib import Path
import requests

base='https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/download/v2.0.0/'
prefix='sentences-background'
root=Path(__file__).resolve().parents[1]
expected=json.loads((root/'reports'/prefix/'test-results.json').read_text(encoding='utf-8'))['model_sha256']
models=root/'models'; models.mkdir(exist_ok=True)
for suffix in ['.keras','.json']:
    name=prefix+suffix
    response=requests.get(base+name,timeout=120); response.raise_for_status()
    if suffix=='.keras' and hashlib.sha256(response.content).hexdigest()!=expected:
        raise ValueError('Sentence model checksum mismatch')
    if suffix=='.json':
        frozen=json.loads((root/'reports'/prefix/'frozen-config.json').read_text(encoding='utf-8'))
        if json.loads(response.content)!=frozen: raise ValueError('Sentence configuration mismatch')
    (models/name).write_bytes(response.content)
    print('Verified '+name)
