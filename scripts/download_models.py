"""Download a model and its label map, checking the committed measured SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',default='ku-mobilenet',choices=['original','mobilenet','ku-mobilenet'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    metadata=json.loads((root/'reports'/args.experiment/'model.json').read_text())
    base='https://github.com/iamnajib71/Real-Time-Bangla-Sign-Language-Detector/releases/download/v2.0.0/'
    target=root/'models'; target.mkdir(exist_ok=True)
    for name in [metadata['file'],Path(metadata['file']).with_suffix('.labels.json').name]:
        response=requests.get(base+name,timeout=180); response.raise_for_status()
        if name==metadata['file'] and hashlib.sha256(response.content).hexdigest()!=metadata['sha256']:
            raise ValueError('Model checksum mismatch')
        # Label maps are separately compared with the measured report.
        if name.endswith('.labels.json'):
            expected=[x['display'] for x in json.loads((root/'reports'/args.experiment/'labels.json').read_text(encoding='utf-8'))]
            if response.json()!=expected:
                raise ValueError('Label map differs from measured experiment')
        (target/name).write_bytes(response.content)
        print('Verified '+name)

if __name__=='__main__':
    main()
