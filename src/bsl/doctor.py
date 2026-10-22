"""Inspect installation and model integrity without importing TensorFlow or opening a camera."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

def inspect(root=None,motion=False):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    checks=[]
    def add(name,ok,detail): checks.append({'check':name,'ok':bool(ok),'detail':str(detail)})
    add('Python', (3,11)<=sys.version_info[:2]<(3,14),sys.version.split()[0])
    requirements=(root/'requirements.txt').read_text(encoding='utf-8').splitlines()
    if motion: requirements+=(root/'requirements-motion.txt').read_text(encoding='utf-8').splitlines()
    for line in dict.fromkeys(requirements):
        if '==' not in line or line.startswith(('#','-')): continue
        name,expected=line.strip().split('==',1)
        try: actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual='missing'
        add(name,actual==expected,f'installed {actual}; expected {expected}')
    metadata=json.loads((root/'reports/ku-mobilenet/model.json').read_text(encoding='utf-8'))
    models=root/'models'; model=models/metadata['file']; labels=model.with_suffix('.labels.json')
    if not model.is_file(): add('Static model',False,'Run python scripts/download_models.py')
    else:
        add('Static model SHA256',hashlib.sha256(model.read_bytes()).hexdigest()==metadata['sha256'],model.name)
        try:
            with zipfile.ZipFile(model) as archive:
                config=json.loads(archive.read('config.json'))
            # The measured exported model ends in the classifier softmax Dense.
            layers=config['config']['layers']
            outputs=next(layer['config']['units'] for layer in reversed(layers) if layer['class_name']=='Dense')
            actual=json.loads(labels.read_text(encoding='utf-8'))
            expected=[row['display'] for row in json.loads((root/'reports/ku-mobilenet/labels.json').read_text(encoding='utf-8'))]
            add('Static label contract',outputs==len(actual) and actual==expected,f'{outputs} outputs; {len(actual)} labels')
        except (OSError,ValueError,KeyError,StopIteration,zipfile.BadZipFile) as error:
            add('Static label contract',False,error)
    if motion:
        for name,info in json.loads((root/'reports/data/motion-models.json').read_text(encoding='utf-8')).items():
            path=models/name
            add(name,path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],'Run python scripts/download_motion_models.py if missing or corrupt')
        info=json.loads((root/'reports/sentences-background/test-results.json').read_text(encoding='utf-8'))
        path=models/'sentences-background.keras'
        add('Sentence model SHA256',path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==info['model_sha256'],'Run python scripts/download_sentence_model.py if missing or corrupt')
        config=models/'sentences-background.json'
        try:
            actual=json.loads(config.read_text(encoding='utf-8'))
            expected=json.loads((root/'reports/sentences-background/frozen-config.json').read_text(encoding='utf-8'))
            add('Sentence configuration',actual==expected,'Configuration matches frozen measured experiment')
        except (OSError,ValueError) as error: add('Sentence configuration',False,error)
    return {'ok':all(c['ok'] for c in checks),'checks':checks,
        'scope':'Installation and artifact integrity only; this does not certify recognition accuracy or camera compatibility'}

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',action='store_true'); parser.add_argument('--output')
    args=parser.parse_args(argv); result=inspect(motion=args.motion)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: Path(args.output).write_text(text+'\n',encoding='utf-8')
    else: print(text)
    return 0 if result['ok'] else 1

if __name__=='__main__': sys.exit(main())
