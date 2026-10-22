"""Publish verified local artifacts using Git's existing credential manager."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import requests

REPO='iamnajib71/Real-Time-Bangla-Sign-Language-Detector'
API='https://api.github.com/repos/'+REPO

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='v2.0.1')
    parser.add_argument('--draft',action='store_true',help='Prepare assets without publishing')
    args=parser.parse_args()
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                          capture_output=True,text=True,check=True)
    credential=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    client=requests.Session()
    client.headers.update(Authorization='Bearer '+credential['password'],Accept='application/vnd.github+json')
    def call(method,url,**kwargs):
        response=client.request(method,url,timeout=180,**kwargs)
        response.raise_for_status()
        return response.json() if response.content else None
    call('PATCH',API,json={'description':'Reproducible Bangla sign recognition with Python, TensorFlow and OpenCV; video demos and measured results.'})
    call('PUT',API+'/topics',json={'names':['computer-vision','deep-learning','tensorflow','opencv','sign-language','bangla','accessibility','cnn']})
    response=client.get(API+'/releases/tags/'+args.tag,timeout=30)
    if response.status_code==404:
        release=call('POST',API+'/releases',json={'tag_name':args.tag,'name':'Bangla Sign Language Detector '+args.tag+' - reproducible edition',
            'target_commitish':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'body':Path('docs/RELEASE.md').read_text(encoding='utf-8'),'draft':True,'prerelease':False})
    else:
        response.raise_for_status(); release=response.json()
    files=[]
    for experiment in ['original','mobilenet','ku-mobilenet']:
        files += [Path('models')/(experiment+'.keras'),Path('models')/(experiment+'.labels.json')]
    files += [Path('models/legacy-model1.h5'),Path('paper/v2/main.pdf'),
              Path('models/sentences-background.keras'),Path('models/sentences-background.json'),
              Path('tmp/overleaf-source.zip')]
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checksum_file=Path('models/SHA256SUMS.txt')
    checksum_file.write_text(''.join(f'{v}  {k}\n' for k,v in checksums.items()),encoding='utf-8')
    files.append(checksum_file)
    uploaded={x['name']:x for x in call('GET',release['assets_url'])}
    for path in files:
        if path.name in uploaded:
            if uploaded[path.name]['size']!=path.stat().st_size:
                raise ValueError('Existing asset has unexpected size: '+path.name)
            digest=uploaded[path.name].get('digest')
            if digest and digest!='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Existing asset checksum differs: '+path.name)
            continue
        upload=release['upload_url'].split('{')[0]
        with path.open('rb') as stream:
            call('POST',upload,params={'name':path.name},headers={'Content-Type':'application/octet-stream'},data=stream)
        print('Uploaded '+path.name,flush=True)
    if args.draft:
        print('Draft assets prepared: '+release['html_url'],flush=True)
        return
    release=call('PATCH',API+'/releases/'+str(release['id']),json={'draft':False})
    print('Published '+release['html_url'],flush=True)

if __name__=='__main__':
    main()
