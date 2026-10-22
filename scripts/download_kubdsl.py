"""Download KU-BdSL v4 USLD from its publisher with per-file SHA256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE='https://data.mendeley.com'
API=BASE+'/public-api/datasets/scpvm2nbkm'

def session():
    s=requests.Session()
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    return s

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='data/external/ku-bdsl-usld')
    args=parser.parse_args()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    http=session()
    response=http.get(API+'/folders/4',timeout=60); response.raise_for_status(); folders=response.json()
    parent=next(x['id'] for x in folders if x['name']=='USLD')
    classes=sorted([x for x in folders if x.get('parent_id')==parent],key=lambda x:x['name'])
    manifest=[]
    for folder in classes:
        response=http.get(API+'/files',params={'folder_id':folder['id'],'version':4,'$start':0,'$limit':1000},timeout=60)
        response.raise_for_status(); files=response.json()
        if len(files)==1000:
            raise ValueError('Pagination required; refusing partial class download')
        for item in files:
            name=item['filename']
            if Path(name).name!=name or not name.lower().endswith(('.jpg','.png','.jpeg')):
                raise ValueError('Unexpected dataset filename')
            manifest.append({'path':folder['name']+'/'+name,'sha256':item['content_details']['sha256_hash'],
                             'url':item['content_details']['download_url']})
    def download(item):
        path=root/item['path']; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            return
        response=session().get(item['url'],timeout=120); response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest()!=item['sha256']:
            raise ValueError('Checksum mismatch: '+item['path'])
        path.write_bytes(response.content)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,_ in enumerate(pool.map(download,manifest),1):
            if n%100==0:
                print(f'Downloaded/verified {n}/{len(manifest)}',flush=True)
    report=Path('reports/data'); report.mkdir(parents=True,exist_ok=True)
    (report/'ku-bdsl-download.json').write_text(json.dumps({'doi':'10.17632/scpvm2nbkm.4','variant':'USLD',
        'license':'CC BY 4.0','classes':len(classes),'images':len(manifest),'files':manifest},indent=2),encoding='utf-8')
    print(f'Complete: {len(manifest)} images, {len(classes)} classes',flush=True)

if __name__=='__main__':
    main()
