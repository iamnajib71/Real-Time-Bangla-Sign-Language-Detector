"""Fetch a fixed, diverse sentence subset from a public multi-part ZIP by byte range."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
import zlib

BASE='https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1/resolve/main/'
parser=argparse.ArgumentParser()
parser.add_argument('--include-background-training',action='store_true')
args=parser.parse_args()
root=Path('data/external/ban-sign-sent')
rows=json.loads((root/'annotations.json').read_text(encoding='utf-8'))[1:]
index=json.loads((root/'archive-index.json').read_text())
by_id={r[0]:r for r in rows}
# Selection uses text only, before inspecting any recognition outcome.
selected=[by_id['23511'],by_id['12419']]
def terms(row): return set(re.findall(r'[^\s।?]+',row[2]))
def distance(a,b):
    x,y=terms(a),terms(b)
    return 1-len(x&y)/max(1,len(x|y))
while len(selected)<10:
    remaining=[r for r in rows if r not in selected]
    chosen=max(remaining,key=lambda r:(min(distance(r,s) for s in selected),-len(r[1]),r[0]))
    selected.append(chosen)
# Additional fixed sentences are held out as out-of-vocabulary inputs.
unknown=[]
for row in sorted(rows,key=lambda r:(len(r[1]),r[0])):
    if row not in selected and len(unknown)<10: unknown.append(row)
ids={r[0] for r in selected+unknown}
entries=[r for r in index if r['name'].endswith('.mp4') and r['name'].split('/')[1] in ids]
PART_SIZE=2097152000
def part_name(disk): return 'Sign_Videos.zip' if disk==9 else f'Sign_Videos.z{disk+1:02d}'
def get_range(disk,offset,length):
    result=bytearray()
    while length:
        boundary=1230871981 if disk==9 else PART_SIZE
        count=min(length,boundary-offset)
        url=BASE+part_name(disk)+f'?download=true&offset={offset}&count={count}'
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+count-1}'})
        with urllib.request.urlopen(request,timeout=180) as response:
            if response.status!=206:
                raise ValueError('Server ignored range; refusing full dataset download')
            block=response.read()
            if len(block)!=count: raise ValueError('Unexpected byte range size')
        result.extend(block); length-=count; disk+=1; offset=0
    return bytes(result)
def download(entry):
    sentence=entry['name'].split('/')[1]
    signer=re.search(r'_U(\d+)_',entry['name']).group(1)
    # Known sentences: U4/U5/U6 train, U7 validation, U19 final test.
    # Unknown sentences are used only for validation and test rejection checks.
    if sentence in {r[0] for r in unknown} and signer not in ['7','19'] and not args.include_background_training: return None
    target=root/'clips'/sentence/Path(entry['name']).name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        header=get_range(entry['disk'],entry['offset'],30)
        if header[:4]!=b'PK\x03\x04': raise ValueError('Bad ZIP local header; check part size')
        name_n,extra_n=struct.unpack_from('<2H',header,26)
        offset=entry['offset']+30+name_n+extra_n
        disk=entry['disk']
        if offset>=PART_SIZE and disk<9: offset-=PART_SIZE; disk+=1
        compressed=get_range(disk,offset,entry['compressed_size'])
        content=zlib.decompress(compressed,-15) if entry['method']==8 else compressed
        if len(content)!=entry['size'] or zlib.crc32(content)&0xffffffff!=entry['crc32']:
            raise ValueError('ZIP content checksum mismatch')
        target.write_bytes(content)
    split='train' if signer in ['4','5','6'] else 'validation' if signer=='7' else 'test'
    return {'path':target.as_posix(),'sentence_id':sentence,'signer':signer,'split':split,
        'known':sentence in {r[0] for r in selected},'text':by_id[sentence][1],'gloss':by_id[sentence][2],
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
manifest=[]
with ThreadPoolExecutor(max_workers=6) as pool:
    for n,item in enumerate(pool.map(download,entries),1):
        if item: manifest.append(item)
        if n%10==0: print(f'Inspected/downloaded {n}/{len(entries)} archive entries',flush=True)
report={'source':'https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1',
        'source_revision':'0e60ad0','license_statement':'No explicit licence in inspected dataset card; no source footage redistributed',
        'selection':'ten text-diverse sentences, chosen before recognition outcomes; ten additional unknown sentences',
        'selected':selected,'unknown':unknown,'records':manifest,
        'split':'U4/U5/U6 train, U7 validation, U19 final test; disjoint signer IDs'}
report['background_training']=args.include_background_training
out=Path('reports/sentences-background' if args.include_background_training else 'reports/sentences'); out.mkdir(parents=True,exist_ok=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'clips':len(manifest),'known_sentences':len(selected),'unknown_sentences':len(unknown)},ensure_ascii=True),flush=True)
