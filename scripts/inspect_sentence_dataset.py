"""Inspect public sentence annotations and archive index without fetching 20 GB."""
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

BASE='https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1/resolve/main/'
root=Path('data/external/ban-sign-sent'); root.mkdir(parents=True,exist_ok=True)
book=root/'sentences.xlsx'
if not book.exists():
    with urllib.request.urlopen(BASE+'Bangla_Sign_Sentence.xlsx?download=true',timeout=60) as r:
        book.write_bytes(r.read())
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as archive:
    shared=[]
    if 'xl/sharedStrings.xml' in archive.namelist():
        shared=[''.join(e.itertext()) for e in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si',ns)]
    sheet=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    rows=[]
    for row in sheet.findall('.//s:row',ns):
        values=[]
        for cell in row.findall('s:c',ns):
            value=cell.find('s:v',ns)
            if value is not None:
                text=value.text or ''
                if cell.get('t')=='s': text=shared[int(text)]
            else:
                text=''.join(cell.itertext())
            values.append(text)
        rows.append(values)
(root/'annotations.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('Annotation rows:',len(rows),flush=True)
print(json.dumps(rows[:5],ensure_ascii=True),flush=True)
for row in rows:
    text=' '.join(row)
    if any(word in text for word in ['কেমন আ','ভালো আ','নাম কী','নাম কি','সাহায্য','ধন্যবাদ']):
        print(json.dumps(row,ensure_ascii=True),flush=True)
request=urllib.request.Request(BASE+'Sign_Videos.zip?download=true',headers={'Range':'bytes=-2097152'})
with urllib.request.urlopen(request,timeout=120) as response:
    print('Archive range status:',response.status,'Content-Range:',response.headers.get('Content-Range'),flush=True)
    if response.status!=206:
        raise RuntimeError('Server did not honour byte range; refusing a full archive download')
    tail=response.read()
(root/'zip-tail.bin').write_bytes(tail)
eocd=tail.rfind(b'PK\x05\x06')
if eocd<0: raise ValueError('Archive end directory not found')
header=struct.unpack_from('<4s4H2LH',tail,eocd)
size=header[5]; offset=header[6]
print('End-directory:',header,flush=True)
position=tail.find(b'PK\x01\x02'); entries=[]
while position>=0 and tail[position:position+4]==b'PK\x01\x02':
    fields=struct.unpack_from('<4s6H3L5H2L',tail,position)
    name=tail[position+46:position+46+fields[10]].decode('utf-8',errors='replace')
    entries.append({'name':name,'disk':fields[13],'offset':fields[16],'compressed_size':fields[8],
                    'size':fields[9],'method':fields[4],'crc32':fields[7]})
    position+=46+fields[10]+fields[11]+fields[12]
(root/'archive-index.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
print('Indexed entries:',len(entries),flush=True)
print(json.dumps(entries[:12]),flush=True)
