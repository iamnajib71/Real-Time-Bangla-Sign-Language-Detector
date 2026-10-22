"""Make and actually transcribe a selected held-out letter-spelling video."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from bsl.predict import Predictor
from bsl.preprocess import preprocess,read_image
from bsl.media import frames
from bsl.text import TextComposer,is_blank

target='আমি আসি'
tokens=['আ','ম','ই',' ','আ','স','ই']
rows=json.loads(Path('reports/mobilenet/predictions.json').read_text(encoding='utf-8'))
assets=Path('assets/sentence'); assets.mkdir(parents=True,exist_ok=True)
video=assets/'letter-spelling.avi'
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),4,(128,128))
if not writer.isOpened(): raise RuntimeError('FFV1 video writer unavailable')
blank=np.full((128,128,3),255,np.uint8)
selected=[]
for token in tokens:
    if token==' ':
        for _ in range(12): writer.write(blank)
        continue
    candidates=[r for r in rows if r['correct'] and r['actual']==token]
    if not candidates: raise ValueError('No correctly predicted held-out example for '+token)
    row=max(candidates,key=lambda r:r['confidence']); selected.append(row)
    image=preprocess(read_image(Path('data/original/Gray')/row['path']))
    frame=np.repeat((image*255).astype(np.uint8),3,axis=2)
    for _ in range(8): writer.write(frame)
    for _ in range(3): writer.write(blank)
for _ in range(8): writer.write(blank)
writer.release()

predictor=Predictor('models/mobilenet.keras')
composer=TextComposer(min_confidence=.4,stable_frames=3,word_gap_frames=12)
font=ImageFont.truetype('assets/fonts/NotoSansBengali.ttf',46)
sans=ImageFont.load_default(size=20); title=ImageFont.load_default(size=29)
cards=[]; predictions=[]
for source,frame in frames(video=video):
    result=predictor(frame); blank_frame=is_blank(frame)
    text=composer.update(result,blank=blank_frame)
    predictions.append({'frame':int(source),**result,'pause_card':blank_frame,'text':text})
    card=Image.new('RGB',(900,500),'#0b1220'); draw=ImageDraw.Draw(card)
    draw.text((32,25),'BANGLA LETTER-TO-TEXT DEMO',font=title,fill='white')
    draw.text((32,78),'Video decoded frame by frame | Selected held-out examples',font=sans,fill='#a5b4c8')
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((270,270))
    card.paste(image,(35,145))
    draw.text((355,145),'Pause / word delimiter' if blank_frame else 'Predicted letter',font=sans,fill='#a5b4c8')
    if not blank_frame:
        draw.text((355,190),result['label'],font=font,fill='#5eead4',anchor='lt')
        draw.text((450,205),f"{result['confidence']:.1%}",font=title,fill='white')
    draw.text((355,285),'Composed text',font=sans,fill='#a5b4c8')
    draw.text((355,330),text,font=font,fill='white',anchor='lt')
    draw.text((32,455),'Static letter spelling + vowel rules; separate from moving phrase recognition',font=sans,fill='#a5b4c8')
    cards.append(card)
if composer.text!=target:
    raise ValueError(f'Actual transcription {composer.text!r} does not match target {target!r}')
cards[0].save(assets/'demo.gif',save_all=True,append_images=cards[1:],duration=250,loop=0,optimize=True)
cards[-1].save(assets/'preview.png')
out=Path('reports/sentence'); out.mkdir(parents=True,exist_ok=True)
report={'task':'selected held-out letter-spelling video, not natural continuous signing','target':target,'actual':composer.text,
        'matched':True,'frames':len(cards),'duration_seconds':len(cards)/4,'confidence_threshold':.4,
        'source_selection':'highest-confidence correctly predicted held-out example per required letter; illustrative, not an independent benchmark',
        'source_predictions':selected,'model_sha256':hashlib.sha256(Path('models/mobilenet.keras').read_bytes()).hexdigest(),
        'all_frames':predictions}
(out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['all_frames','source_predictions']},ensure_ascii=True))
