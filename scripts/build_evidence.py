"""Build a 30-second held-out-image slideshow and browseable prediction audit."""
import argparse
import html
import json
import unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',default='reports/original')
    parser.add_argument('--data',default='data/original/Gray')
    parser.add_argument('--name',default='original')
    args=parser.parse_args()
    root=Path(args.report); rows=json.loads((root/'predictions.json').read_text(encoding='utf-8'))
    assets=Path('assets')/args.name; assets.mkdir(parents=True,exist_ok=True)
    thumbs=assets/'test'; thumbs.mkdir(exist_ok=True)
    fontpath='assets/fonts/NotoSansBengali.ttf'
    bengali=ImageFont.truetype(fontpath,64)
    sans=ImageFont.truetype('DejaVuSans.ttf',20) if Path('DejaVuSans.ttf').exists() else ImageFont.load_default(size=20)
    heading=ImageFont.load_default(size=32)
    def visible_label(label):
        return ('◌'+label) if unicodedata.category(label[0]).startswith('M') else label
    cards=[]; table=[]
    # Deterministic examples: successes and failures, both included when available.
    correct=[(i,r) for i,r in enumerate(rows) if r['correct']]
    wrong=[(i,r) for i,r in enumerate(rows) if not r['correct']]
    chosen=(correct[:6]+wrong[:6])
    if len(chosen)<12:
        used={i for i,_ in chosen}
        chosen += [(i,r) for i,r in enumerate(rows) if i not in used][:12-len(chosen)]
    for i,row in enumerate(rows):
        image=Image.open(Path(args.data)/row['path']).convert('RGB')
        image.thumbnail((256,256)); image.save(thumbs/f'{i:03d}.png')
        image_url=f'../assets/{args.name}/test/{i:03d}.png'
        status='Correct' if row['correct'] else 'Incorrect'
        table.append(f"<article class='{status.lower()}'><img loading='lazy' src='{image_url}' alt='Held-out hand sign {i}'><h3>{status}</h3><p>Actual: {html.escape(row['actual'])}<br>Predicted: {html.escape(row['predicted'])}<br>Score: {row['confidence']:.1%}</p><small>{html.escape(row['path'])}</small></article>")
    for count,(i,row) in enumerate(chosen):
        card=Image.new('RGB',(900,500),'#0b1220'); draw=ImageDraw.Draw(card)
        draw.text((32,22),'BANGLA SIGN LANGUAGE DETECTOR',font=heading,fill='#edf2f7')
        draw.text((32,70),'Held-out image slideshow | static letters',font=sans,fill='#a5b4c8')
        image=Image.open(thumbs/f'{i:03d}.png').convert('RGB'); image.thumbnail((310,310));
        card.paste(image,(40+(310-image.width)//2,130+(310-image.height)//2))
        draw.text((400,125),'Predicted',font=sans,fill='#a5b4c8')
        draw.text((400,165),visible_label(row['predicted']),font=bengali,fill='#5eead4',anchor='lt')
        draw.text((400,250),f"Confidence {row['confidence']:.1%}",font=heading,fill='white')
        draw.text((400,305),'Actual',font=sans,fill='#a5b4c8')
        draw.text((510,310),visible_label(row['actual']),font=bengali,fill='white',anchor='lt')
        draw.text((400,390),'CORRECT' if row['correct'] else 'INCORRECT',font=heading,fill='#5eead4' if row['correct'] else '#fb7185')
        draw.text((32,462),f'{count+1:02d}/12 | Uncalibrated score | Held-out predictions',font=sans,fill='#a5b4c8')
        cards.append(card)
    cards[0].save(assets/'demo.gif',save_all=True,append_images=cards[1:],duration=2500,loop=0,optimize=True)
    cards[0].save(assets/'preview.png')
    metrics=json.loads((root/'metrics.json').read_text())
    attribution=('KU-BdSL v4, USLD. Abdullah Al Jaid Jim, Ibrahim Rafi, Md. Zahid Akon, Uzzal Biswas and Abdullah-Al Nahid (2023). '
                 'DOI: 10.17632/scpvm2nbkm.4. CC BY 4.0. Thumbnails resized; prediction overlays added.' if args.name.startswith('ku') else
                 'Original thesis hand crops, Nazmul Hassan (2022). Thumbnails resized; prediction overlays added. See data/original/README.md for provenance limits.')
    docs=Path('docs'); docs.mkdir(exist_ok=True)
    (docs/f'{args.name}-results.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bangla Sign Lab - prediction audit</title>
<style>body{{background:#0b1220;color:#edf2f7;font:17px system-ui;max-width:1200px;margin:32px auto;padding:24px}}a{{color:#5eead4}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:20px}}article{{background:#172237;padding:18px;border-radius:12px}}img{{width:100%;height:200px;object-fit:contain;background:white}}.incorrect{{border:2px solid #fb7185}}small{{overflow-wrap:anywhere}}button{{padding:12px;margin:8px;cursor:pointer}}</style>
<h1>Reproduction results: complete held-out audit</h1><p>{metrics['accuracy']:.2%} accuracy on {metrics['test_n']} test images. Image-level split, seed 2026; signer-independent generalisation has not been measured. Confidence is an uncalibrated softmax score.</p>
<p>{html.escape(attribution)} <a href="https://data.mendeley.com/datasets/scpvm2nbkm/4">Public dataset</a></p>
<p>Every held-out prediction is shown, including mistakes. The GIF uses selected successes and failures and is not a live-signing video.</p>
<button onclick="filter('all')">All</button><button onclick="filter('correct')">Correct</button><button onclick="filter('incorrect')">Incorrect</button><div class="grid">{''.join(table)}</div>
<script>function filter(kind){{document.querySelectorAll('article').forEach(x=>x.hidden=kind!=='all'&&!x.classList.contains(kind));}}</script></html>''',encoding='utf-8')
    summary={'duration_seconds':len(cards)*2.5,'frames':len(cards),'test_images':len(rows),'selected_test_indices':[i for i,_ in chosen],
             'correct_selected':sum(r['correct'] for _,r in chosen),'incorrect_selected':sum(not r['correct'] for _,r in chosen),
             'type':'held-out still-image slideshow; not natural signing video','attribution':attribution}
    (root/'demo.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
