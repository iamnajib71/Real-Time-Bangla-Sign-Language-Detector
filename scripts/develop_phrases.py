"""Select features and rejection thresholds on the two development videos only.

The old BRAC test is now development data. No result from this search is a new
held-out test score. Fresh independent footage is reserved for final evaluation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from bsl.phrase import dtw
from bsl.phrase_features import phrase_features,mirror_sequence

VOCAB=['হ্যালো','আপনি কেমন আছেন?','আমি ভালো আছি','আপনার নাম কী?']
annotations=json.loads(Path('data/phrases/tutorials.json').read_text(encoding='utf-8'))
segments=[]
for s in annotations['segments']:
    cache=np.load(Path('tmp/motion')/(s['video']+'.npz'))
    mask=(cache['times']>=s['start'])&(cache['times']<s['end'])&cache['visible']
    if mask.sum()<8: continue
    label=s['label'] if s['label'] in VOCAB else 'Unrecognised'
    segments.append({**s,'label':label,'features':cache['features'][mask]})
trials=[]; details={}
for variant in ['pose','motion','shape','shape-motion']:
    features=[phrase_features(s['features'],variant) for s in segments]
    mirrored=[phrase_features(mirror_sequence(s['features']),variant) for s in segments]
    for allow_mirror in [False,True]:
      rows=[]
      for i,s in enumerate(segments):
        references=[(j,r) for j,r in enumerate(segments) if r['video']!=s['video']]
        ranked=sorted((min(dtw(features[i],features[j]),dtw(features[i],mirrored[j])) if allow_mirror else dtw(features[i],features[j]),r['label']) for j,r in references)
        best,label=ranked[0]; other=next(d for d,l in ranked if l!=label)
        rows.append({'video':s['video'],'actual':s['label'],'candidate':label,'distance':best,'margin':(other-best)/max(other,1e-8)})
      for threshold in [.08,.12,.16,.20,.25,.30,.40,.50]:
        for margin in [.0,.05,.10,.20,.30]:
            known=[r for r in rows if r['actual']!='Unrecognised']; unknown=[r for r in rows if r['actual']=='Unrecognised']
            def prediction(r):
                return r['candidate'] if r['distance']<=threshold and r['margin']>=margin else 'Unrecognised'
            accuracy=sum(prediction(r)==r['actual'] for r in known)/len(known)
            rejection=sum(prediction(r)=='Unrecognised' for r in unknown)/len(unknown)
            objective=(accuracy+rejection)/2
            trial={'variant':variant,'allow_mirror':allow_mirror,'threshold':threshold,'margin':margin,'known_correct':round(accuracy*len(known)),
                   'known_n':len(known),'unknown_rejected':round(rejection*len(unknown)),'unknown_n':len(unknown),
                   'balanced_score':objective,'known_accuracy':accuracy,'unknown_rejection':rejection}
            trials.append(trial)
      details[variant+str(allow_mirror)]=rows
best=max(trials,key=lambda r:(r['balanced_score'],r['unknown_rejection'],-r['threshold'],r['margin']))
out=Path('reports/phrases-v2'); out.mkdir(parents=True,exist_ok=True)
report={'protocol':'two-video leave-one-source-out development; BRAC is no longer a blind test',
        'vocabulary':VOCAB,'selected':best,'trials':trials,'development_predictions':details[best['variant']+str(best['allow_mirror'])],
        'held_out_status':'fresh independent footage not evaluated by this script'}
(out/'development.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
templates=[phrase_features(s['features'],best['variant']) for s in segments]
labels=[s['label'] for s in segments]
if best['allow_mirror']:
    templates += [phrase_features(mirror_sequence(s['features']),best['variant']) for s in segments]
    labels += [s['label'] for s in segments]
np.savez_compressed('models/phrases-v2.npz',templates=np.stack(templates),labels=np.array(labels))
config={**best,'vocabulary':VOCAB,'method':'body-normalized pose, reassociated hands, palm-normalized shape and temporal DTW',
        'template_sha256':hashlib.sha256(Path('models/phrases-v2.npz').read_bytes()).hexdigest(),
        'reference_videos':['grameenphone','brac'],'test_selection':'independent footage excluded from all development searches'}
Path('models/phrases-v2.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
(out/'frozen-config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(best),flush=True)
