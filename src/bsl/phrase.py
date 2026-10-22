"""Small-vocabulary motion-template experiment, with explicit unknown rejection."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .media import effective_fps

def resample(sequence,length=32):
    sequence=np.asarray(sequence,dtype=np.float32)
    if len(sequence)<2:
        raise ValueError('Need at least two motion frames')
    return np.stack([np.interp(np.linspace(0,1,length),np.linspace(0,1,len(sequence)),sequence[:,i])
                     for i in range(sequence.shape[1])],axis=1).astype(np.float32)

def dtw(a,b):
    """Mean L1 local distance with a bounded warping path; a distance, not probability."""
    costs=np.mean(np.abs(a[:,None,:]-b[None,:,:]),axis=2)
    state=np.full((len(a)+1,len(b)+1),np.inf); state[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(max(1,i-8),min(len(b),i+8)+1):
            state[i,j]=costs[i-1,j-1]+min(state[i-1,j],state[i,j-1],state[i-1,j-1])
    return float(state[-1,-1]/max(len(a),len(b)))

class PhraseRecognizer:
    def __init__(self,path='models/phrases.npz',threshold=.20,min_margin=.10):
        data=np.load(path,allow_pickle=False)
        self.templates=data['templates']; self.labels=list(data['labels'])
        self.threshold=threshold; self.min_margin=min_margin

    def __call__(self,sequence):
        query=resample(sequence)
        ranked=sorted((dtw(query,template),label) for template,label in zip(self.templates,self.labels))
        distance,label=ranked[0]
        other=next((d for d,l in ranked if l!=label),float('inf'))
        margin=1.0 if np.isinf(other) else (other-distance)/max(other,1e-6)
        accepted=distance<=self.threshold and margin>=self.min_margin
        return {'label':str(label) if accepted else 'Unrecognised','candidate':str(label),
                'distance':distance,'margin':margin,'accepted':accepted,
                'score_type':'template distance; not calibrated confidence'}

def build(args):
    from .motion import extract_video
    annotations=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    cache=Path('tmp/motion'); cache.mkdir(parents=True,exist_ok=True)
    recordings={}
    for video in annotations['videos']:
        times,features,visible=extract_video(video['path'],args.models,video.get('roi'))
        recordings[video['id']]=(times,features,visible)
        np.savez_compressed(cache/(video['id']+'.npz'),times=times,features=features,visible=visible)
        print('Extracted '+video['id'],flush=True)
    templates=[]; labels=[]
    for segment in annotations['segments']:
        if segment['split']!='reference': continue
        times,features,visible=recordings[segment['video']]
        mask=(times>=segment['start'])&(times<segment['end'])&visible
        if mask.sum()<8: raise ValueError('Too few tracked reference frames')
        templates.append(resample(features[mask])); labels.append(segment['label'])
    target=Path(args.templates); target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,templates=np.stack(templates),labels=np.array(labels))
    recognizer=PhraseRecognizer(target,args.threshold,args.margin)
    predictions=[]
    for segment in annotations['segments']:
        if segment['split']!='test': continue
        times,features,visible=recordings[segment['video']]
        mask=(times>=segment['start'])&(times<segment['end'])&visible
        result=recognizer(features[mask]) if mask.sum()>=8 else {'label':'Unrecognised','accepted':False,'reason':'insufficient tracking'}
        predictions.append({**segment,**result,'actual':segment['label'],'correct':result['label']==segment['label'],
                            'tracked_frames':int(mask.sum())})
    known=[r for r in predictions if r['actual']!='Unrecognised']
    unknown=[r for r in predictions if r['actual']=='Unrecognised']
    report={'method':'upper-body/hand landmarks and DTW; one reference example per phrase',
            'task':'manually segmented phrase classification; not open-ended sentence transcription',
            'reference_source':'Grameenphone tutorial','test_source':'BRAC tutorial; different signer',
            'reference_phrases':len(set(labels)),'threshold':args.threshold,'min_margin':args.margin,
            'test_segments':len(predictions),'known_segments':len(known),
            'known_correct':sum(r['correct'] for r in known),
            'known_accuracy':sum(r['correct'] for r in known)/len(known) if known else None,
            'unknown_segments':len(unknown),'unknown_rejected':sum(r['correct'] for r in unknown),
            'unknown_rejection_rate':sum(r['correct'] for r in unknown)/len(unknown) if unknown else None,
            'predictions':predictions,'limitations':['One reference signer, very small manually segmented test',
              'Tutorial captions supply manual labels only; model consumes landmarks, not caption pixels',
              'Thresholds specified before this test; no test-based tuning',
              'No public-deployment readiness claim; natural stream segmentation unmeasured']}
    out=Path('reports/phrases'); out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=True),flush=True)

def live(args):
    from .motion import LandmarkTracker,crop_signer
    from .preview import annotate
    recognizer=PhraseRecognizer(args.templates,args.threshold,args.margin)
    tracker=LandmarkTracker(args.models)
    capture=cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        tracker.close(); raise ValueError('Cannot open video/camera')
    buffer=[]; transcript=[]; last_emit=''; frame_index=0; sample_count=0
    stable_candidate=''; stable_count=0; last_emit_time=-10.
    fps=effective_fps(capture.get(cv2.CAP_PROP_FPS))
    next_sample=0.; start=time.monotonic()
    result={'label':'Unrecognised','accepted':False,'distance':1.}
    output=open(args.output,'w',encoding='utf-8') if args.output else None
    try:
        while True:
            ok,frame=capture.read()
            if not ok: break
            t=frame_index/fps if args.video else time.monotonic()-start; frame_index+=1
            if t<next_sample: continue
            next_sample=t+.125
            feature,valid=tracker.extract(crop_signer(frame,args.roi),int(t*1000))
            sample_count+=1
            buffer.append(feature if valid else np.zeros_like(feature)); buffer=buffer[-64:]
            if len(buffer)>=16 and sample_count%4==0:
                candidates=[recognizer(np.array(buffer[-n:])) for n in (16,32,48,64) if len(buffer)>=n]
                accepted=[r for r in candidates if r['accepted']]
                result=min(accepted,key=lambda r:r['distance']) if accepted else {'label':'Unrecognised','accepted':False,'distance':1.}
                if result['accepted'] and result['label']==stable_candidate:
                    stable_count+=1
                else:
                    stable_candidate=result['label']; stable_count=1
                if result['accepted'] and stable_count>=2 and result['label']!=last_emit and t-last_emit_time>=2.5:
                    transcript.append(result['label']); last_emit=result['label']
                    last_emit_time=t
                    row={'seconds':t,**result,'text':' '.join(transcript)}
                    if output: output.write(json.dumps(row,ensure_ascii=False)+'\n'); output.flush()
                    else: print(json.dumps(row,ensure_ascii=True),flush=True)
            if not args.no_preview:
                # Similarity is labelled explicitly in the window title, never a probability.
                view=annotate(frame,{'label':result['label'],'confidence':max(0,1-result['distance'])},' '.join(transcript[-2:]))
                cv2.imshow('Common phrases EXPERIMENT | score = 1-distance | q quit | c clear',view)
                key=cv2.waitKey(1)&255
                if key in (ord('q'),27): break
                if key==ord('c'): transcript.clear(); last_emit=''
    finally:
        capture.release(); tracker.close()
        if output: output.close()
        if not args.no_preview: cv2.destroyAllWindows()

def main(argv=None):
    parser=argparse.ArgumentParser(description='Experimental common-phrase motion recognition')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['build','live']:
        p=sub.add_parser(command)
        p.add_argument('--models',default='models'); p.add_argument('--templates',default='models/phrases.npz')
        p.add_argument('--threshold',type=float,default=.20); p.add_argument('--margin',type=float,default=.10)
        if command=='build': p.add_argument('--annotations',default='data/phrases/tutorials.json')
        else:
            p.add_argument('--video'); p.add_argument('--camera',type=int,default=0)
            p.add_argument('--roi',type=float,nargs=4); p.add_argument('--no-preview',action='store_true'); p.add_argument('--output')
    args=parser.parse_args(argv)
    try: return build(args) if args.command=='build' else live(args)
    except (ValueError,FileNotFoundError) as error: parser.exit(2,str(error)+'\n')

if __name__=='__main__': main()
