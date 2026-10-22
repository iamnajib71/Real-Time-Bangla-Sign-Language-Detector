"""Inference adapter for the measured, bounded sentence-vocabulary experiment."""
import json
from pathlib import Path
import numpy as np
from .phrase_features import phrase_features
from .phrase import dtw

class SentenceRecognizer:
    def __init__(self,models='models',prefix=None):
        models=Path(models)
        prefix=prefix or ('sentences-background' if (models/'sentences-background.json').exists() else 'sentences')
        self.config=json.loads((models/(prefix+'.json')).read_text(encoding='utf-8'))
        if self.config['backend']=='gru':
            from tensorflow.keras.models import load_model
            self.model=load_model(models/(prefix+'.keras'),compile=False)
        else:
            bank=np.load(models/(prefix+'-dtw.npz'),allow_pickle=False)
            self.templates=bank['templates']; self.targets=bank['targets']

    def __call__(self,sequence):
        features=phrase_features(sequence,self.config['feature_variant'])
        if self.config['backend']=='gru':
            scores=np.asarray(self.model(features[None],training=False))[0]
            index=int(scores.argmax()); score=float(scores[index]); accepted=score>=self.config['threshold'] and index<len(self.config['labels'])
        else:
            distances=[dtw(features,t) for t in self.templates]
            best=int(np.argmin(distances)); index=int(self.targets[best]); score=float(distances[best]); accepted=score<=self.config['threshold'] and index<len(self.config['labels'])
        return {'label':self.config['labels'][index] if accepted else 'Unrecognised','accepted':bool(accepted),
                'score':score,'score_type':'uncalibrated softmax' if self.config['backend']=='gru' else 'DTW distance'}

def main(argv=None):
    import argparse
    from .predict import ROOT
    parser=argparse.ArgumentParser(description='Bounded sentence-vocabulary video inference')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video'); source.add_argument('--camera',type=int)
    parser.add_argument('--models',default=str(ROOT/'models')); parser.add_argument('--prefix')
    parser.add_argument('--roi',type=float,nargs=4); parser.add_argument('--output')
    args=parser.parse_args(argv)
    try:
        if args.camera is not None:
            from .gui import launch
            return launch(Path(args.models)/'ku-mobilenet.keras',mode='sentences',camera=args.camera)
        from .motion import extract_video
        _,features,visible=extract_video(args.video,args.models,args.roi)
        result=SentenceRecognizer(args.models,args.prefix)(features[visible]) if visible.sum()>=8 else {
            'label':'Unrecognised','accepted':False,'reason':'insufficient tracked signing frames'}
        result.update(input=Path(args.video).name,tracked_frames=int(visible.sum()),task='completed-clip sentence classification')
        text=json.dumps(result,ensure_ascii=False)
        if args.output: Path(args.output).write_text(text+'\n',encoding='utf-8')
        else: print(json.dumps(result,ensure_ascii=True))
    except (ValueError,FileNotFoundError) as error: parser.exit(2,str(error)+'\n')

if __name__=='__main__': main()
