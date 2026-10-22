"""Train a small temporal model on a fixed signer-ID-disjoint sentence subset.

Selection uses U7 validation only; U19 is evaluated once after model/threshold
selection. The task is a bounded sentence vocabulary, not open-ended translation.
"""
import os
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import hashlib
import json
from pathlib import Path
import time
import argparse
import numpy as np
from bsl.motion import extract_video
from bsl.phrase_features import phrase_features
from bsl.phrase import dtw

parser=argparse.ArgumentParser(); parser.add_argument('--background',action='store_true'); args=parser.parse_args()
PREFIX='sentences-background' if args.background else 'sentences'
ROOT=Path('reports')/PREFIX; manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
if (ROOT/'test-results.json').exists(): raise FileExistsError('Final test already measured; refusing to overwrite it')
cache=Path('data/external/ban-sign-sent/features'); cache.mkdir(exist_ok=True)
records=manifest['records']; sequences=[]
start=time.perf_counter()
for i,record in enumerate(records):
    path=cache/(Path(record['path']).stem+'.npz')
    if path.exists():
        saved=np.load(path,allow_pickle=False); features=saved['features']; visible=saved['visible']
        if visible.sum()<8:
            times,features,visible=extract_video(record['path'],fps=8)
            np.savez_compressed(path,features=features,visible=visible,times=times)
    else:
        times,features,visible=extract_video(record['path'],fps=8)
        np.savez_compressed(path,features=features,visible=visible,times=times)
    record['tracked_frames']=int(visible.sum())
    sequences.append(phrase_features(features[visible],'shape-motion') if visible.sum()>=8 else None)
    print(f'Tracked {i+1}/{len(records)} clips',flush=True)
extract_seconds=time.perf_counter()-start
labels=[r[0] for r in manifest['selected']]
texts=[r[1] for r in manifest['selected']]
def data(split,known=None):
    indices=[i for i,r in enumerate(records) if r['split']==split and (known is None or r['known']==known)
             and (split!='train' or sequences[i] is not None)]
    shape=next(s.shape for s in sequences if s is not None)
    x=np.stack([sequences[i] if sequences[i] is not None else np.zeros(shape,np.float32) for i in indices])
    y=np.array([labels.index(records[i]['sentence_id']) if records[i]['known'] else -1 for i in indices])
    return x,y,indices
train_x,train_y,_=data('train',None if args.background else True)
if args.background: train_y=np.where(train_y<0,len(labels),train_y)
val_x,val_y,val_indices=data('validation')
val_tracked=np.array([sequences[i] is not None for i in val_indices])
dimension=train_x.shape[-1]
import tensorflow as tf
tf.keras.utils.set_random_seed(2026); tf.config.experimental.enable_op_determinism()
model=tf.keras.Sequential([tf.keras.Input((32,dimension)),tf.keras.layers.GRU(64),
    tf.keras.layers.Dropout(.3),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(len(labels)+int(args.background),activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy')
rng=np.random.default_rng(2026); history=[]; best=(-1.,-float('inf')); best_weights=None; best_epoch=0
for epoch in range(80):
    order=rng.permutation(len(train_x)*4)%len(train_x)
    losses=[]
    for i in range(0,len(order),8):
        ids=order[i:i+8]; batch=train_x[ids].copy()
        for j in range(len(batch)):
            axis=np.linspace(0,1,32)
            warped=np.clip(axis**rng.uniform(.8,1.2),0,1)
            batch[j]=np.stack([np.interp(warped,axis,batch[j,:,k]) for k in range(dimension)],axis=1)
        batch+=rng.normal(0,.01,batch.shape).astype(np.float32)
        model.reset_metrics()
        losses.append(float(model.train_on_batch(batch,train_y[ids])))
    probabilities=np.asarray(model(val_x,training=False)); known=val_y>=0
    acc=float(np.mean(probabilities[known].argmax(1)==val_y[known]))
    rejection=float(np.mean(probabilities[~known].argmax(1)==len(labels))) if args.background else 0.
    objective=(acc+rejection)/2 if args.background else acc
    nll=float(-np.mean(np.log(np.maximum(probabilities[known,val_y[known]],1e-8))))
    history.append({'epoch':epoch+1,'train_batch_loss_mean':float(np.mean(losses)),'validation_known_accuracy':acc,'validation_background_rejection':rejection,'validation_known_nll':nll})
    if (objective,-nll)>best:
        best=(objective,-nll); best_weights=model.get_weights(); best_epoch=epoch+1
    if epoch%10==9: print(json.dumps(history[-1]),flush=True)
model.set_weights(best_weights)
val_prob=np.asarray(model(val_x,training=False))
# Validation-only threshold selection; report coverage and false accepts as well as accuracy.
def measure(y,predicted,accepted):
    known=y>=0; unknown=~known
    correct=accepted&known&(predicted==y)
    return {'known_n':int(known.sum()),'known_correct':int(correct.sum()),
        'known_accuracy':float(correct.sum()/known.sum()) if known.any() else None,
        'unknown_n':int(unknown.sum()),'unknown_rejected':int((unknown&~accepted).sum()),
        'unknown_rejection':float((unknown&~accepted).sum()/unknown.sum()) if unknown.any() else None,
        'accepted_n':int(accepted.sum()),'accepted_precision':float(correct.sum()/accepted.sum()) if accepted.any() else 0.,
        'balanced_score':float((correct.sum()/known.sum()+(unknown&~accepted).sum()/unknown.sum())/2)}
trials=[]
for threshold in [.2,.3,.4,.5,.6,.7,.8,.9,.95]:
    prediction=val_prob.argmax(1)
    trials.append({'backend':'gru','threshold':threshold,**measure(val_y,prediction,(val_prob.max(1)>=threshold)&val_tracked&(prediction<len(labels)))})
distances=np.array([[dtw(query,template) for template in train_x] for query in val_x])
nn=distances.argmin(1); nearest=distances.min(1); nn_labels=train_y[nn]
for threshold in [.03,.05,.08,.1,.12,.15,.2,.3]:
    trials.append({'backend':'dtw','threshold':threshold,**measure(val_y,nn_labels,(nearest<=threshold)&val_tracked&(nn_labels<len(labels)))})
selected=max(trials,key=lambda r:(r['balanced_score'],r['accepted_precision'],r['unknown_rejection']))
config={**selected,'vocabulary_ids':labels,'labels':texts,'feature_variant':'shape-motion','features':dimension,'time_steps':32,
        'seed':2026,'epochs':80,'selected_epoch':best_epoch,'method':'GRU temporal classifier or DTW baseline, selected on validation only',
        'split':'U4/U5/U6 train; U7 validation; U19 untouched test','training_clips':len(train_x),
        'source':manifest['source'],'data_license':manifest['license_statement'],'extract_seconds':extract_seconds}
config['background_class']=args.background
config['test_disclosure']='U19 is excluded from fitting and validation; a prior conservative baseline test outcome was observed' if args.background else 'U19 evaluated after selection, once'
Path('models').mkdir(exist_ok=True); model.save(f'models/{PREFIX}.keras')
Path(f'models/{PREFIX}.labels.json').write_text(json.dumps(texts+(['Unrecognised'] if args.background else []),ensure_ascii=False,indent=2),encoding='utf-8')
np.savez_compressed(f'models/{PREFIX}-dtw.npz',templates=train_x,targets=train_y)
(ROOT/'validation.json').write_text(json.dumps({'trials':trials,'selected':selected,'history':history},indent=2),encoding='utf-8')
Path(f'models/{PREFIX}.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
(ROOT/'frozen-config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
# All selection is complete above. Only now access the final test inputs/labels.
test_x,test_y,test_indices=data('test')
test_tracked=np.array([sequences[i] is not None for i in test_indices])
if selected['backend']=='gru':
    scores=np.asarray(model(test_x,training=False)); predicted=scores.argmax(1); strength=scores.max(1)
    accepted=(strength>=selected['threshold'])&test_tracked&(predicted<len(labels))
else:
    scores=np.array([[dtw(q,t) for t in train_x] for q in test_x]); strength=scores.min(1)
    predicted=train_y[scores.argmin(1)]; accepted=(strength<=selected['threshold'])&test_tracked&(predicted<len(labels))
metrics=measure(test_y,predicted,accepted)
predictions=[]
for i,index in enumerate(test_indices):
    record=records[index]
    predictions.append({**record,'predicted':texts[predicted[i]] if accepted[i] else 'Unrecognised',
        'accepted':bool(accepted[i]),'score':float(strength[i]),'correct':bool(accepted[i] and predicted[i]==test_y[i]) if test_y[i]>=0 else not bool(accepted[i])})
report={'task':'10-class bounded sentence classification with unknown rejection; not unrestricted translation',
        'split':config['split'],'selected':selected,'test':metrics,'predictions':predictions,
        'model_sha256':hashlib.sha256(Path(f'models/{PREFIX}.keras').read_bytes()).hexdigest(),
        'test_disclosure':config['test_disclosure'],
        'tracking_audit':[{'path':r['path'],'split':r['split'],'known':r['known'],'tracked_frames':r['tracked_frames']} for r in records],
        'limitations':['Only five dataset user IDs; identities not independently verified','One clip per class per signer',
          'Manually segmented complete clips; streaming segmentation not validated','Dataset card licence not stated; no footage redistributed']}
(ROOT/'test-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'selected':selected,'test':metrics}),flush=True)
