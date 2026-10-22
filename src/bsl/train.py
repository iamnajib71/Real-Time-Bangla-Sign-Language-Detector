"""Fixed, content-deduplicated split and a single original-CNN training run."""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import time
from collections import defaultdict, Counter

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
import cv2
import numpy as np
from .preprocess import preprocess, read_image
from .media import EXTENSIONS

SEED = 2026
# Folder order is saved explicitly; directory iterators use lexicographic order.
# Class 6 was a duplicate of class 4 in the old GUI; mapping corrected to O (ও).
THESIS_LABELS = ['অ','আ','ই','উ','এ','ও','ক','খ','গ','ঘ','চ','ছ','জ','ঝ','ট','ঠ','ড','ঢ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য়','র','ল','স','হ','ড়','ং','ঃ']

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def inventory(root):
    root = Path(root)
    folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    labels = [p.name for p in folders]
    groups = defaultdict(list)
    unreadable = []
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                pixels = preprocess(read_image(path))
            except ValueError:
                unreadable.append(path.relative_to(root).as_posix())
                continue
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            groups[digest].append((folder.name, path.relative_to(root).as_posix()))
    conflicts = {h: v for h, v in groups.items() if len({x[0] for x in v}) > 1}
    records = []
    for digest, entries in sorted(groups.items()):
        if digest not in conflicts:
            label, path = entries[0]
            records.append({'path': path, 'class': label, 'sha256_preprocessed': digest,
                            'duplicates': [p for _, p in entries[1:]]})
    return labels, records, {'input_images': sum(map(len, groups.values())),
        'unique_usable_images': len(records), 'duplicates_removed': sum(len(v)-1 for h,v in groups.items() if h not in conflicts),
        'conflicting_content': conflicts, 'unreadable': unreadable,
        'class_counts': dict(Counter(r['class'] for r in records))}

def split_records(records, seed=SEED):
    rng = random.Random(seed)
    result = []
    for label in sorted({r['class'] for r in records}):
        group = [dict(r) for r in records if r['class'] == label]
        rng.shuffle(group)
        if len(group) < 5:
            raise ValueError(f'Insufficient unique images for class {label}')
        ntest = max(1, round(len(group)*.15))
        nval = max(1, round(len(group)*.15))
        for i, record in enumerate(group):
            record['split'] = 'test' if i < ntest else 'validation' if i < ntest+nval else 'train'
            result.append(record)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/original/Gray')
    parser.add_argument('--out', default='reports/original')
    parser.add_argument('--model', default='models/original.keras')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--architecture', choices=['original','mobilenet'], default='original')
    parser.add_argument('--split-from', help='Reuse a saved split after validating image hashes')
    args = parser.parse_args(argv)
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from .model import original_cnn, mobilenet_baseline
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('Run already measured; choose a new output directory to preserve evidence')
    labels, records, audit = inventory(args.data)
    records = split_records(records)
    if args.split_from:
        saved = json.loads(Path(args.split_from).read_text(encoding='utf-8'))
        key = lambda r: (r['path'],r['class'],r['sha256_preprocessed'])
        if {key(r) for r in saved} != {key(r) for r in records}:
            raise ValueError('Saved split does not match current dataset')
        records = saved
    if not args.external:
        if set(labels) != {str(i) for i in range(1,37)}:
            raise ValueError('Thesis run requires all 36 numeric classes')
        display = [THESIS_LABELS[int(x)-1] for x in labels]
    else:
        display = ['/'.join(chr(int(c)) for c in x.split('-')) if all(c.isdigit() for c in x.split('-')) else x for x in labels]
    audit.update(seed=SEED, split_counts=dict(Counter(r['split'] for r in records)), classes=labels,
                 split_unit='unique preprocessed image; signer identity unavailable; near duplicates may remain')
    write_json(out/'inventory.json', audit)
    write_json(out/'split.json', records)
    write_json(out/'labels.json', [{'folder': x, 'display': y} for x,y in zip(labels,display)])
    sets = {}
    for name in ('train','validation','test'):
        subset = [r for r in records if r['split']==name]
        sets[name] = (np.stack([preprocess(read_image(Path(args.data)/r['path'])) for r in subset]),
                      np.array([labels.index(r['class']) for r in subset]), subset)
    model = original_cnn(len(labels)) if args.architecture == 'original' else mobilenet_baseline(len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    (out/'architecture.json').write_text(model.to_json(), encoding='utf-8')
    augmentation = ImageDataGenerator(shear_range=.2, zoom_range=.2, horizontal_flip=True)
    train_model = model
    if args.architecture == 'mobilenet':
        # Cache frozen features once on CPU. The final saved model still accepts images.
        feature_model = tf.keras.Model(model.input, model.layers[-3].output)
        feature_sets = {}
        for name,(images,targets,subset) in sets.items():
            features = np.concatenate([np.asarray(feature_model(images[i:i+16],training=False))
                                       for i in range(0,len(images),16)])
            feature_sets[name] = (features,targets,subset)
        train_model = tf.keras.Sequential([tf.keras.Input((1280,)),model.layers[-2],model.layers[-1]])
        train_model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
        sets = feature_sets
    x,y,_ = sets['train']
    generator = augmentation.flow(x, y, batch_size=18, seed=SEED, shuffle=True) if args.architecture=='original' else None
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_model.reset_metrics()
        if generator is not None:
            for _ in range(len(generator)):
                xb,yb = next(generator)
                train = train_model.train_on_batch(xb, yb, return_dict=True)
        else:
            order = np.random.default_rng(SEED+epoch).permutation(len(x))
            for i in range(0,len(x),18):
                ix=order[i:i+18]
                train = train_model.train_on_batch(x[ix],y[ix],return_dict=True)
        vx,vy,_ = sets['validation']
        scores = np.concatenate([np.asarray(train_model(vx[i:i+32], training=False)) for i in range(0,len(vx),32)])
        val_acc = float(np.mean(scores.argmax(1)==vy))
        item = {'epoch':epoch+1, **{k:float(v) for k,v in train.items()}, 'val_accuracy':val_acc}
        history.append(item)
        print(json.dumps(item), flush=True)
    seconds = time.perf_counter()-start
    tx,ty,subset = sets['test']
    probabilities = np.concatenate([np.asarray(train_model(tx[i:i+32], training=False)) for i in range(0,len(tx),32)])
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(ty,predicted,labels=list(range(len(labels))))
    metrics = {'label':'Reproduction results', 'accuracy':float(accuracy_score(ty,predicted)),
        'architecture':args.architecture,
        'test_n':len(ty), 'classes':len(labels), 'seed':SEED, 'epochs':args.epochs,
        'train_seconds':seconds, 'parameters':model.count_params(), 'batch_size':18,
        'optimizer':'Adam defaults; learning rate 0.001',
        'augmentation':('shear 0.2, zoom 0.2, horizontal flip (legacy settings; not language validated)'
                        if args.architecture=='original' else 'none; frozen ImageNet features, dropout 0.2 on classifier'),
        'feature_extraction':'frozen MobileNetV2; cached features; head training time excludes extraction' if args.architecture=='mobilenet' else None,
        'tensorflow':tf.__version__, 'python':platform.python_version(), 'platform':platform.platform(),
        'per_class':classification_report(ty,predicted,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)}
    write_json(out/'metrics.json', metrics)
    write_json(out/'history.json', history)
    np.savetxt(out/'confusion_matrix.csv',matrix,fmt='%d',delimiter=',')
    write_json(out/'predictions.json', [{**r,'actual_index':int(a),'predicted_index':int(p),'actual':display[a],
              'predicted':display[p],'confidence':float(probabilities[i,p]),'correct':bool(a==p)}
              for i,(r,a,p) in enumerate(zip(subset,ty,predicted))])
    with (out/'per_class.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['folder','label','precision','recall','f1','support'])
        for label, char in zip(labels,display):
            d=metrics['per_class'][label]
            writer.writerow([label,char,d['precision'],d['recall'],d['f1-score'],d['support']])
    model_path=Path(args.model); model_path.parent.mkdir(parents=True,exist_ok=True)
    model.save(model_path)
    write_json(model_path.with_suffix('.labels.json'), display)
    write_json(out/'model.json', {'file':model_path.name,'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(10,9))
    image=ax.imshow(matrix,cmap='Blues'); fig.colorbar(image,ax=ax)
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
           xlabel='Predicted folder ID',ylabel='Actual folder ID', title='Reproduction: held-out test confusion matrix')
    ax.tick_params(axis='x',labelrotation=90,labelsize=7); ax.tick_params(axis='y',labelsize=7)
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k!='per_class'}),flush=True)

if __name__=='__main__':
    main()
