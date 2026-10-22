"""Re-render saved confusion counts without rerunning models or changing measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for name in ['original','mobilenet','ku-mobilenet']:
    root=Path('reports')/name
    labels=[x['folder'] for x in json.loads((root/'labels.json').read_text(encoding='utf-8'))]
    matrix=np.loadtxt(root/'confusion_matrix.csv',delimiter=',',dtype=int)
    fig,ax=plt.subplots(figsize=(10,9)); image=ax.imshow(matrix,cmap='Blues'); fig.colorbar(image,ax=ax)
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
           xlabel='Predicted folder ID',ylabel='Actual folder ID',title='Reproduction: held-out test confusion matrix')
    ax.tick_params(axis='x',labelrotation=90,labelsize=7); ax.tick_params(axis='y',labelsize=7)
    fig.tight_layout(); fig.savefig(root/'confusion_matrix.png',dpi=160); plt.close(fig)
    path=root/'metrics.json'; data=json.loads(path.read_text(encoding='utf-8'))
    data['label']='Reproduction results'
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
