"""Derive sentence metrics without fitting or selecting anything on the test set."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('reports/sentences-background')
report=json.loads((root/'test-results.json').read_text(encoding='utf-8'))
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
# Clarify provenance without changing any predictions or measured values.
report['split']='U4/U5/U6 train; U7 validation; U19 test (prior baseline outcome observed)'
report['limitations'] += ['Background sentence types occur in training; rejection does not measure novel unsupported sentence types']
(root/'test-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
names=[x[0] for x in manifest['selected']]+['other']
text_to_id={x[1]:x[0] for x in manifest['selected']}
y=[r['sentence_id'] if r['known'] else 'other' for r in report['predictions']]
p=[text_to_id.get(r['predicted'],'other') for r in report['predictions']]
metrics=classification_report(y,p,labels=names,output_dict=True,zero_division=0)
matrix=confusion_matrix(y,p,labels=names)
(root/'classification-metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
with (root/'per_class.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.writer(stream); writer.writerow(['sentence_id','precision','recall','f1','support'])
    for label in names:
        m=metrics[label]; writer.writerow([label,m['precision'],m['recall'],m['f1-score'],int(m['support'])])
np.savetxt(root/'confusion_matrix.csv',matrix,delimiter=',',fmt='%d',header=','.join(names),comments='')
fig,ax=plt.subplots(figsize=(9,8)); ax.imshow(matrix,cmap='Blues')
ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,
       xlabel='Predicted sentence ID / rejected',ylabel='Actual sentence ID / background',title='Bounded sentence test: U19 dataset signer ID')
plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
for i in range(len(names)):
    for j in range(len(names)):
        if matrix[i,j]: ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>4 else 'black')
fig.tight_layout(); fig.savefig(root/'confusion_matrix.png',dpi=160); plt.close(fig)
print(json.dumps(report['test']))
