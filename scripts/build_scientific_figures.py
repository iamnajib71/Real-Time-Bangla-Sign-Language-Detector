"""Publication figures from saved histories and confusion counts; no model refitting."""
import json
from pathlib import Path
import shutil
import hashlib
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out=Path('paper/v2/figures'); out.mkdir(exist_ok=True)
def load(name,file): return json.loads((Path('reports')/name/file).read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(7.0,2.5))
for name,label,color in [('original','Original CNN','#b94b45'),('mobilenet','Frozen MobileNetV2','#147a81')]:
    h=load(name,'history.json'); axes[0].plot([r['epoch'] for r in h],[r['val_accuracy'] for r in h],label=label,color=color)
h=load('ku-mobilenet','history.json'); axes[1].plot([r['epoch'] for r in h],[r['val_accuracy'] for r in h],color='#147a81')
h=load('sentences-background','validation.json')['history']
axes[2].plot([r['epoch'] for r in h],[r['validation_known_accuracy'] for r in h],label='Supported accuracy',color='#147a81')
axes[2].plot([r['epoch'] for r in h],[r['validation_background_rejection'] for r in h],label='Background rejection',color='#b78a28')
epoch=load('sentences-background','frozen-config.json')['selected_epoch']
axes[2].axvline(epoch,color='#5b6570',linestyle=':',label=f'Selected epoch {epoch}')
for ax,title in zip(axes,['(a) Historical images\nValidation accuracy','(b) KU-BdSL images\nValidation accuracy','(c) Sentence model\nValidation (U7)']):
    ax.set(title=title,xlabel='Epoch',ylabel='Fraction',ylim=(-.03,1.04)); ax.grid(alpha=.2)
axes[0].legend(fontsize=7,loc='upper left'); axes[2].legend(fontsize=7,loc='lower right')
fig.tight_layout(); fig.savefig(out/'validation-curves.pdf'); fig.savefig(out/'validation-curves.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.0,2.05)); ax.set(xlim=(0,10.3),ylim=(0,2.05)); ax.axis('off')
rows=[(.99,['Hand crop','128 x 128\nOtsu binary','Frozen MobileNetV2\n+ softmax head','Letter/group\n+ editable text']),
      (.05,['Complete sign clip','Pose + hand landmarks\n32 x 104 features','GRU(64) + Dense(32)\n11-way softmax','Reject or suggest\nUser confirms caption'])]
for y,labels in rows:
    for i,label in enumerate(labels):
        x=i*2.6
        ax.add_patch(FancyBboxPatch((x+.04,y+.08),2.32,.69,boxstyle='round,pad=0.03',facecolor='#eaf3f5',edgecolor='#377886'))
        ax.text(x+1.2,y+.425,label,ha='center',va='center',fontsize=9)
        if i<3: ax.annotate('',xy=(x+2.61,y+.425),xytext=(x+2.4,y+.425),arrowprops={'arrowstyle':'->','color':'#345'})
ax.text(.06,1.95,'Separate pipelines; scores are not interchangeable',fontsize=9,weight='bold')
fig.tight_layout(pad=.3); fig.savefig(out/'pipeline.pdf'); fig.savefig(out/'pipeline.png',dpi=180); plt.close(fig)
shutil.copyfile('reports/sentences-background/confusion_matrix.png',out/'sentence-confusion.png')
matrix_path=Path('reports/sentences-background/confusion_matrix.csv')
names=matrix_path.read_text(encoding='utf-8').splitlines()[0].split(',')
matrix=np.loadtxt(matrix_path,delimiter=',',skiprows=1,dtype=int)
fig,ax=plt.subplots(figsize=(5.5,4.0)); ax.imshow(matrix,cmap='Blues')
ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,
       xlabel='Predicted sentence ID / rejected',ylabel='Actual sentence ID / background')
ax.tick_params(labelsize=8); plt.setp(ax.get_xticklabels(),rotation=50,ha='right')
for i in range(len(names)):
    for j in range(len(names)):
        if matrix[i,j]: ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>4 else 'black')
fig.tight_layout(); fig.savefig(out/'sentence-confusion.pdf'); plt.close(fig)

path=Path('paper/v2/main.tex'); text=path.read_text(encoding='utf-8')
text=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',lambda match:'' if any(r'\label{fig:'+label+'}' in match.group() for label in ['pipeline','curves','sentence-confusion']) else match.group(),text,flags=re.S)
settings=r'\renewcommand{\topfraction}{.9}\renewcommand{\textfraction}{.05}\renewcommand{\floatpagefraction}{.85}'
text=text.replace(settings,'')
# Let text and figures flow naturally; keep the long class tables together.
text=text.replace(r'\newpage','')
text=text.replace(r'\begin{document}',r'\renewcommand{\topfraction}{.9}\renewcommand{\textfraction}{.05}\renewcommand{\floatpagefraction}{.85}'+'\n'+r'\begin{document}')
text=text.replace(r'\section{Per-class metrics}',r'\newpage\section{Per-class metrics}')
pipeline=r'''\begin{figure}[!ht]\centering
\includegraphics[width=\textwidth]{figures/pipeline.pdf}
\caption{\small Inference architecture. Static classification and complete-clip sentence recognition use separate feature contracts and label spaces. The original CNN comparison is specified in the method text. Manual confirmation is an interface safeguard, not evidence of recognition accuracy.}\label{fig:pipeline}
\end{figure}
'''
text=text.replace(r'\section{Models and training setup}',pipeline+'\n'+r'\section{Models and training setup}')
curves=r'''\begin{figure}[!ht]\centering
\includegraphics[width=\textwidth]{figures/validation-curves.pdf}
\caption{\small Unsmoothed validation histories from the saved runs. Panels (a) and (b) show all 12 image-training epochs. Panel (c) shows all 80 temporal epochs and the checkpoint selected using U7 validation. Test clips are not plotted or used for checkpoint selection. Training accuracy is omitted because the historical logger records the last batch rather than an epoch aggregate.}\label{fig:curves}
\end{figure}
'''
text=text.replace(r'\section{Metrics}',curves+'\n'+r'\section{Metrics}')
temporal=r'''\begin{figure}[!ht]\centering
\includegraphics[width=.85\textwidth]{figures/sentence-confusion.pdf}
\caption{\small Raw confusion counts for 20 U19 sentence-test clips: one per supported sentence and ten background clips. Numeric IDs correspond to the published vocabulary manifest; ``other'' pools unsupported sentence types and rejected outputs. Six supported captions are correct, and eight background clips are rejected; two background clips are falsely captioned. Background types also occur in training, and a prior baseline outcome on this signer was observed.}\label{fig:sentence-confusion}
\end{figure}
'''
text=text.replace(r'\section{Limitations, ethics and accessibility}',temporal+'\n'+r'\section{Limitations, ethics and accessibility}')
path.write_text(text,encoding='utf-8')
sources=[Path('reports')/name/'history.json' for name in ['original','mobilenet','ku-mobilenet']]
sources += [Path('reports/sentences-background')/name for name in ['validation.json','frozen-config.json','confusion_matrix.csv']]
(out/'sources.json').write_text(json.dumps({str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
print('Generated pipeline, validation curves and sentence confusion figure from saved evidence')
