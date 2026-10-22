"""Check the released model, label map and predictions agree with the saved audit."""
import argparse
import json
from pathlib import Path
from bsl.predict import Predictor

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',default='ku-mobilenet')
parser.add_argument('--data',default='data/external/ku-bdsl-usld')
args=parser.parse_args()
report=Path('reports')/args.experiment
rows=json.loads((report/'predictions.json').read_text(encoding='utf-8'))
predictor=Predictor(Path('models')/(args.experiment+'.keras'))
correct=0; maximum_score_difference=0.
for row in rows:
    measured=predictor(Path(args.data)/row['path'])
    if measured['index']!=row['predicted_index'] or measured['label']!=row['predicted']:
        raise ValueError('Exported prediction differs: '+row['path'])
    maximum_score_difference=max(maximum_score_difference,abs(measured['confidence']-row['confidence']))
    correct+=measured['index']==row['actual_index']
result={'test_n':len(rows),'correct':correct,'accuracy':correct/len(rows),
        'every_label_matches':True,'maximum_score_difference':maximum_score_difference}
if maximum_score_difference>1e-4:
    raise ValueError('Exported scores differ beyond float tolerance')
(report/'export-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
