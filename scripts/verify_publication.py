"""Check manuscript numbers and released models against immutable prediction evidence."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paper=(root/'paper/v2/main.tex').read_text(encoding='utf-8')
numbers=json.loads((root/'paper/v2/numbers.json').read_text(encoding='utf-8'))
for name in ['original','mobilenet','ku-mobilenet']:
    path=root/'reports'/name
    m=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
    rows=json.loads((path/'predictions.json').read_text(encoding='utf-8'))
    assert len(rows)==m['test_n']
    assert abs(sum(r['correct'] for r in rows)/len(rows)-m['accuracy'])<1e-12
    for key,value in numbers[name].items(): assert m[key]==value
    assert f"{m['accuracy']*100:.2f}" in paper
    metadata=json.loads((path/'model.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root/'models'/metadata['file']).read_bytes()).hexdigest()==metadata['sha256']
temporal=json.loads((root/'reports/sentences-background/test-results.json').read_text(encoding='utf-8'))
m=temporal['test']; rows=temporal['predictions']
assert sum(r['correct'] for r in rows if r['known'])==m['known_correct']
assert sum(not r['accepted'] for r in rows if not r['known'])==m['unknown_rejected']
assert f"{m['known_correct']}/10 supported" in paper and f"{m['unknown_rejected']}/10 background" in paper
assert hashlib.sha256((root/'models/sentences-background.keras').read_bytes()).hexdigest()==temporal['model_sha256']
numbers['sentences-background']=m
(root/'paper/v2/numbers.json').write_text(json.dumps(numbers,indent=2),encoding='utf-8')
print('Verified all static and temporal manuscript counts, accuracies and model hashes')
