"""Check manuscript numbers and released models against immutable prediction evidence."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
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
for label in ['pipeline','curves','sentence-confusion']:
    assert paper.count(r'\label{fig:'+label+'}')==1,label
for name,digest in json.loads((root/'paper/v2/figures/sources.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((root/'models/sentences-background.keras').read_bytes()).hexdigest()==temporal['model_sha256']
numbers['sentences-background']=m
(root/'paper/v2/numbers.json').write_text(json.dumps(numbers,indent=2),encoding='utf-8')
ci=root/'reports/ci-tests.xml'
if ci.exists():
    suites=list(ET.parse(ci).getroot())
    assert not sum(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites)
    count=sum(int(s.get('tests',0)) for s in suites)
    for file in ['README.md','docs/RESULTS.md','docs/RELEASE.md']:
        path=root/file; text=path.read_text(encoding='utf-8')
        text=re.sub(r'\d+ (?:software|automated) tests passed (?:locally|in (?:local verification|CPU CI))',f'{count} automated tests passed in CPU CI',text)
        text=re.sub(r'\d+ passing (?:local|automated) tests(?: in (?:local verification|CPU CI))?',f'{count} passing automated tests in CPU CI',text)
        if file=='docs/RESULTS.md': text=text.replace('`reports/tests.xml`','`reports/ci-tests.xml`')
        path.write_text(text,encoding='utf-8')
    print(f'Verified {count} passing CPU CI tests against the saved JUnit artifact')
print('Verified all static and temporal manuscript counts, accuracies and model hashes')
