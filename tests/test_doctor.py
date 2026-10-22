import hashlib
import json
import zipfile
from bsl.doctor import inspect

def installation(tmp_path):
    reports=tmp_path/'reports/ku-mobilenet'; reports.mkdir(parents=True)
    models=tmp_path/'models'; models.mkdir()
    (tmp_path/'requirements.txt').write_text('',encoding='utf-8')
    model=models/'ku-mobilenet.keras'
    with zipfile.ZipFile(model,'w') as archive:
        archive.writestr('config.json',json.dumps({'config':{'layers':[{'class_name':'Dense','config':{'units':2}}]}}))
    (models/'ku-mobilenet.labels.json').write_text(json.dumps(['ক','খ']),encoding='utf-8')
    (reports/'labels.json').write_text(json.dumps([{'display':'ক'},{'display':'খ'}]),encoding='utf-8')
    (reports/'model.json').write_text(json.dumps({'file':model.name,'sha256':hashlib.sha256(model.read_bytes()).hexdigest()}),encoding='utf-8')
    return models

def test_doctor_detects_model_corruption_and_reordered_labels(tmp_path):
    models=installation(tmp_path)
    assert inspect(tmp_path)['ok']
    (models/'ku-mobilenet.labels.json').write_text(json.dumps(['খ','ক']),encoding='utf-8')
    assert not next(c for c in inspect(tmp_path)['checks'] if c['check']=='Static label contract')['ok']
    with (models/'ku-mobilenet.keras').open('ab') as stream: stream.write(b'changed')
    assert not next(c for c in inspect(tmp_path)['checks'] if c['check']=='Static model SHA256')['ok']

def test_doctor_reports_missing_dependency_without_loading_ml_runtime(tmp_path):
    installation(tmp_path)
    (tmp_path/'requirements.txt').write_text('bsl-nonexistent-package-for-test==1.0\n',encoding='utf-8')
    result=inspect(tmp_path)
    check=next(c for c in result['checks'] if c['check']=='bsl-nonexistent-package-for-test')
    assert not result['ok'] and not check['ok'] and 'missing' in check['detail']
