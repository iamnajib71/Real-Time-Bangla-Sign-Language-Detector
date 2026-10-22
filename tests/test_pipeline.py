import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import pytest
from bsl.preprocess import otsu, preprocess, read_image
from bsl.predict import Predictor
from bsl.media import frames, predictions

FIXTURES = Path(__file__).parent/'fixtures'

def test_otsu_bimodal_pixels():
    image = np.array([[10,10,200,200],[10,10,200,200]],dtype=np.uint8)
    assert np.array_equal(otsu(image), [[0,0,255,255],[0,0,255,255]])

def test_resize_normalize_and_channels():
    result = preprocess(read_image(FIXTURES/'dark.png'))
    assert result.shape == (128,128,1)
    assert result.dtype == np.float32
    assert set(np.unique(result)) <= {0.,1.}

def test_bad_image(tmp_path):
    path=tmp_path/'bad.png'; path.write_text('invalid')
    with pytest.raises(ValueError, match='decode'):
        read_image(path)

@pytest.fixture
def fixture_model(tmp_path):
    """Known deterministic test classifier, never used as research evidence."""
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    model=Sequential([Input((128,128,1)),GlobalAveragePooling2D(),Dense(2,activation='softmax')])
    model.layers[-1].set_weights([np.array([[-8.,8.]],dtype=np.float32),np.array([4.,-4.],dtype=np.float32)])
    path=tmp_path/'fixture.keras'; model.save(path)
    path.with_suffix('.labels.json').write_text(json.dumps(['অ','আ']),encoding='utf-8')
    return path

def test_prediction_real_keras_fixture(fixture_model):
    predictor=Predictor(fixture_model)
    dark=predictor(FIXTURES/'dark.png'); bright=predictor(FIXTURES/'bright.png')
    assert dark['label']=='অ'
    assert bright['label']=='আ'
    assert 0.9 < dark['confidence'] <= 1

def test_fixture_video_end_to_end_cli(fixture_model,tmp_path):
    output=tmp_path/'predictions.jsonl'
    subprocess.run([sys.executable,'-m','bsl','webcam','--video',str(FIXTURES/'bimodal.avi'),
        '--model',str(fixture_model),'--output',str(output)],check=True,timeout=300)
    rows=[json.loads(x) for x in output.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==8
    assert [r['label'] for r in rows]==['অ']*4+['আ']*4
    assert [r['source'] for r in rows]==list(map(str,range(8)))

def test_images_use_same_pipeline(fixture_model):
    predictor=Predictor(fixture_model)
    rows=list(predictions(predictor,images=FIXTURES))
    assert len(rows)==2
    assert {r['label'] for r in rows}=={'অ','আ'}

def test_video_errors():
    with pytest.raises(ValueError, match='Cannot open'):
        list(frames(video='no-such-video.avi'))

def test_pyqt_image_and_recorded_video(fixture_model):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PyQt5.QtWidgets import QApplication
    from bsl.gui import StartingPage
    app=QApplication.instance() or QApplication([])
    window=StartingPage(Predictor(fixture_model))
    window.show_frame(read_image(FIXTURES/'bright.png'))
    assert 'আ' in window.result.text()
    window.open_capture(str(FIXTURES/'bimodal.avi'))
    for _ in range(10):
        if window.capture is not None:
            window.tick()
    assert window.capture is None
    window.close()

def test_split_no_content_leakage():
    from bsl.train import split_records
    records=[{'class':str(c),'path':f'{c}/{i}.png','sha256_preprocessed':f'{c}-{i}'} for c in range(2) for i in range(20)]
    split=split_records(records)
    assert split==split_records(records)
    assert {r['split'] for r in split}=={'train','test','validation'}
    assert len({r['sha256_preprocessed'] for r in split})==len(split)
