import cv2
import numpy as np
from bsl import __main__ as cli

def test_camera_cli_opens_guide_and_quits(monkeypatch,capsys):
    samples=[]; shown=[]; closed=[]
    class Predictor:
        def __init__(self,*args,**kwargs): pass
        def __call__(self,image):
            samples.append(image.shape)
            return {'label':'ম','confidence':.9,'index':0}
    frame=np.zeros((480,640,3),np.uint8)
    monkeypatch.setattr(cli,'Predictor',Predictor)
    monkeypatch.setattr(cli,'frames',lambda **kw:iter([('0',frame),('1',frame)]))
    monkeypatch.setattr(cv2,'imshow',lambda name,image:shown.append(image.shape))
    monkeypatch.setattr(cv2,'waitKey',lambda delay:ord('q'))
    monkeypatch.setattr(cv2,'destroyAllWindows',lambda:closed.append(True))
    cli.main(['webcam'])
    assert samples==[(312,312,3)]
    assert shown==[(480,640,3)]
    assert closed==[True]
    assert '0.9' in capsys.readouterr().out
