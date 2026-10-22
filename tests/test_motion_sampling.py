import numpy as np
from bsl import motion

def test_implausible_video_fps_does_not_drop_signing_frames(monkeypatch):
    closed=[]
    class Capture:
        def __init__(self,*args): self.index=0
        def isOpened(self): return True
        def get(self,property): return 1000.
        def read(self):
            self.index+=1
            return (True,np.zeros((32,32,3),np.uint8)) if self.index<=32 else (False,None)
        def release(self): closed.append('video')
    class Tracker:
        def __init__(self,*args): pass
        def extract(self,frame,timestamp): return np.zeros(98,np.float32),True
        def close(self): closed.append('tracker')
    monkeypatch.setattr(motion.cv2,'VideoCapture',Capture)
    monkeypatch.setattr(motion,'LandmarkTracker',Tracker)
    times,features,visible=motion.extract_video('fixture-with-bad-metadata.mp4',fps=8)
    assert len(times)>=8 and features.shape[1]==98 and visible.all()
    assert set(closed)=={'video','tracker'}
