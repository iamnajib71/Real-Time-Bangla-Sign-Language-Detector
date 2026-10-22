"""Recorded-file sampling follows media time, independent of GUI playback speed."""
import numpy as np
from bsl.gui import QApplication,StartingPage,QFileDialog
from pathlib import Path

def test_uploaded_video_uses_media_clock_and_resets_old_phrase(monkeypatch):
    app=QApplication.instance() or QApplication([])
    window=StartingPage(lambda frame:{'label':'আ','confidence':.9})
    path=str(Path(__file__).parent/'fixtures/bimodal.avi')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(path,''))
    window.pending_phrase='stale'; window.motion_buffer=[np.zeros(98)]
    window.select_video()
    assert window.pending_phrase=='' and window.motion_buffer==[]
    seen=[]
    window.show_frame=lambda frame:seen.append(window.file_time)
    while window.capture is not None: window.tick()
    assert len(seen)==8
    assert np.allclose(seen,np.arange(8)/window.source_fps)
    assert window.file_time is None
    window.close()

def test_tracking_loss_clears_a_stale_caption_suggestion():
    app=QApplication.instance() or QApplication([])
    window=StartingPage(lambda frame:{'label':'আ','confidence':.9})
    class Tracker:
        def extract(self,frame,timestamp): return np.zeros(98,np.float32),False
        def close(self): pass
    window.tracker=Tracker()
    window.phrase_recognizer=lambda sequence:{'label':'stale','accepted':True}
    window.pending_phrase='stale'; window.file_time=0.
    window.process_motion(np.zeros((32,32,3),np.uint8))
    window.confirm_phrase()
    assert window.pending_phrase=='' and window.transcript.toPlainText()==''
    window.close()

def test_motion_sampling_uses_file_timestamps_when_playback_is_fast():
    app=QApplication.instance() or QApplication([])
    window=StartingPage(lambda frame:{'label':'আ','confidence':.9})
    timestamps=[]
    class Tracker:
        def extract(self,frame,timestamp):
            timestamps.append(timestamp)
            return np.zeros(98,np.float32),True
        def close(self): pass
    window.tracker=Tracker()
    window.phrase_recognizer=lambda sequence:{'label':'Unrecognised','accepted':False}
    frame=np.zeros((32,32,3),np.uint8)
    for i in range(60):
        window.file_time=i/30
        window.process_motion(frame)
    assert len(timestamps)==16
    assert timestamps[0]==0 and timestamps[-1]==1900
    window.close()
