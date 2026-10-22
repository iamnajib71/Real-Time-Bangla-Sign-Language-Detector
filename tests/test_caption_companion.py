import os
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from bsl.gui import QApplication,StartingPage

def test_manual_corrections_and_caption_window():
    app=QApplication.instance() or QApplication([])
    window=StartingPage(lambda frame:{'label':'ম','confidence':.9})
    window.transcript.setPlainText('সাহায্য করুন')
    assert window.overlay.caption.text()=='সাহায্য করুন'
    assert window.composer.text=='সাহায্য করুন'
    window.clear_text(); assert window.transcript.toPlainText()==''
    window.close()

def test_phrase_suggestions_need_confirmation(monkeypatch):
    app=QApplication.instance() or QApplication([])
    window=StartingPage(lambda frame:{'label':'ম','confidence':.9})
    window.mode.setCurrentIndex(1)
    class Tracker:
        def extract(self,frame,timestamp): return np.zeros(98,np.float32),True
        def close(self): pass
    window.tracker=Tracker()
    window.phrase_recognizer=lambda frames:{'label':'হ্যালো','accepted':True}
    clock=iter(np.arange(100,110,.2))
    monkeypatch.setattr('bsl.gui.time.monotonic',lambda:float(next(clock)))
    for _ in range(18): window.show_frame(np.zeros((48,64,3),np.uint8))
    assert window.pending_phrase=='হ্যালো'
    assert window.transcript.toPlainText()==''
    window.confirm_phrase()
    assert window.transcript.toPlainText()=='হ্যালো'
    window.close()
