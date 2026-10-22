"""Local video-call caption companion; corrections and sharing stay under user control."""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
# Initialise the TensorFlow native runtime before constructing Qt widgets on Windows.
import tensorflow as tf
from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtGui import QImage,QPixmap,QFont,QFontDatabase
from PyQt5.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,
    QFileDialog,QMessageBox,QTextEdit,QComboBox,QInputDialog)
from .predict import Predictor,ROOT
from .preprocess import read_image
from .preview import hand_crop
from .text import TextComposer,is_blank
from .media import effective_fps

class CaptionWindow(QWidget):
    def __init__(self):
        super().__init__(flags=Qt.Window|Qt.WindowStaysOnTopHint)
        self.setWindowTitle('Bangla captions - share this window in your call')
        self.resize(900,180)
        layout=QVBoxLayout(self)
        self.caption=QLabel('Captions will appear here')
        self.caption.setWordWrap(True); self.caption.setFont(QFont('Noto Sans Bengali',26))
        layout.addWidget(self.caption)
        self.setStyleSheet('background:#101827;color:white;padding:18px;')

class StartingPage(QWidget):
    def __init__(self,predictor):
        super().__init__()
        QFontDatabase.addApplicationFont(str(ROOT/'assets/fonts/NotoSansBengali.ttf'))
        self.predictor=predictor; self.capture=None; self.camera_mode=False
        self.composer=TextComposer(); self.overlay=CaptionWindow()
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick)
        self.tracker=None; self.phrase_recognizer=None; self.motion_buffer=[]; self.pending_phrase=''
        self.last_motion=0.; self.updating_text=False
        self.setWindowTitle('Bangla Sign Language | Caption companion'); self.resize(1120,780)
        self.setStyleSheet('''QWidget{background:#0b1220;color:#edf2f7;font-size:15px}
            QPushButton{background:#1e3348;border:1px solid #36516c;border-radius:7px;padding:10px}
            QPushButton:hover{background:#2a4a61} QTextEdit,QComboBox{background:#152235;border:1px solid #36516c;border-radius:8px;padding:10px}
            QTextEdit{font-size:28px}
            QLabel#heading{font-size:28px;font-weight:600} QLabel#muted{color:#a5b4c8}''')
        layout=QVBoxLayout(self)
        heading=QLabel('Bangla caption companion'); heading.setObjectName('heading'); layout.addWidget(heading)
        subtitle=QLabel('Live video, readable text, and a caption window to share in an online call.'); subtitle.setObjectName('muted'); layout.addWidget(subtitle)
        toolbar=QHBoxLayout(); layout.addLayout(toolbar)
        self.mode=QComboBox(); self.mode.addItems(['Isolated letters','Personal phrases (experimental)','Sentence vocabulary (experimental)'])
        self.mode.currentIndexChanged.connect(self.change_mode); toolbar.addWidget(self.mode)
        for title,action in [('Start camera',lambda:self.open_capture(0)),('Open video',self.select_video),
                             ('Open image',self.select_image),('Stop',self.stop),('Load model',self.load_model)]:
            self.button(toolbar,title,action)
        body=QHBoxLayout(); layout.addLayout(body,1)
        left=QVBoxLayout(); body.addLayout(left,3)
        self.preview=QLabel('Start your camera or open a recorded video')
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(540,350)
        self.preview.setStyleSheet('background:#152235;border-radius:12px;border:1px solid #36516c;')
        left.addWidget(self.preview,1)
        self.result=QLabel('Letter mode: place one hand sign inside the guide.'); self.result.setWordWrap(True); left.addWidget(self.result)
        self.status=QLabel('Ready. Video processing stays on this computer.'); self.status.setObjectName('muted'); left.addWidget(self.status)
        right=QVBoxLayout(); body.addLayout(right,2)
        right.addWidget(QLabel('Conversation text - edit recognition mistakes'))
        self.transcript=QTextEdit(); self.transcript.setFont(QFont('Noto Sans Bengali',24))
        self.transcript.setPlaceholderText('Recognised letters and confirmed phrases appear here. You can also type.')
        self.transcript.textChanged.connect(self.manual_edit); right.addWidget(self.transcript,1)
        for actions in [[('Space',self.add_space),('Undo',self.undo),('Clear',self.clear_text)],
                        [('Copy',self.copy_text),('Save',self.save_text),('Caption window',self.show_overlay)],
                        [('Teach phrase from clip',self.teach_phrase),('Confirm phrase',self.confirm_phrase)],
                        [('Supported phrases',self.supported_phrases)]]:
            row=QHBoxLayout(); right.addLayout(row)
            for title,action in actions: self.button(row,title,action)
        note=QLabel('Phrase suggestions require confirmation. The phrase recognizer is experimental; scores are uncalibrated.')
        note.setWordWrap(True); note.setObjectName('muted'); right.addWidget(note)

    @staticmethod
    def button(layout,title,action):
        button=QPushButton(title); button.clicked.connect(action); layout.addWidget(button); return button

    def set_text(self,text):
        self.updating_text=True; self.transcript.setPlainText(text); self.updating_text=False
        self.overlay.caption.setText(text[-250:] or 'Captions will appear here')

    def manual_edit(self):
        if not self.updating_text: self.composer.tokens=list(self.transcript.toPlainText())
        self.overlay.caption.setText(self.transcript.toPlainText()[-250:] or 'Captions will appear here')

    def show_frame(self,frame):
        box=None
        if self.mode.currentIndex()>=1:
            self.process_motion(frame)
        else:
            sample,box=hand_crop(frame) if self.camera_mode else (frame,None)
            result=self.predictor(sample); before=self.composer.text
            self.composer.update(result,blank=is_blank(sample))
            if self.composer.text!=before: self.set_text(self.composer.text)
            self.result.setText(f"{result['label']}   {result['confidence']:.1%} (uncalibrated score)")
        frame=frame.copy()
        if box:
            x,y,w,h=box; cv2.rectangle(frame,(x,y),(x+w,y+h),(70,220,170),2)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,_=rgb.shape
        image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def process_motion(self,frame):
        if self.tracker is None or self.phrase_recognizer is None:
            self.result.setText('Teach a personal phrase from a labelled clip to enable suggestions.'); return
        now=time.monotonic()
        if now-self.last_motion<.125: return
        self.last_motion=now
        features,valid=self.tracker.extract(frame,int(now*1000))
        if not valid: self.result.setText('No usable upper-body tracking'); return
        self.motion_buffer.append(features); self.motion_buffer=self.motion_buffer[-192:]
        if len(self.motion_buffer)<16: return
        result=self.phrase_recognizer(np.asarray(self.motion_buffer))
        self.pending_phrase=result['label'] if result['accepted'] else ''
        self.result.setText(('Suggestion: '+self.pending_phrase+' - confirm to add') if self.pending_phrase else 'Unrecognised - keep signing or type a correction')

    def change_mode(self):
        self.motion_buffer=[]; self.pending_phrase=''
        self.phrase_recognizer=None
        if self.mode.currentIndex()>=1:
            try:
                from .motion import LandmarkTracker
                from .phrase import PhraseRecognizer
                if self.mode.currentIndex()==2:
                    from .sentence import SentenceRecognizer
                    self.phrase_recognizer=SentenceRecognizer(ROOT/'models')
                else:
                    path=ROOT/'models/personal-phrases.npz'
                    if path.exists(): self.phrase_recognizer=PhraseRecognizer(path)
                if self.phrase_recognizer is not None and self.tracker is None:
                    self.tracker=LandmarkTracker(ROOT/'models')
            except (ImportError,FileNotFoundError,RuntimeError) as error:
                QMessageBox.warning(self,'Phrase setup',f'Install requirements-motion.txt and run scripts/download_motion_models.py.\n{error}')

    def teach_phrase(self):
        self.stop()
        path,_=QFileDialog.getOpenFileName(self,'Choose a short clip containing one phrase','','Videos (*.mp4 *.avi *.webm *.mov)')
        if not path: return
        label,ok=QInputDialog.getText(self,'Phrase meaning','Type the Bengali meaning of the complete clip:')
        if not ok or not label.strip(): return
        try:
            from .motion import extract_video
            from .phrase import resample,PhraseRecognizer
            capture=cv2.VideoCapture(path)
            duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/max(capture.get(cv2.CAP_PROP_FPS),1)
            capture.release()
            if not 1<=duration<=20: raise ValueError('Choose a trimmed 1-20 second clip of one complete phrase.')
            _,features,valid=extract_video(path,ROOT/'models')
            if valid.sum()<8: raise ValueError('Not enough tracked signing frames. Use a clear upper-body clip.')
            bank=ROOT/'models/personal-phrases.npz'; new=resample(features[valid])[None]; labels=[label.strip()]
            if bank.exists():
                old=np.load(bank,allow_pickle=False); new=np.concatenate([old['templates'],new]); labels=list(old['labels'])+labels
            bank.parent.mkdir(exist_ok=True); np.savez_compressed(bank,templates=new,labels=np.array(labels))
            self.phrase_recognizer=PhraseRecognizer(bank); self.mode.setCurrentIndex(1); self.change_mode()
            self.status.setText('Personal phrase saved locally. Test on separate recordings before relying on it.')
        except (ImportError,ValueError,FileNotFoundError,RuntimeError) as error:
            QMessageBox.warning(self,'Could not teach phrase',str(error))

    def confirm_phrase(self):
        if self.pending_phrase:
            self.set_text((self.transcript.toPlainText().rstrip()+' '+self.pending_phrase).strip())
            self.composer.tokens=list(self.transcript.toPlainText()); self.pending_phrase=''; self.motion_buffer=[]

    def supported_phrases(self):
        recognizer=self.phrase_recognizer
        if recognizer is None: text='Choose a phrase model or teach a phrase first.'
        elif hasattr(recognizer,'config'): text='\n'.join(recognizer.config['labels'])
        else: text='\n'.join(sorted(set(recognizer.labels)))
        QMessageBox.information(self,'Supported phrase vocabulary',text)

    def load_model(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose model with matching label map','','Keras model (*.keras)')
        if path:
            try: self.predictor=Predictor(path); self.status.setText('Model loaded: '+Path(path).name)
            except (ValueError,FileNotFoundError) as error: QMessageBox.warning(self,'Model error',str(error))

    def select_image(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose hand crop','','Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            self.stop()
            try: self.show_frame(read_image(path))
            except ValueError as error: QMessageBox.warning(self,'Input error',str(error))

    def select_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose video','','Videos (*.avi *.mp4 *.webm *.mov)')
        if path: self.open_capture(path)

    def open_capture(self,source):
        self.stop(); self.camera_mode=isinstance(source,int); self.capture=cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.stop(); QMessageBox.warning(self,'Input unavailable','Cannot open camera or video. Check the device, permissions and file.'); return
        self.status.setText('Camera running - fit the sign inside the guide.' if self.camera_mode else 'Processing recorded video locally.')
        fps=effective_fps(self.capture.get(cv2.CAP_PROP_FPS))
        self.timer.start(100 if self.camera_mode else max(10,round(1000/fps)))

    def tick(self):
        ok,frame=self.capture.read()
        if ok: self.show_frame(frame)
        else: self.stop()

    def stop(self):
        self.timer.stop()
        if self.capture is not None: self.capture.release(); self.capture=None
        self.camera_mode=False

    def add_space(self): self.composer.space(); self.set_text(self.composer.text)
    def undo(self): self.composer.backspace(); self.set_text(self.composer.text)
    def clear_text(self): self.composer.clear(); self.set_text('')
    def copy_text(self): QApplication.clipboard().setText(self.transcript.toPlainText())
    def show_overlay(self): self.overlay.show(); self.overlay.raise_()
    def save_text(self):
        path,_=QFileDialog.getSaveFileName(self,'Save transcript','','Text (*.txt)')
        if path: Path(path).write_text(self.transcript.toPlainText(),encoding='utf-8')

    def closeEvent(self,event):
        self.stop(); self.overlay.close()
        if self.tracker is not None: self.tracker.close()
        event.accept()

def launch(model,labels=None,mode='letters',camera=None):
    app=QApplication.instance() or QApplication(sys.argv)
    window=StartingPage(Predictor(model,labels))
    window.mode.setCurrentIndex({'letters':0,'personal':1,'sentences':2}[mode]); window.show()
    if camera is not None: QTimer.singleShot(0,lambda:window.open_capture(camera))
    return app.exec_()
