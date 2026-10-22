"""Render the actual desktop app with a held-out image for visual verification."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from bsl.gui import QApplication,StartingPage
from bsl.predict import Predictor
from bsl.preprocess import read_image
import json

app=QApplication.instance() or QApplication([])
window=StartingPage(Predictor('models/ku-mobilenet.keras'))
rows=json.loads(Path('reports/ku-mobilenet/predictions.json').read_text(encoding='utf-8'))
row=next(r for r in rows if r['correct'] and r['actual']=='ম')
window.show(); app.processEvents()
window.show_frame(read_image(Path('data/external/ku-bdsl-usld')/row['path']))
window.set_text('আমি আসি')
window.status.setText('Recorded image preview | editable transcript and caption sharing controls')
app.processEvents()
Path('assets/app').mkdir(parents=True,exist_ok=True)
window.grab().save('assets/app/caption-companion.png')
window.close()
