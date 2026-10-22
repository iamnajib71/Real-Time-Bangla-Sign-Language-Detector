from functools import lru_cache
from pathlib import Path
import unicodedata
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def hand_crop(frame, fraction=.65):
    h,w=frame.shape[:2]; side=max(1,int(min(h,w)*fraction))
    x=(w-side)//2; y=(h-side)//2
    return frame[y:y+side,x:x+side],(x,y,side,side)

@lru_cache(maxsize=1)
def bengali_font():
    path=Path(__file__).resolve().parents[2]/'assets/fonts/NotoSansBengali.ttf'
    return ImageFont.truetype(str(path),32) if path.exists() else ImageFont.load_default(size=24)

def annotate(frame,result,text='',box=None):
    canvas=frame.copy()
    if box:
        x,y,w,h=box
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(70,220,170),2)
    rgb=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(rgb)
    width,height=rgb.size
    draw.rectangle((0,max(0,height-110),width,height),fill='#0b1220')
    label=result['label']; label=('◌'+label) if label and unicodedata.category(label[0]).startswith('M') else label
    draw.text((10,height-100),f"{label}  {result['confidence']:.1%}",font=bengali_font(),fill='#5eead4',anchor='lt')
    draw.text((10,height-56),text,font=bengali_font(),fill='white',anchor='lt')
    return cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR)
