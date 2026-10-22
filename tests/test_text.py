from bsl.text import TextComposer,render_bengali
from bsl.preview import hand_crop
import numpy as np

def test_bengali_vowel_composition():
    assert render_bengali(['আ','ম','ই',' ','আ','স','ই'])=='আমি আসি'

def test_stability_repeats_spaces_and_corrections():
    composer=TextComposer(stable_frames=3)
    result={'label':'ম','confidence':.9}
    for _ in range(10): composer.update(result)
    assert composer.text=='ম'
    composer.update(blank=True)
    for _ in range(3): composer.update(result)
    assert composer.text=='মম'
    composer.space()
    for _ in range(3): composer.update({'label':'আ','confidence':.9})
    assert composer.text=='মম আ'
    composer.backspace(); assert composer.text=='মম'
    composer.clear(); assert composer.text==''

def test_ambiguous_and_low_scores_do_not_emit_text():
    composer=TextComposer()
    for _ in range(10): composer.update({'label':'ব/ভ','confidence':.99})
    for _ in range(10): composer.update({'label':'ম','confidence':.1})
    assert composer.text==''

def test_camera_crop_guide():
    frame=np.zeros((480,640,3),dtype=np.uint8)
    crop,box=hand_crop(frame)
    x,y,w,h=box
    assert crop.shape==(h,w,3)
    assert x>=0 and y>=0 and x+w<=640 and y+h<=480
