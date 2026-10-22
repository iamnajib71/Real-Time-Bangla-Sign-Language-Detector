"""Debounced static-letter text entry; not continuous sign-language translation."""
import numpy as np

VOWEL_SIGNS={'আ':'া','ই':'ি','উ':'ু','এ':'ে','ও':'ো'}

def render_bengali(tokens):
    text=''
    for token in tokens:
        if token==' ':
            text=text.rstrip()+' '
        elif token in VOWEL_SIGNS and text and '\u0995'<=text[-1]<='\u09b9':
            text+=VOWEL_SIGNS[token]
        else:
            text+=token
    return text.rstrip()

class TextComposer:
    def __init__(self, stable_frames=3, min_confidence=.6, word_gap_frames=12):
        self.stable_frames=stable_frames; self.min_confidence=min_confidence
        self.word_gap_frames=word_gap_frames; self.tokens=[]
        self.candidate=None; self.count=0; self.latched=False; self.blank_count=0

    @property
    def text(self):
        return render_bengali(self.tokens)

    def update(self, result=None, blank=False):
        if blank:
            self.blank_count+=1; self.candidate=None; self.count=0; self.latched=False
            if self.blank_count==self.word_gap_frames and self.tokens and self.tokens[-1]!=' ':
                self.tokens.append(' ')
            return self.text
        self.blank_count=0
        if result is None or result['confidence']<self.min_confidence or '/' in result['label']:
            self.candidate=None; self.count=0
            return self.text
        label=result['label']
        if label==self.candidate:
            self.count+=1
        else:
            self.candidate=label; self.count=1
        if self.count>=self.stable_frames and not self.latched:
            self.tokens.append(label); self.latched=True
        return self.text

    def space(self):
        if self.tokens and self.tokens[-1]!=' ':
            self.tokens.append(' ')
        self.latched=False; self.candidate=None; self.count=0

    def backspace(self):
        if self.tokens:
            self.tokens.pop()

    def clear(self):
        self.tokens.clear(); self.candidate=None; self.count=0; self.latched=False

def is_blank(image):
    """Only a uniform pause card is a delimiter; no general no-hand claim."""
    return float(np.std(image))<3.
