"""One pipeline for images, files and camera frames; file inputs need no display."""
from pathlib import Path
import math
import cv2
from .preprocess import read_image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def effective_fps(value):
    value=float(value or 0)
    return value if math.isfinite(value) and 0<value<=120 else 30.

def frames(video=None, images=None, camera=0):
    if images is not None:
        paths = sorted(p for p in Path(images).rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f"No images in {images}")
        for path in paths:
            yield str(path), read_image(path)
        return
    capture = cv2.VideoCapture(str(video) if video is not None else camera)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open input: {video if video is not None else camera}")
    count = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield str(count), image
            count += 1
        if count == 0:
            raise ValueError("Input contains no decoded frames")
    finally:
        capture.release()

def predictions(predictor, **kwargs):
    for source, image in frames(**kwargs):
        yield {"source": source, **predictor(image)}
