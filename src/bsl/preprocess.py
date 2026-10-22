from pathlib import Path
import cv2
import numpy as np

SIZE = 128

def read_image(path):
    """Decode Unicode paths on Windows as well as POSIX."""
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image

def otsu(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 pixels")
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def preprocess(image):
    """Resize, Otsu binarize, normalize; same contract in training and inference."""
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (otsu(image).astype(np.float32) / 255.0)[..., None]
