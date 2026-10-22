import json
from pathlib import Path
import numpy as np
from .preprocess import preprocess, read_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "ku-mobilenet.keras"

class Predictor:
    def __init__(self, model_path=DEFAULT_MODEL, labels_path=None, model=None, labels=None):
        if model is None:
            from tensorflow.keras.models import load_model
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model missing: {model_path}. Download the v2.0.0 release assets or run training.")
            model = load_model(model_path, compile=False)
        self.model = model
        self.labels = labels if labels is not None else json.loads(
            Path(labels_path or Path(model_path).with_suffix('.labels.json')).read_text(encoding='utf-8'))
        if int(model.output_shape[-1]) != len(self.labels):
            raise ValueError("Model outputs and label count disagree")

    def __call__(self, image):
        image = read_image(image) if isinstance(image, (str, Path)) else image
        scores = np.asarray(self.model(preprocess(image)[None], training=False))[0]
        if scores.shape != (len(self.labels),) or not np.isfinite(scores).all():
            raise ValueError("Invalid model output")
        index = int(np.argmax(scores))
        return {"index": index, "label": self.labels[index], "confidence": float(scores[index])}
