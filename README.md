# Real-Time Bangla Sign Language Detector

B.Sc. thesis project (North South University, 2022): a desktop application that recognises Bangla Sign Language hand gestures from a webcam in real time and shows the matching Bangla character.

## How it works

1. **Pre-processing (OpenCV):** frames are converted to greyscale and binarised with Otsu thresholding (`OtsuCOnvert.py`) to isolate the hand shape, then resized to 128 × 128.
2. **Model (Keras CNN):** two convolution blocks (32 and 64 filters) with max-pooling and dropout, then a 512-unit dense layer and a 36-class softmax (`ishara.py`). It was trained with data augmentation (shear, zoom and horizontal flip) through `ImageDataGenerator`.
3. **App (PyQt5):** `StartingPage.py` loads the trained model (`model1.h5`), captures frames, predicts the gesture and renders the Bangla character on screen. It can also classify a single image chosen from disk.

## Stack

Python · OpenCV · Keras/TensorFlow · NumPy · PyQt5 · Pillow

## Run

```bash
pip install opencv-python tensorflow numpy pillow pyqt5 h5py
python StartingPage.py
```

This code was written in 2022 against older Keras APIs (`keras.layers.normalization`, `np_utils`). On a current TensorFlow install, change those imports to `tensorflow.keras` equivalents. `StartingPage.py` also loads the model and font from absolute `F:/NSU/...` paths, so point those at this folder before running.

## Paper

Hassan, N. *Bangla Sign Language Gesture Recognition System*, ScienceOpen Preprints, 2022 (non-peer-reviewed). DOI [10.14293/S2199-1006.1.SOR-.PPUF56Q.v1](https://doi.org/10.14293/S2199-1006.1.SOR-.PPUF56Q.v1)
