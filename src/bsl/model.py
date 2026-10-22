from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def original_cnn(classes):
    """Architecture transcribed from ishara.py at v2022-thesis."""
    return Sequential([
        Input((128, 128, 1)), Conv2D(32, 5, activation='relu'),
        Conv2D(32, 5, activation='relu'), MaxPooling2D(5), Dropout(.25),
        Conv2D(64, 3, activation='relu'), Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2), Dropout(.25), Flatten(), Dense(512, activation='relu'),
        Dropout(.5), Dense(classes, activation='softmax')])

def mobilenet_baseline(classes):
    """Frozen ImageNet MobileNetV2; no claim that it is a state-of-the-art model."""
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Concatenate, Rescaling, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    inputs = Input((128,128,1))
    rgb = Concatenate()([inputs, inputs, inputs])
    rgb = Rescaling(2.0, offset=-1.0)(rgb)
    backbone = MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
    backbone.trainable = False
    features = GlobalAveragePooling2D()(backbone(rgb, training=False))
    outputs = Dense(classes, activation='softmax')(Dropout(.2)(features))
    return Model(inputs, outputs)
