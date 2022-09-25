# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
model = Sequential()

#model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1	),strides=(1,1), activation = 'relu'))
model.add(Conv2D(32, (5, 5), input_shape=(128, 128, 1 ),strides=(1,1), activation = 'relu'))
model.add(Conv2D(32, (5, 5),strides=(1,1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(5,5)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3, 3),strides=(1,1), activation = 'relu'))
model.add(Conv2D(64,(3, 3),strides=(1,1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(36, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
#                          height_shift_range=0.08, zoom_range=0.08)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Otsu_again/train_set',
        target_size=(128, 128),
        batch_size=18,
        class_mode='categorical',  color_mode = 'grayscale')
#training_set=training_set.reshape(training_set.shape[0], 64, 64, 1)
test_set = test_datagen.flow_from_directory(
        'Otsu_again/test_set',
        target_size=(128, 128),
        batch_size=8,
        class_mode='categorical', color_mode = 'grayscale')
#test_set.reshape(test_set.shape[0], 64, 64, 1)
# model.fit_generator(training_set, steps_per_epoch=1005//36, epochs=5, 
#                     validation_data=test_set, validation_steps=300//36)
classiFier=model.fit_generator(
        training_set,
        steps_per_epoch=713//18,
        epochs=12,
        validation_data = test_set,
        validation_steps = 292//8
      )
import h5py
model.save('modelAkib.h5')

print(classiFier.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(classiFier.history['acc'])
plt.plot(classiFier.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(classiFier.history['loss'])
plt.plot(classiFier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



