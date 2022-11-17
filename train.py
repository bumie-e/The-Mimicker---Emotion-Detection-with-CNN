import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.applications.efficientnet import EfficientNetB0
from keras.preprocessing.image import load_img
import tensorflow as tf


validation_dir = r'/kaggle/input/ferdata/test'
train_dir = r'/kaggle/input/ferdata/train'
picture_size = 48

no_of_classes = 7
batch_size  = 128

datagen_train  = ImageDataGenerator()
datagen_val = ImageDataGenerator()

train_set = datagen_train.flow_from_directory(train_dir,
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)


test_set = datagen_val.flow_from_directory(validation_dir,
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)


model = keras.Sequential([
# Data Augmentation 
  

  EfficientNetB0(weights='imagenet',include_top=False, classes=no_of_classes, input_shape=(48,48,3)), #Extract features 

  layers.BatchNormalization(),
#CNN layer 

  layers.Conv2D(256,(3,3),padding = 'same'), 
  layers.BatchNormalization(),
  layers.LeakyReLU(alpha=0.1),
  layers.Conv2D(512,(3,3),padding = 'same'), 
  layers.BatchNormalization(),
  layers.LeakyReLU(alpha=0.1),
  layers.MaxPooling2D(pool_size = (2,2),padding = 'same'), 
  layers.Dropout(0.25), 


  layers.Flatten(),
#Fully connected 1st layer 

  layers.Dense(256),
  layers.BatchNormalization(), 
  layers.Activation('relu'), 
  layers.Dropout(0.25), 

# Fully connected layer 2nd layer

  layers.Dense(512),
  layers.BatchNormalization(), 
  layers.Activation('relu'), 
  layers.Dropout(0.25), 


  layers.Dense(no_of_classes, activation='softmax'),
                          
])
print(model.summary())

early_stopping = EarlyStopping(
    min_delta=0.001, 
    patience=3, 
    restore_best_weights=True,
)

epochs = 25

model.compile(loss='categorical_crossentropy',
              optimizer = 'Adam',
              metrics=['accuracy'])

history = model.fit(train_set,
                                steps_per_epoch=train_set.n//train_set.batch_size,
                                epochs=epochs,
                                validation_data = test_set,
                                validation_steps = test_set.n//test_set.batch_size,
                                callbacks=[early_stopping],
                                )

#if 'name' == ' __main__':

    