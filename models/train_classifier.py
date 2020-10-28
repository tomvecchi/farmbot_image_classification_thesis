""" 
    Created by Tom Vecchi for FarmBot Thesis 2020
    
    This file contains code for training the classifier model. Data and the finished model are not included.

    Citations: Rizwan, M (2018). AlexNet implementation using Keras. Retrieved 7/10/20 from https://engmrk.com/alexnet-implementation-using-keras/
    Additionally, some sections are partially based off my own previous work in ELEC4630 (Image Processing and COmputer Vision).
"""

import keras
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger

TRAINING_LOGS_FILE = "logs.csv"

IMAGE_SIZE = 40
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 30
NUM_CLASSES = 3

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# Alexnet CNN Model based on code from https://engmrk.com/alexnet-implementation-using-keras/
# Modifed by me to run faster & prevent overfitting
import numpy as np
np.random.seed(1000)

#Instantiate an empty model
model = Sequential()
model.add(BatchNormalization(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3),))

model.add(Conv2D(filters=24,  kernel_size=(7,7), strides=(4,4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

model.add(Flatten()) # Switch to fully-connected layers

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES)) # 3 outputs
model.add(Activation('softmax'))

# Compilation
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=Adam(), 
              metrics=["accuracy"])


# Data
path = "."
training_dir = path + os.path.sep + "training" 
validation_dir = path + os.path.sep + "validation"

# Data augmentation for training dataset
training_datagen = ImageDataGenerator(
    #rescale=1./255, # I think this breaks it?
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
training_data = training_datagen.flow_from_directory(
    training_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")

# Validation data generator
validation_datagen = ImageDataGenerator(
    #rescale=1./255
    )
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")


# Perform training
model.fit_generator(
    training_data,
    steps_per_epoch=len(training_data.filenames) // BATCH_SIZE, #Done to speed it up
    #steps_per_epoch = 10,
    epochs=EPOCHS,
    validation_data=validation_data,
    validation_steps=len(validation_data.filenames) // BATCH_SIZE,
    #validation_steps=5, 
    callbacks=[CSVLogger(TRAINING_LOGS_FILE,
                          append=False,
                          separator=";")], 
    verbose=1)

model.save("plant_class_adam_bn_80x32.h5")


# Predict image using just-trained model
files = np.random.choice(os.listdir(validation_dir), 5)

model = load_model("./plant_class_adam_bn_80x32.h5")

for file in files:     
    img = cv2.imread(validation_dir + os.path.sep + file) #Test image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (40,40))
    img = np.array(img)
    img = np.expand_dims(img, 0)

    pred = model.predict(img)

    pred_list = max(pred).tolist()
    pred_class = pred_list.index(max(pred_list))
    print(pred_class, file)