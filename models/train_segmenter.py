"""
    Created by Tom Vecchi for 2020 FarmBot Thesis

    This file contains the model and training code for the segmentation algorithm. The finished model and training data aren't included.

    Citations: zhixuhao. “unet”. github.com. https://github.com/zhixuhao/unet/blob/master/model.py (retrieved 14/7/20).
            Additionally, some sections are partially based off my own previous work in ELEC4630 (Image Processing and Computer Vision).

"""

import keras
import skimage.exposure
import cv2
#from google.colab.patches import cv2_imshow # Only if using Google Colab environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, GlobalMaxPooling2D
from keras.layers import Input, UpSampling2D, Reshape, add, concatenate
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger, ProgbarLogger


HEIGHT = 256
WIDTH  = 256
CHANNELS = 3
BATCH_SIZE = 10

TRAINING_DATA_DIR = "./training"
TRAINING_MASKS_DIR = "./masks"

OUTPUT_FILE = "./training_output.csv"

#Use U-Net architecture to perform segmentation
# Citation: This architecture is based on the model used by https://github.com/zhixuhao/unet/blob/master/model.py
inputs = Input((HEIGHT, WIDTH, CHANNELS))

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
reshape= Reshape((-1,1))(conv10)
act = Activation('relu')(reshape)
model = Model(inputs=inputs, outputs=act)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.summary()



# Data generation- uses data augmentation functionality of Keras to generate randomised image/mask pairs as needed-
# this way, even though there's only like 200 images, it should still get plenty of data
def training_generator():
  seed = np.random.randint(1,10000) #Random, but transformations will be the same for both generators

  augmentations = ImageDataGenerator(rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='reflect')
  
  image_generator = augmentations.flow_from_directory(
      TRAINING_DATA_DIR,
      classes=["seg_training"],
      target_size=(HEIGHT, WIDTH),
      color_mode="rgb",
      class_mode=None,
      batch_size=BATCH_SIZE,
      shuffle=True,
      seed=seed
  )

  mask_generator = augmentations.flow_from_directory(
      TRAINING_MASKS_DIR,
      classes=["seg_masks"],
      target_size=(HEIGHT,WIDTH),
      color_mode="grayscale",
      class_mode=None,
      batch_size=BATCH_SIZE,
      shuffle=True,
      seed=seed
  )

  # The following part continually generates batch after batch of randomised training image/mask pairs
  while(1):
        imgs = image_generator.next()
        masks = mask_generator.next()
        masks = masks.reshape(BATCH_SIZE, HEIGHT*WIDTH, 1)

        yield imgs, masks

# As training_generator, but for validation. 
def val_generator():
  seed = np.random.randint(1,10000)

  augmentations = ImageDataGenerator() # No transformations- different images to training set
  
  image_generator = augmentations.flow_from_directory(
      TRAINING_DATA_DIR,
      classes=["seg_val_img"],
      target_size=(HEIGHT, WIDTH),
      color_mode="rgb",
      class_mode=None,
      batch_size=BATCH_SIZE,
      shuffle=True,
      seed=seed
  )

  mask_generator = augmentations.flow_from_directory(
      TRAINING_MASKS_DIR,
      classes=["seg_val_masks"],
      target_size=(HEIGHT,WIDTH),
      color_mode="grayscale",
      class_mode=None,
      batch_size=BATCH_SIZE,
      shuffle=True,
      seed=seed
  )

  # The following part continually generates batch after batch of randomised training image/mask pairs
  while(1):
        imgs = image_generator.next()
        masks = mask_generator.next()
        masks = masks.reshape(BATCH_SIZE, HEIGHT*WIDTH, 1)

        yield imgs, masks


# Do training
print("Training the u-net")


model.fit_generator(  
        generator=training_generator(),
        steps_per_epoch=300, 
        validation_data=val_generator(),
        epochs=1, 
        validation_steps=1,
        verbose=1,
        shuffle=False,
        callbacks=[CSVLogger(OUTPUT_FILE,
                          append=False,
                          separator=";")]
)

model.save("./plant_seg.h5")


# Predict image using just trained model
model = load_model("./plant_seg.h5")
test_image = cv2.imread("test.jpg")
test_image = cv2.resize(test_image, (256, 256))
output = model.predict(test_image[None,...].astype(np.float32))[0]  
output = output.reshape((HEIGHT,WIDTH,1))
#output = skimage.exposure.rescale_intensity(output)

cv2_imshow(output*255)
print(output)
