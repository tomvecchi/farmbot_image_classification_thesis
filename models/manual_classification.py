""" 
    Created by Tom Vecchi for FarmBot Thesis 2020
    
    This file is used to manually classify the individual regions of interest in an image, then save the cropped images
    for use in training the classifier. This greatly speeds up the process of obtaining sufficient training data.
"""

import os
import cv2
import keras
import shutil
import skimage
import warnings
import numpy as np  
from skimage.measure import regionprops, label
from keras.models import Sequential, Model, load_model

#Set root folder & destination of each class's training/validation images
main_folder = r""

class_1 = r""
class_2 = r""
class_3 = r""

val_1 = r""
val_2 = r""
val_3 = r""

class_1_symbol = '1'
class_2_symbol = '2'
class_3_symbol = '3'

# Valid image file extensions
extensions = [".png", ".jpg"]

#Get list of all img files in active directory (plus subdirectories)
subdirs = [d for d in os.listdir(main_folder)]
files = []
for i in range(len(subdirs)):
    files += [subdirs[i] + os.path.sep + file 
                for file in os.listdir(main_folder + os.path.sep + subdirs[i])
                if any(file.endswith(ext) for ext in extensions)] # Bruh
files = np.random.choice(files, 100)
print(files)
count = 0

print("Found ", len(files), "files")

# Get segmentator
model = load_model("plant_seg.h5")

# Loop through all files, move them depending on user's input
for file in files:

    full_address = main_folder + os.path.sep + file # Load image from starting folder
    print(file)
    test_image = cv2.imread(full_address)
    test_image = cv2.resize(test_image, (256, 256))
    test_image = np.array(test_image, dtype=np.float32)/255.0
    
    display_image = np.copy(test_image) #
    output = model.predict(test_image[None,...].astype(np.float32))[0]  # Generate prediction
    output = output.reshape((256,256,1))*255
    
    plant = output >= 75 #Segment, find regions of interest
    output3 = np.concatenate([plant, plant, plant], axis=2)
    #test_image[output3 == 0] /= 5 # Remove background. Note, this may make the two NNs too interdependent 
    output = output/255 # don't know why it needs this to display properly

    #use skimage regionprops
    properties = regionprops(label(plant))
    region_count = 0
    for prop in properties:
        if prop.area > 200: # Ignore little glitches/false returns- too hard to extract any info from
            region_count += 1
            display_image = cv2.rectangle(display_image, (prop.bbox[1], prop.bbox[0]), (prop.bbox[4], prop.bbox[3]), (255,0,0), 1)
            cropped = test_image[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], 0:3]

            cv2.imshow("crop", cropped) #Just the roi
            cv2.imshow(file, display_image) #Unedited image
            cv2.imshow("seg res", output) #Raw seg result

            key = cv2.waitKey(300000) 
            subfolder = file.split("\\")[0]
            new_name = (subfolder + "_" + file.split("\\")[1].split(".")[0] + "_" + str(region_count) + ".png")

            print(new_name)
            if (key == ord(class_1_symbol)):
                print("Saved ", new_name, " to class healthy")
                if np.random.randint(5) == 0: 
                    cv2.imwrite(val_1 + os.path.sep + new_name, cropped*255)
                else: 
                    cv2.imwrite(class_1 + os.path.sep + new_name, cropped*255)
                count += 1

            elif (key == ord(class_2_symbol)):
                print("Saved ", new_name, " to class dead")
                if np.random.randint(5) == 0: 
                    cv2.imwrite(val_2 + os.path.sep + new_name, cropped*255)
                else: 
                    cv2.imwrite(class_2 + os.path.sep + new_name, cropped*255)
                count += 1

            elif (key == ord(class_3_symbol)):
                print("Saved ", new_name, " to class non-plant")
                if np.random.randint(5) == 0: 
                    cv2.imwrite(val_3 + os.path.sep + new_name, cropped*255)
                else: 
                    cv2.imwrite(class_3 + os.path.sep + new_name, cropped*255)
                count += 1

            elif (key == ord('q')): # Stop looping
                exit()

            else:
                print("user did not select a valid class, moving on")
                print("Sorted ", count, " files")

            cv2.destroyAllWindows()