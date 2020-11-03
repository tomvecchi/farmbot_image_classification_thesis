"""
    Created by Tom Vecchi for 2020 UQ Farmbot Thesis Project

    The DetectedRegion class describes a single region of interest obtained
    using the segmenter. It contains methods for getting the size and health state
    of this particular region (the health status is obtained using the classifier.)
"""

import numpy as np
import cv2
import logging
from keras.models import Sequential, Model, load_model
from util import PlantStatus

classes = ["dead", "healthy", "non-plant"]

# Classifies it as dead if its prediction for the dead class is above a threshold,
# even if another class is higher- err on the side of deads
def argmax_favouring_deads(pred_vector):

    if pred_vector[0] > 0.01: # arbitrary num- gives good results
        return 0 # dead
    else:
        return pred_vector.index(max(pred_vector))


# Class for representing an individual region detected by the segmentation algorithm.
class DetectedRegion(object):

    def __init__(self, image_data, segmented_mask):
        self.raw_image = image_data
        self.segmented_image = segmented_mask
        
        self.plant_size = np.sum(self.segmented_image) # Count 1 pixels in binary image
        self.health_status = None

    # Runs classifier on image (using raw data)
    def classify(self, classifier, visual_inspection=False): 
        img = np.copy(self.raw_image)
        img = cv2.resize(img, (40,40))
        img = np.array(img)
        img = np.expand_dims(img, 0)

        pred = classifier.predict(img * 255)
        
        pred_list = max(pred).tolist()
        pred_class = argmax_favouring_deads(pred_list)

        if visual_inspection:
            logging.info("Predicted class = " + classes[pred_class] + str(pred_list))
            cv2.imshow("self.rawimage", self.raw_image)
            cv2.waitKey(0)

        self.health_status = PlantStatus(pred_class) # Use enum