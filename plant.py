"""
    Created by Tom Vecchi for 2020 UQ Farmbot Thesis Project

    The Plant class represents a single plant in the FarmBot's bed. It contains methods
    for imaging that plant using the FarmBot's camera, as well as methods for running
    the segmenter and classifier on the resulting images. 
"""
import os
import cv2
import time
import creds
import client
import logging
import skimage
import datetime
import schedule
import numpy as np

# This stops tensorflow spam at the start
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from client import FarmbotClient
from skimage.measure import regionprops, label
from download_images import download_all_images
from keras.models import Sequential, Model, load_model
from detected_region import DetectedRegion
from util import PlantStatus, apply_image_threshold, timestamp


# Class for representing a plant in the FarmBot's bed.
class Plant(object):

    image_data = None
    segmented_regions = []
    total_healthy_area = 0
    dead_areas = False
    
    def __init__(self, plant_type, x, y):
        self.type = plant_type
        self.x_coordinate = x
        self.y_coordinate = y

    def __str__(self):
        return "Plant of type " + self.type + " at coords " + self.x_coordinate + "," + self.y_coordinate
    
    # Sends command to Farmbot client to image this plant.
    def take_image(self, farmbot_client):
        try:
            farmbot_client.move(int(self.x_coordinate), int(self.y_coordinate), 0)
            time.sleep(30) # Wait for it to move to destination
            farmbot_client.take_photo()
        except Exception:
            logging.ERROR("Failed to capture image from farmbot")
            logging.ERROR(Exception)

    # Attempts to download the most recent image from cloud storage.
    # This will be the desired one.
    def download_image(self):
        try:
            downloaded_image = download_all_images(1, "imgs")
        except Exception:
            logging.error("Error on image download: ", x, " ", y)
            logging.error(Exception)
        
        if len(downloaded_image) != 0:
            try:
                self.image_data = cv2.imread(downloaded_image[0])
            except Exception:
                logging.error("Error- could not read image ", downloaded_image[0])

    # Runs segmentation algorithm and obtains segmented regions
    def run_segmenter(self, segmenter, visual_inspection=False):
        test_image = cv2.resize(self.image_data, (256, 256))
        test_image = np.array(test_image, dtype=np.float32)/255.0
        
        display_image = np.copy(test_image) 

        output = segmenter.predict(test_image[None,...].astype(np.float32))[0]  # Generate prediction
        output = output.reshape((256,256,1))*255
        
        plant = apply_image_threshold(output)
        output = output/255 # Needs this to display properly

        properties = regionprops(label(plant))
        region_count = 0
        for prop in properties:
            if prop.area > 200: # Ignore little glitches/false returns- too hard to extract any info from
                region_count += 1
                display_image = cv2.rectangle(display_image, (prop.bbox[1], prop.bbox[0]), (prop.bbox[4], prop.bbox[3]), (255,0,0), 1)
                cropped = test_image[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], 0:3]
                cropped_mask = output[prop.bbox[0]:prop.bbox[3], prop.bbox[1]:prop.bbox[4], 0:3]

                if visual_inspection: # Show each cropped region
                    cv2.imshow("crop", cropped)
                    cv2.imshow("file", display_image)
                    cv2.imshow("seg res", cropped_mask)
                    cv2.waitKey(0)

                new_region = DetectedRegion(cropped, cropped_mask)
                self.segmented_regions.append(new_region)

    # Runs classifier on cropped sub-images detected by segmenter. This allows for a higher
    # level of precision.
    def run_classifier(self, classifier, visual_inspection=False):

        for region in self.segmented_regions:
            region.classify(classifier, visual_inspection)

    # Measure size of healthy plants in image (converting from pixels to 
    # whatever units are needed)
    def get_healthy_size(self, pixel_area_factor=1):
        num_healthy_pixels = 0

        for region in self.segmented_regions:
            if region.health_status == PlantStatus.HEALTHY:
                num_healthy_pixels += region.plant_size

        self.total_healthy_area = num_healthy_pixels * pixel_area_factor
        logging.debug("Total healthy area", self.total_healthy_area)

    # Checks if this plant's image contains any dead leaves
    def check_for_deads(self):
        for region in self.segmented_regions:
            if region.health_status == PlantStatus.DEAD:
                self.dead_areas = True
                logging.info("Dead leaf detected")

    # Human-readable report on the size and health of this plant.
    # The final, useful information extracted from the raw data.
    def status_report(self, log_file=None):
        self.get_healthy_size()
        self.check_for_deads()

        if log_file == None:
            log_file = self.type + self.x_coordinate + "_" + self.y_coordinate + ".log"
        
        log_file_address = "logs" + os.path.sep + log_file
        if os.path.exists(log_file_address): # Append to file, but only if it already exists
            mode = "a"
        else:
            mode = "w"

        with open(log_file_address, mode) as f:
            f.write("\n")
            f.write(""+timestamp()+","+str(self.total_healthy_area)+","+str(self.dead_areas))
