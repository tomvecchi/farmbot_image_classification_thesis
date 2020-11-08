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

from enum import Enum
from client import FarmbotClient
from send_email import send_email
from skimage.measure import regionprops, label
from download_images import download_all_images
from keras.models import Sequential, Model, load_model

from plant import Plant
from detected_region import DetectedRegion
from util import PlantStatus, timestamp

     
# Converts lines in the plant file into variables
def parse_plant_file_line(line):
    elements = line.split(",")
    try: 
        plant_type = elements[0]
        
        x = elements[1]
        
        y = elements[2].rstrip()

    except: 
        return None, None, None

    if len(elements) != 3:
        return None, None, None
         # Invalid
    
    return plant_type, x, y

# Sets up MQTT client
def load_farmbot_client(device_id, token, mqtt_server):
    try:
        mqtt_client = FarmbotClient(device_id, token, mqtt_server)
    except Exception:
        logging.ERROR("Could not initialise FarmbotClient")
        logging.ERROR(Exception)
        return None

    return mqtt_client


# Loads in Keras models- segmenter and classifier- from disk.
def load_ml_models(segmenter_file, classifier_file):
    seg_address = "models" + os.path.sep + segmenter_file
    class_address = "models" + os.path.sep + classifier_file
    segmenter = None
    classifier = None

    # Load segmenter model
    try:
        segmenter =  load_model(seg_address)
    except exception:
        logging.error("Error- could not load segmentation neural net")
        logging.error(exception)

    # Load classifier model
    try:
        classifier =  load_model(class_address)
    except exception:
        logging.error("Error- could not load classification neural net")
        logging.error(exception)

    if segmenter is not None and classifier is not None:
        logging.info("Neural nets loaded successfully")
        return segmenter, classifier
    else:
        logging.error("Other error on loading neural nets")
        exit(1)


# Generates a final message to be sent to the user by email or whatever  
def generate_final_message(plants):

    dead_flag = False
    output = ""
    output += "Results of plant health analysis at time " + timestamp() + " :\r\n"
    
    for p in plants:
        output += "Plant of type " + p.type + " at location " + p.x_coordinate + "," + p.y_coordinate + " has total healthy area of " + str(p.total_healthy_area)
        output += "\r\n"

        if p.dead_areas:
            output += ". Note: Dead leaves were detected at this location."
            dead_flag = True
        output += "\r\n"

    if dead_flag:
        output += "Warning: Some plants may be dead.\r\n"
    
    output += "Thank you for using the Plant Healthy Monitor.\r\n"
    return output
    
    


# Loops through list of plants, imaging and collecting data on each
def daily_routine_demo(plant_file):
        
    # Ensures that when we do looping, only images remaining 
    # are new ones to be downloaded
    #download_all_images(450, "imgs") 
    #exit(0)
    farmbot_client = load_farmbot_client(creds.device_id, creds.token, creds.mqtt_server)
    segmenter, classifier = load_ml_models("segmenter_model.h5", "classifier_model.h5")

    f = open(plant_file, "r")
    plants = [] # List of all plants in the garden bed
    
    for line in f:
        plant_type, x, y = parse_plant_file_line(line)
        if plant_type is not None:
            plants.append(Plant(plant_type, x, y)) # Create new plant object

    if len(plants) == 0:
        logging.warning("No valid entries in plant file, ending")
        return

    # The main event. Add new functions as needed in the section below.
    for p in plants:
        print(p)
        #p.take_image(farmbot_client) # Send instruction to farmbot
        #time.sleep(60) # Give it some time to ensure image is uploaded, before retrieving it
        p.download_image()
       
        p.run_segmenter(segmenter, True)

        p.run_classifier(classifier, True)

        p.status_report()

    # Notify user of results
    farmbot_client.shutdown()

    output_msg = generate_final_message(plants)

    logging.debug(output_msg)
    #send_email(output_msg)


#Main control
#This part uses the schedule library to determine how often the main routine is called
#Could be once per day, could be more often
#This can be changed if an alternative method (eg cronjob) is chosen instead

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    daily_routine_demo("plant_locations.txt")
    
    exit(0) #Remove this line if deploying for actual ongoing operations
    schedule.every(24).hours.do(daily_routine("plant_locations.txt"))

    while True:
        schedule.run_pending()
        time.sleep(1)