"""
    Created by Tom Vecchi for 2020 UQ Farmbot Thesis Project

    Utility functions used by the other classes.
"""

from enum import Enum
import datetime

# Class for representing possible plant states
class PlantStatus(Enum):
    DEAD = 0
    HEALTHY = 1
    NOT_PLANT = 2

# Thresholds a greyscale image to binary (NOTE: changes original)
def apply_image_threshold(image, threshold=75):
    image[image >= threshold] = 255
    image[image < threshold] = 0
    return image

# Gets time in convenient format
def timestamp():
    raw_time = datetime.datetime.now()
    return raw_time.strftime("%Y/%m/%d_%H:%M:%S")