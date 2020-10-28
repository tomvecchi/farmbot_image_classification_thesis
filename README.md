# FarmBot Image Classification Thesis
# By Tom Vecchi, 2020

Uses deep learning techniques to annotate and analyse images collected from the FarmBot, and logs this data for analysis. 

## Contents

The FarmBot user credentials and the email account which will be used for sending results must be entered in the creds.py file. Additionally the trained models, classifier_model.h5 and segmenter_model.h5, must be present in the ./models folder. \
The plant_locations.txt file contains a list of plants to image in the format type,x,y. \
The models should be saved in the models/ directory. The trained models are not included since Github doesn't allow files larger than 25 MB.

## Usage

Run daily_routine.py. It will image all the plants in plant_locations.txt and perform the analysis on them, then log this information in .log files in the logs folder. Use visualiser.py [filename] to plot the data in a given log file. 

### How it works

The code communicates with the FarmBot over MQTT and downloads the images using a HTTP REST API. The analysis uses a two-step approach of semantic segmentation using a U-Net algorithm, followed by a modified AlexNet for classifying healthy and dead plants. 

### Acknowledgements

This thesis would not have been possible without the help of my thesis supervisor Dr Matt D'Souza, and thesis tutor Scott Thomason.
