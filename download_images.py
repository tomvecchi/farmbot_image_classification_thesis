#!/usr/bin/env python3
"""
  Originally created by matpalm
  https://github.com/matpalm/farmbot_api

  Edited by Tom Vecchi for 2020 thesis project

  This file contains a function for downloading collected images from the FarmBot's
  server.
"""

from datetime import timezone
from dateutil.parser import parse
import creds
import json
import os
import requests
import sys
import time

def download_all_images(max=3, destination="."):
  print("Beginning image download")
  REQUEST_HEADERS = {'Authorization': 'Bearer ' + creds.token, 'content-type': "application/json"}

  count = 0
  downloaded = 0
  new_images = [] #All new images captured since it was last called (we will do processing on these)

  while True:

    response = requests.get('https://my.farmbot.io/api/images', headers=REQUEST_HEADERS)
    images = response.json()
    print("Found #images", len(images))

    if len(images) == count:
      print("All new images downloaded")
      return new_images

    for image_info in images:
      #print(count)

      if 'placehold.it' in image_info['attachment_url']:
        print("IGNORE! placeholder", image_info['id'])
        continue

      dts = parse(image_info['attachment_processed_at'])
      dts = dts.replace(tzinfo=timezone.utc).astimezone(tz=None)
      
      local_img_dir = destination + os.path.sep + "%s" % dts.strftime("%Y%m%d")
      if not os.path.exists(local_img_dir):
        os.makedirs(local_img_dir)

    
      local_img_name = "%s/%s.jpg" % (local_img_dir, dts.strftime("%H%M%S"))
    
      #Only do this part if the file is not already downloaded (my addition) 
      if not os.path.exists(local_img_name):
        print("Downloading >", local_img_name)
        downloaded = downloaded + 1

        # download image from google storage and save locally
        captured_img_name = image_info['meta']['name']
        if captured_img_name.startswith("/tmp/images"):
          req = requests.get(image_info['attachment_url'], allow_redirects=True)
          open(local_img_name, 'wb').write(req.content)
          #Record that we have a new image so that the other parts of the app can use it
          new_images.append(local_img_name)
          print("...done")

      count = count + 1

      if downloaded == max:
        print("Downloaded ", max, " images")
        return new_images
