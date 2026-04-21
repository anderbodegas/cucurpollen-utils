'''
This script loops over the annotated images and creates an overlayed image with
its mask.

A folder named 'overlayed' is created inside the CucurPollen folder to store
the generated overlays
'''

#-----------------------------------------------------------------------------#

import os
import pandas as pd
import cv2
import numpy as np

#-----------------------------------------------------------------------------#

# Folders
masks_folder = 'CucurPollen/masks'
overlayeds_folder = 'CucurPollen/overlayed'
if not os.path.exists(overlayeds_folder):
    os.makedirs(overlayeds_folder)
    print('Folder "overlayeds" created')

# Read master_image table
master_image = pd.read_csv('CucurPollen/metadata/master_image.csv', sep = ';')
master_image = master_image[master_image['annotated']]

#-----------------------------------------------------------------------------#

# Loop over annotated images and generate overlayed images
for image_name, image_path in zip(master_image['name'], master_image['path']):

    print(f'Processing: {image_name}')

    # Read image and mask
    image = cv2.imread(image_path, flags = cv2.IMREAD_UNCHANGED)
    mask = np.load(f'{masks_folder}/{image_name}.npy')

    # Create label color array
    add = np.zeros_like(image)
    add[mask == 1] = [0, 0, 255]
    add[mask == 2] = [0, 255, 0]
    add[mask == 3] = [255, 0, 0]

    # Create overlayed image
    overlayed = cv2.addWeighted(image, 0.7, add, 0.3, gamma = 0)

    # Increase brightness
    overlayed = cv2.convertScaleAbs(overlayed, alpha = 1, beta = 50)

    # Save overlayed image
    cv2.imwrite(f'{overlayeds_folder}/{image_name}.png', overlayed)
