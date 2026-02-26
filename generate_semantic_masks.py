'''
This script loops over the annotations and generates masks for every image.
Pixel labels are processed in this order: pollen_tube > germinated_pollen_grain
> non_germinated_pollen_grain, so minority labels overwrite majority classes in
case of overlap.
'''

#-----------------------------------------------------------------------------#

import os
import pandas as pd
import json
import numpy as np

from pycocotools import mask as mask_utils

#-----------------------------------------------------------------------------#

# Mapping from category name to number
category_name_2_number = {
    'non_germinated_grain': 1,
    'germinated_grain': 2,
    'pollen_tube': 3
}

# Folders
annotations_folder = 'CucurPollen/annotations'
masks_folder = 'CucurPollen/masks'
if not os.path.exists(masks_folder):
    os.makedirs(masks_folder)
    print('Folder "masks" created')

# Loop over annotations
for annotation_name in os.listdir(annotations_folder):

    print(f'Processing: {annotation_name}')

    # Open the .json
    with open(os.path.join(annotations_folder, annotation_name), 'r') as f:
        data = json.load(f)

        # Mapping from category id to category name
        category_id_2_name = dict()
        for category in data['categories']:
            category_id_2_name[category['id']] = category['name']

        # Dictionary containing the information for every mask
        mask_dict = dict()
        for image in data['images']:
            mask_dict[image['id']] = dict()
            mask_dict[image['id']]['name'] = image['file_name'][:-4]
            mask_dict[image['id']]['shape'] = (image['height'], image['width'])
            mask_dict[image['id']]['mask'] = np.zeros(
                shape = mask_dict[image['id']]['shape'],
                dtype = 'uint8'
            )

        # Sort the annotations by category (PT > GG > NG)
        annotations_sorted = sorted(
            data['annotations'],
            key = lambda x: category_name_2_number[
                category_id_2_name[x['category_id']]
            ],
            reverse = True
        )

        # Loop over the sorted annotations
        for annotation in annotations_sorted:

            # Get annotation's image info
            image_id = annotation['image_id']
            label = category_name_2_number[
                category_id_2_name[annotation['category_id']]
            ]
            h, w = mask_dict[image_id]['shape']

            # Get the run-length encoding and fill the mask with final labels
            polygon = annotation['segmentation']
            rle = mask_utils.frPyObjects(polygon, h, w)
            rle = mask_utils.merge(rle)
            decoded = mask_utils.decode(rle)
            mask_dict[image_id]['mask'][decoded == 1] = label

        # Loop over generated masks and save them as .npy
        for mask in mask_dict.values():

            # Get name
            name = mask['name']

            # Save mask
            np.save(os.path.join(masks_folder, f'{name}.npy'), mask['mask'])