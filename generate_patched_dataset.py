'''
This script loops over the annotated images, preprocesses them, and generates
a ready-to-use dataset of patches with the selected patch size.

A folder named 'dataset' is created inside the CucurPollen folder to store
the generated dataset. It contains the 'train', 'val', and 'test' directories
with the images ('images') and masks ('masks') patches for each split.
'''

#-----------------------------------------------------------------------------#

from utils import splits, patchify

import os
import pandas as pd
import cv2
import numpy as np

#-----------------------------------------------------------------------------#

# Patch size
patch_size = 512

# Folders
dataset_folder = 'CucurPollen/dataset'
images_folder = 'CucurPollen/images'
masks_folder = 'CucurPollen/masks'
files_folder = 'files'
if not os.path.exists(dataset_folder):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_folder, split, 'images'))
        print(f'Folder "dataset/{split}/images" created')
        os.makedirs(os.path.join(dataset_folder, split, 'masks'))
        print(f'Folder "dataset/{split}/masks" created')

# Read master_image table
master_image = pd.read_csv('CucurPollen/metadata/master_image.csv', sep = ';')
master_image = master_image[master_image['annotated']]

# Shuffle rows for a random split
master_image = master_image.sample(
    frac = 1,
    random_state = 42
).reset_index(drop = True)

#-----------------------------------------------------------------------------#

# Initialize patch counters
counters = {
    'train': 1,
    'val': 1
}

# Initialize class distribution for train split
class_distribution_train = {
    'bg': [],
    'ng': [],
    'gg': [],
    'pt': []
}

# Loop over annotated images and generate patches
for image_name, image_path in zip(master_image['name'], master_image['path']):

    print(f'Processing: {image_name}')

    # Read image and mask
    image = cv2.imread(image_path, flags = cv2.IMREAD_UNCHANGED)
    mask = np.load(f'{masks_folder}/{image_name}.npy')

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert Leica images to ensure pollen objects appear darker than bg
    if '_LEI_' in image_name:
        image = cv2.bitwise_not(image)

    # Min-max normalize
    image = cv2.normalize(
        image,
        None,
        alpha = 0,
        beta = 255, 
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_8U
    )

    #-------------------------------------------------------------------------#

    # Get image type (species + macroscopy)
    image_type = image_name.split('_')[1:3]
    image_type = f'{image_type[0]}_{image_type[1]}'

    # Select split and update counter
    if splits[image_type]['n_train'] < splits[image_type]['total_train']:
        split = 'train'
    elif splits[image_type]['n_val'] < splits[image_type]['total_val']:
        split = 'val'
    else:
        split = 'test'

        # Save full images
        np.save(
            os.path.join(
                dataset_folder,
                split,
                'images',
                f'{image_name}.npy'
            ),
            image
        )
        np.save(
            os.path.join(
                dataset_folder,
                split,
                'masks',
                f'{image_name}.npy'
            ),
            mask
        )

        # Ignore the patching process
        continue

    # Increase number of processed images
    splits[image_type][f'n_{split}'] += 1

    # Update class distribution if split is train
    if split == 'train':
        for i, cls in enumerate(['bg', 'ng', 'gg', 'pt']):
            class_distribution_train[cls].append((mask == i).sum())

    #-------------------------------------------------------------------------#

    # Patch image
    patch_image_list, position_image_list, shape_resized = patchify(
        array = image,
        patch_size = patch_size,
        interpolation = 'linear'
    )

    # Patch mask
    patch_mask_list, position_mask_list, shape_resized = patchify(
        array = mask,
        patch_size = patch_size,
        interpolation = 'linear'
    )

    # Loop over patches
    for patch_image, patch_mask in zip(patch_image_list, patch_mask_list):

        # Save patches
        np.save(
            os.path.join(
                dataset_folder,
                split,
                'images',
                f'patch_{counters[split]}.npy'
            ),
            patch_image
        )
        np.save(
            os.path.join(
                dataset_folder,
                split,
                'masks',
                f'patch_{counters[split]}.npy'
            ),
            patch_mask
        )

        # Increase patch counter
        counters[split] += 1

# Save class distribution
class_distribution_train = pd.DataFrame(class_distribution_train)
class_distribution_train.to_excel(
    f'{dataset_folder}/class_distribution_train.xlsx',
    index = False
)