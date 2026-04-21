'''
This script tests the models trained with train_models.py on the test dataset.
Ensure that the seed for the splitting is the same in both scripts before
running them.

A folder named 'test/{model_name}' is created in the root folder to store the
images overlayed with model predicionts. An excel named 'test_metrics.xlsx' is
generated inside the 'test' folder containing the test metrics.
'''

#-----------------------------------------------------------------------------#

from torchmetrics.classification import MulticlassConfusionMatrix
from utils import (
    transforms,
    initialize_models,
    patchify,
    unpatchify,
    compute_metrics
)

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
    
#-----------------------------------------------------------------------------#

# Get metrics header
test_metrics = {
    'model_name': list(),
    'iou_bg': list(),
    'iou_ng': list(),
    'iou_gg': list(),
    'iou_pt': list(),
    'miou': list()
}

# Options
model_name = 'unet'
input_size = 512
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#-----------------------------------------------------------------------------#

# Folders
dataset_folder = 'CucurPollen/dataset'
folder_plots = f'test/{model_name}'
if not os.path.exists(folder_plots):
    os.makedirs(folder_plots)
    print('Folder created')

# Get checkpoint folder
folder_checkpoint = f'checkpoints/{model_name}'

# Initialize model
model = initialize_models(model_name)

# Load weights
state_dict = torch.load(
    f'{folder_checkpoint}/best_loss.pth',
    weights_only = True
)
model.load_state_dict(state_dict)

# Move model to device
model = model.to(device)

# Model in eval mode
model.eval()

# Create confusion matrix
confusion_matrix = MulticlassConfusionMatrix(num_classes = 4)

for image_name in os.listdir(f'{dataset_folder}/test/images'):

    # Read image and mask
    image = np.load(f'{dataset_folder}/test/images/{image_name}')
    mask = np.load(f'{dataset_folder}/test/masks/{image_name}')

    # Patch image
    patch_list, position_list, shape_resized = patchify(
        array = image,
        patch_size = input_size,
        interpolation = 'linear'
    )

    # Initialize list for pred patches
    preds_list = []

    # Loop over patches and predict them
    for patch in patch_list:
            
        # Prepare for model inference
        patch = transforms['base'](patch)
        patch = patch.unsqueeze(0).to(device)

        # Inference
        preds = model(patch)

        # Prepare prediction
        preds = torch.argmax(preds, dim = 1).squeeze().cpu().numpy()

        # Add to list
        preds_list.append(preds)

    # Merge predicted patches
    preds = unpatchify(
        patch_list = preds_list,
        position_list = position_list,
        shape_resized = shape_resized,
        shape_original = image.shape,
        interpolation = None
    )

    # Normalize image
    image = image - image.min()
    image = image / image.max()

    # Plot image, GT, and preds
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image, cmap = 'gray')
    ax[0].set_title('Image')
    ax[0].set_axis_off()
    ax[1].imshow(mask, cmap = 'cividis')
    ax[1].set_title('GT')
    ax[1].set_axis_off()
    ax[2].imshow(preds, cmap = 'cividis')
    ax[2].set_title('Prediction')
    ax[2].set_axis_off()
    
    # Save figure
    plt.tight_layout()
    plt.axis('off')
    fig.savefig(f'{folder_plots}/{image_name[:-4]}.png')
    plt.close()

    # Update confusion matrix
    confusion_matrix.update(
        torch.from_numpy(preds),
        torch.from_numpy(mask)
    )

# Compute metrics and update logs
computed_matrix = confusion_matrix.compute()
with torch.no_grad():
    iou, dice, precision, recall = compute_metrics(
        computed_matrix = computed_matrix,
        per_class = True
    )

# Update test metrics
test_metrics['model_name'].append(model_name)
test_metrics['iou_bg'].append(iou[0].item())
test_metrics['iou_ng'].append(iou[1].item())
test_metrics['iou_gg'].append(iou[2].item())
test_metrics['iou_pt'].append(iou[3].item())
test_metrics['miou'].append(iou[1:].mean().item())

# Convert into dataframe and save
test_metrics = pd.DataFrame(test_metrics)
test_metrics.to_excel('test/test_metrics.xlsx', index = False)