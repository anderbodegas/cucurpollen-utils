'''
This script contains some functions and classes needed in the other scripts.
'''

#-----------------------------------------------------------------------------#

from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import tv_tensors
from torchvision.transforms import v2

import os
import torch
import torchmetrics
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import cv2

#-----------------------------------------------------------------------------#

transforms = {
    'base': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True)
    ]),
    'aug_geom': v2.Compose([
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomVerticalFlip(p = 0.5)
    ]),
    'aug_bright': v2.ColorJitter(brightness = 0.05, contrast = 0.05)
}

#-----------------------------------------------------------------------------#

splits = {
    'MEL_AXC': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 32,
        'total_val': 5,
        'total_test': 5
    },
    'MEL_LEI': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 1,
        'total_val': 1,
        'total_test': 1
    },
    'WAT_AXC': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 32,
        'total_val': 5,
        'total_test': 5
    },
    'WAT_LEI': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 1,
        'total_val': 1,
        'total_test': 1
    },
    'CUC_LEI': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 12,
        'total_val': 2,
        'total_test': 2
    },
    'PUM_LEI': {
        'n_train': 0,
        'n_val': 0,
        'n_test': 0,
        'total_train': 12,
        'total_val': 2,
        'total_test': 2
    },
}

#-----------------------------------------------------------------------------#

class CucurPollenDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            folder: str,
            transforms: dict,
            augment: bool
        ):
        super().__init__()
        self.folder_images = os.path.join(folder, 'images')
        self.folder_masks = os.path.join(folder, 'masks')
        self.list_images = sorted(os.listdir(self.folder_images))
        self.list_masks = sorted(os.listdir(self.folder_masks))
        self.transforms = transforms
        self.augment = augment

    def __getitem__(self, i: int):

        # Read image and mask
        image = np.load(os.path.join(self.folder_images, self.list_images[i]))
        mask = np.load(os.path.join(self.folder_masks, self.list_masks[i]))

        # Base transformations
        image = self.transforms[f'base'](image)
        mask = tv_tensors.Mask(torch.from_numpy(mask).long())
            
        # Apply augmentations
        if self.augment:
            image, mask = self.transforms['aug_geom'](image, mask)
            # image = self.transforms['aug_bright'](image)

        return image, mask
    
    def __len__(self):
        return len(self.list_images)

#-----------------------------------------------------------------------------#

class MultiDiceCE(torch.nn.Module):
    def __init__(
            self,
            w_class_dice: torch.Tensor,
            w_class_ce: torch.Tensor,
            w_dice: float = 0.5,
            w_ce: float = 0.5,
            background_idx: int = 0
        ):
        super().__init__()
        self.w_class_dice = w_class_dice
        self.w_class_ce = w_class_ce
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.loss_ce = torch.nn.CrossEntropyLoss(
            weight = self.w_class_ce,
            reduction = 'none'
        )
        self.background_idx = background_idx
        self.eps = 1e-6

    def forward(self, y_pred: torch.tensor, y_targ: torch.tensor):

        # Compute CE
        dims = (1, 2)
        ce = self.loss_ce(y_pred, y_targ).mean(dim = dims).mean()

        # Compute Dice
        preds = torch.softmax(y_pred, dim = 1)
        targs = y_targ.long()
        dice = 0.0
        w = 0.0
        for i, w_i in enumerate(self.w_class_dice):
            if i == self.background_idx:
                continue
            targs_i = (targs == i).float()
            preds_i = preds[:, i]
            if targs_i.sum().item() == 0:
                continue
            inter = (preds_i * targs_i).sum(dim = dims)
            union = preds_i.sum(dim = dims) + targs_i.sum(dim = dims)
            dice_i = (2 * inter + self.eps) / (union + self.eps)
            dice += w_i * dice_i.mean()
            w += w_i
        dice = dice / w if (w > 0) else 1

        # Return weighted loss
        return self.w_dice * (1 - dice) + self.w_ce * ce

#-----------------------------------------------------------------------------#

def compute_metrics(computed_matrix: torch.Tensor, per_class: bool = False):

    eps = 1e-8

    # Basic metrics
    TP = torch.diag(computed_matrix)
    FP = computed_matrix.sum(dim = 0) - TP
    FN = computed_matrix.sum(dim = 1) - TP

    # Per-class metrics
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    if per_class:
        return iou, dice, precision, recall

    # Remove background (class 0)
    fg = slice(1, None)

    iou = iou[fg].mean().item()
    dice = dice[fg].mean().item()
    precision = precision[fg].mean().item()
    recall = recall[fg].mean().item()

    # Global accuracy
    accuracy = (TP.sum() / computed_matrix.sum()).item()

    return iou, dice, precision, recall, accuracy

#-----------------------------------------------------------------------------#

def train(
    model: torch.nn.Module,
    loader_train: torch.utils.data.DataLoader,
    loader_val: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    n_epochs: int,
    patience_epochs: int,
    folder_checkpoint: str,
    device: torch.device | str,
):
    
    # Get logs
    logs = {
        'epoch': list(),
        'minutes': list(),
        'patience_counter': list(),
        'learning_rate': list(),
        'loss_train': list(),
        'loss_val': list(),
        'iou_train': list(),
        'iou_val': list(),
        'dice_train': list(),
        'dice_val': list(),
        'precision_train': list(),
        'precision_val': list(),
        'recall_train': list(),
        'recall_val': list(),
        'accuracy_train': list(),
        'accuracy_val': list()
    }

    # Ensure all lists are empty
    for key in logs.keys():
        logs[key] = []
    print('Logs reset')

    # Initialize best loss
    best_loss = np.inf

    # Create confusion matrix
    confusion_matrix = MulticlassConfusionMatrix(num_classes = 4).to(device)

    # Move model to device
    model = model.to(device)

    # Start epoch and patience counter
    epoch = 1
    patience_counter = 1

    # Epoch loop
    while (epoch <= n_epochs) and (patience_counter <= patience_epochs):

        # Update logs
        logs['epoch'].append(epoch)
        logs['patience_counter'].append(patience_counter)
        logs['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Get intial time
        time_init = time.time()

        # Train loop
        loss, computed_matrix = loop_train(
            model = model,
            loader_train = loader_train,
            optimizer = optimizer,
            confusion_matrix = confusion_matrix,
            loss_function = loss_function,
            device = device
        )

        # Compute metrics and update logs
        with torch.no_grad():
            iou, dice, precision, recall, accuracy = compute_metrics(
                computed_matrix = computed_matrix
            )
            logs['iou_train'].append(iou)
            logs['dice_train'].append(dice)
            logs['precision_train'].append(precision)
            logs['recall_train'].append(recall)
            logs['accuracy_train'].append(accuracy)
            logs['loss_train'].append(loss)

        # Validation loop
        loss, computed_matrix = loop_val(
            model = model,
            loader_val = loader_val,
            confusion_matrix = confusion_matrix,
            loss_function = loss_function,
            device = device
        )

        # Compute metrics and update logs
        with torch.no_grad():
            iou, dice, precision, recall, accuracy = compute_metrics(
                computed_matrix = computed_matrix
            )
            logs['iou_val'].append(iou)
            logs['dice_val'].append(dice)
            logs['precision_val'].append(precision)
            logs['recall_val'].append(recall)
            logs['accuracy_val'].append(accuracy)
            logs['loss_val'].append(loss)
        
        # Get final time and calculate total time
        time_final = time.time()
        time_total = (time_final - time_init) / 60.0

        # Update logs
        logs['minutes'].append(time_total)

        # Print progress
        print((
            f'EPOCH [{logs['epoch'][-1]} / {n_epochs}] | '
            f'MIN.: {logs['minutes'][-1]: .2f} | '
            f'PATIENCE: {logs['patience_counter'][-1]} | '
            f'LR: {logs['learning_rate'][-1]: .2e} | '
            f'LOSS T.: {logs['loss_train'][-1]: .2f} | '
            f'LOSS V.: {logs['loss_val'][-1]: .2f} | '
        ))

        # Check if loss has improved
        if logs['loss_val'][-1] < best_loss:
            best_loss = logs['loss_val'][-1]
            patience_counter = 1
            torch.save(model.state_dict(), f'{folder_checkpoint}/best_loss.pth')
            plot_curves(logs, folder_checkpoint, 'best_loss')
        else:
            patience_counter += 1

        # Save last model weights
        torch.save(model.state_dict(), f'{folder_checkpoint}/last.pth')
        plot_curves(logs, folder_checkpoint, 'last')

        # Save logs
        if epoch % 10 == 0:
            logs_df = pd.DataFrame(logs)
            logs_df.to_excel(f'{folder_checkpoint}/logs.xlsx', index = False)

        # Update epoch
        epoch += 1

    # Save logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_excel(f'{folder_checkpoint}/logs.xlsx', index = False)

#-----------------------------------------------------------------------------#

def loop_train(
    model: torch.nn.Module,
    loader_train: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    confusion_matrix: torchmetrics.Metric,
    loss_function,
    device: torch.device | str
):
    
    # Reset confusion matrix
    confusion_matrix.reset()
    
    # Set model in training mode
    model.train()

    # Initialize epoch loss
    epoch_loss = 0.0

    # Train loop
    for idx, (image, mask) in enumerate(loader_train):

        # Reset optimizer
        optimizer.zero_grad()

        # Move to device
        image, mask = image.to(device), mask.to(device)

        # Forward pass
        mask_pred = model(image)

        # Update confusion matrix
        with torch.no_grad():
            confusion_matrix.update(torch.argmax(mask_pred, dim = 1), mask)

        # Compute loss function
        loss = loss_function(mask_pred, mask)

        # Update epoch loss
        epoch_loss += loss.item()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    return epoch_loss / len(loader_train), confusion_matrix.compute()

#-----------------------------------------------------------------------------#

def loop_val(
    model: torch.nn.Module,
    loader_val: torch.utils.data.DataLoader,
    confusion_matrix: torchmetrics.Metric,
    loss_function,
    device: torch.device | str
):
    
    # Reset confusion matrix
    confusion_matrix.reset()
    
    # Set model in evaluation mode
    model.eval()

    # Initialize epoch loss
    epoch_loss = 0.0

    # Disable gradient calculations
    with torch.no_grad():

        # Val loop
        for image, mask in loader_val:

            # Move to device
            image, mask = image.to(device), mask.to(device)

            # Forward pass
            mask_pred = model(image)

            # Update confusion matrix
            confusion_matrix.update(torch.argmax(mask_pred, dim = 1), mask)

            # Compute loss function
            loss = loss_function(mask_pred, mask)

            # Update epoch loss
            epoch_loss += loss.item()

    return epoch_loss / len(loader_val), confusion_matrix.compute()

#-----------------------------------------------------------------------------#

def plot_curves(logs: dict, folder_checkpoint: str, model_name: str):

    # Get epoch
    epoch = logs['epoch'][-1]

    # Plot progress
    plt.style.use('ggplot')
    plt.plot(
        range(1, epoch + 1),
        logs['loss_train'],
        label = 'Train Loss',
        color = 'red',
        linestyle = '-'
    )
    plt.plot(
        range(1, epoch + 1),
        logs['loss_val'],
        label = 'Val Loss',
        color = 'red',
        linestyle = '--'
    )
    plt.plot(
        range(1, epoch + 1),
        logs['precision_train'],
        label = 'Train Pre',
        color = 'blue',
        linestyle = '-'
    )
    plt.plot(
        range(1, epoch + 1),
        logs['precision_val'],
        label = 'Val Pre',
        color = 'blue',
        linestyle = '--'
    )
    plt.plot(
        range(1, epoch + 1),
        logs['recall_train'],
        label = 'Train Rec',
        color = 'green',
        linestyle = '-'
    )
    plt.plot(
        range(1, epoch + 1),
        logs['recall_val'],
        label = 'Val Rec',
        color = 'green',
        linestyle = '--'
    )
    plt.title('Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Precision / Recall')
    plt.ylim(0, 1)
    plt.legend()

    # Save figure
    plt.savefig(f'{folder_checkpoint}/{model_name}.png')
    plt.close()
    plt.style.use('default')

#-----------------------------------------------------------------------------#

def initialize_models(model_name: str):
    if model_name == 'unet':
        return smp.Unet(
            encoder_name = 'resnet34',
            encoder_depth = 5,
            encoder_weights = None,
            in_channels = 1,
            classes = 4
        )
    elif model_name == 'unet++':
        return smp.UnetPlusPlus(
            encoder_name = 'resnet34',
            encoder_depth = 5,
            encoder_weights = None,
            in_channels = 1,
            classes = 4
        )
    elif model_name == 'deeplabv3':
        return smp.DeepLabV3(
            encoder_name = 'resnet34',
            encoder_depth = 5,
            encoder_weights = None,
            in_channels = 1,
            classes = 4
        )
    elif model_name == 'deeplabv3+':
        return smp.DeepLabV3Plus(
            encoder_name = 'resnet34',
            encoder_depth = 5,
            encoder_weights = None,
            in_channels = 1,
            classes = 4
        )
    elif model_name == 'segformer':
        return smp.Segformer(
            encoder_name = 'mit_b2',
            encoder_depth = 5,
            encoder_weights = None,
            in_channels = 1,
            classes = 4
        )

#-----------------------------------------------------------------------------#

def patchify(array: np.array, patch_size: int, interpolation: str = 'linear'):

    # Get image dimensions
    H_0, W_0 = array.shape

    # Get number of patches
    n = max(int(np.round(H_0 / patch_size, 0)), 1)
    m = max(int(np.round(W_0 / patch_size, 0)), 1)

    # Get new shape
    H, W = n * patch_size, m * patch_size

    # Resize
    if interpolation == 'linear':
        resized = cv2.resize(array, (W, H), interpolation = cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        resized = cv2.resize(array, (W, H), interpolation = cv2.INTER_NEAREST)

    # Intialize patch and position lists
    patch_list = []
    position_list = []

    # Loop to patch
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):

            # Get image and mask patch
            patch = resized[i : i + patch_size, j : j + patch_size]

            # Update path and position lists
            patch_list.append(patch)
            position_list.append((i, j))

    return patch_list, position_list, resized.shape

#-----------------------------------------------------------------------------#

def unpatchify(
    patch_list: list,
    position_list: list,
    shape_resized: tuple,
    shape_original: tuple,
    interpolation: str = 'nearest'
):

    # Get patch size
    patch_size = patch_list[0].shape[0]

    # Initialize reconstructed array
    recons = np.zeros(shape_resized)

    # Merge patches
    for patch, (i, j) in zip(patch_list, position_list):

        # Add patch to its position in the reconstructed image
        recons[i : i + patch_size, j : j + patch_size] = patch

    # Resize to the original dimensions
    H, W = shape_original
    if interpolation == 'linear':
        resized = cv2.resize(recons, (W, H), interpolation = cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        resized = cv2.resize(recons, (W, H), interpolation = cv2.INTER_NEAREST)
    else:
        return recons

    return resized