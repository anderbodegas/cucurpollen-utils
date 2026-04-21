'''
This script is a minimal training example using the patched CucurPollen
dataset. A set of baseline models can be selected using the SMP package,
including U-Net, U-Net++, Deeplabv3, Deeplabv3+, Segformer, etc. A patience
criterion based on the validation loss is used to prevent overfitting and save
time.

A folder named 'checkpoints/{model_name}' is created in the root folder to
store the logs and model checkpoints.
'''

#-----------------------------------------------------------------------------#

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import utils
import random

#-----------------------------------------------------------------------------#

# Reproducibility
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
#-----------------------------------------------------------------------------#

# Options
model_name = 'unet'
n_epochs = 100
patience_epochs = 10
input_size = 512
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#-----------------------------------------------------------------------------#

# Ensure checkpoint folder exists
folder_checkpoint = f'checkpoints/{model_name}'
if not os.path.exists(folder_checkpoint):
    os.makedirs(folder_checkpoint)
    print('Folder created')

# Create datasets for each split
dataset_train = utils.CucurPollenDataset(
    folder = 'CucurPollen/dataset/train',
    transforms = utils.transforms,
    augment = True
)
dataset_val = utils.CucurPollenDataset(
    folder = 'CucurPollen/dataset/val',
    transforms = utils.transforms,
    augment = False
)

# Get generator for reproducibility
generator = torch.Generator()
generator.manual_seed(seed)

# Get number of workers
num_workers = min(4, os.cpu_count())

# Get worker initializon function
def seed_worker(worker_id: int):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Get train and val loaders
loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size = batch_size,
    shuffle = True,
    generator = generator,
    num_workers = num_workers,
    worker_init_fn = seed_worker
)
loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size = batch_size,
    shuffle = False,
    generator = generator,
    num_workers = num_workers,
    worker_init_fn = seed_worker
)

# Initialize model
model = utils.initialize_models(model_name)

# Initialize optimizer
optimizer = torch.optim.AdamW(
    params = [p for p in model.parameters() if p.requires_grad],
    lr = 1e-3,
    weight_decay = 1e-3
)

# Criterion and accuracy function
loss_function = smp.losses.DiceLoss(
    mode = 'multiclass',
    from_logits = True,
    smooth = 1e-6
)
loss_function = loss_function.to(device)

# Train model
utils.train(
    model = model,
    loader_train = loader_train,
    loader_val = loader_val,
    optimizer = optimizer,
    loss_function = loss_function,
    n_epochs = n_epochs,
    patience_epochs = patience_epochs,
    folder_checkpoint = folder_checkpoint,
    device = device
)