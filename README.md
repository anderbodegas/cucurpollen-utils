# cucurpollen-utils

This repository provides reference scripts to facilitate the use of the CucurPollen microscopy dataset.

![Annotation Example](figure.png)

## Overview

The repository includes tools for:

- Converting COCO-format instance annotations into semantic segmentation masks
- Generating a patched dataset for training
- Training and evaluating segmentation models
- Visualizing annotations and model outputs

These scripts are intended as minimal yet practical examples to support reproducibility and simplify further work.

---

## Scripts

### `generate_semantic_masks.py`

Generates semantic segmentation masks from COCO-format annotation files located in: `CucurPollen/annotations`.

For each annotated image, a single-channel mask (`.npy`) is created and saved in: `CucurPollen/masks`.

Pixel values correspond to the following classes:

| Label value | Class name           |
|------------|----------------------|
| 0          | background           |
| 1          | non_germinated_grain |
| 2          | germinated_grain     |
| 3          | pollen_tube          |

In cases of overlapping annotations, pixel labels are assigned using the following priority: non_germinated_grain > germinated_grain > pollen_tube, ensuring that minority classes overwrite lower-priority labels during mask construction.


This ensures that higher-priority classes overwrite lower-priority labels during mask construction.

---

### `generate_patched_dataset.py`

Loops over the annotated images, preprocesses them, and generates a ready-to-use dataset of patches with the selected patch size.

---

### `train_models.py`

Minimal training example using the patched CucurPollen dataset. A set of baseline models can be selected using the SMP package, including U-Net, U-Net++, Deeplabv3, Deeplabv3+, Segformer, etc. A patience criterion based on the validation loss is used to prevent overfitting and save time.

---

### `test_models.py`

Tests the models trained with `train_models.py` on the test dataset. It also generates figures to illustrate model predictions.

---

### `visualize_semantic_masks.py`

Loops over the annotated images and creates an overlayed image with its mask.

---

### `utils.py`

Contains some functions and classes needed in the other scripts.

---

## Dataset

The complete CucurPollen dataset is publicly available in the Zenodo repository under DOI:  
**10.5281/zenodo.18736035**
---

## Citation
