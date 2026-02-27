# cucurpollen-utils

Utility scripts for generating and visualizing semantic segmentation masks from the CucurPollen dataset.

![Annotation Example](figure.png)

## Overview

This repository provides reference scripts to facilitate the use of the CucurPollen microscopy dataset. The tools convert COCO-format instance annotations into semantic segmentation masks and generate visualization overlays for inspection.

These scripts are intended as minimal examples to support reproducibility and simplify integration into computer vision workflows.

---

## Scripts

### generate_semantic_masks.py

Generates semantic segmentation masks from COCO-format annotation files located in: CucurPollen/annotations. For each annotated image, a single-channel mask (`.npy`) is created and saved in: CucurPollen/masks.

Pixel values correspond to the following classes:

| Label value | Class name                  |
|------------|-----------------------------|
| 0          | Background                  |
| 1          | non_germinated_grain        |
| 2          | germinated_grain            |
| 3          | pollen_tube                 |

In cases of overlapping annotations, pixel labels are assigned using the following priority: non_germinated_grain > germinated_grain > pollen_tube, ensuring that minority classes overwrite lower-priority labels during mask construction.

### visualize_semantic_masks.py

Creates overlay visualizations for annotated images.

For each image flagged as `annotated = True` in: CucurPollen/metadata/master_image.csv, the script:

1. Loads the original microscopy image.
2. Loads the corresponding semantic mask (`.npy`).
3. Generates a color-coded overlay:
   - Red: non_germinated_grain
   - Green: germinated_grain
   - Blue: pollen_tube
4. Saves the overlay image to: CucurPollen/overlayed/.

This allows rapid qualitative inspection of annotation consistency.

---

## Dataset

The complete CucurPollen dataset is publicly available in the Zenodo repository under DOI: 10.5281/zenodo.18736035

---

## Citation

If you use the CucurPollen dataset or these utilities in your research, please cite:
