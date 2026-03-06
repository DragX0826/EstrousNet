# EstrousNet Draft Report

## Title
Automatic Estrous Cycle Stage Detection from Vaginal Smear Microscopy Images

## Problem
Manual estrous staging is labor-intensive and subjective. This project builds an interpretable ML pipeline for stage prediction.

## Data
- 18 microscopy images (10x and 40x views)

## Method
1. Image preprocessing
2. Cell segmentation (watershed)
3. Cell feature extraction
4. Cell-type classification
5. Stage inference by ratio rules

## Preliminary Results
- Add segmentation overlays and stage summary table here.

## Future Work
- Manual cell annotations for supervised cell classifier
- U-Net segmentation
- CNN cell patch classifier + Grad-CAM

