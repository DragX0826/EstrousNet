# EstrousNet Baseline Results

Date: 2026-03-06

## Summary

What was achieved:

- Built and ran an end-to-end microscopy image analysis pipeline.
- Implemented and executed a semi-automatic annotation workflow.
- Extracted cell-level morphology/intensity features.
- Produced rule-based image-level stage inference results.

Current limitations:

- Human-labeled cells are only 3.
- Classifier performance is not yet a valid estimate.
- Stage outputs collapsed to a single class (`Proestrus`).
- More reviewed annotations are required before reliable evaluation.

## Dataset

- images: 18 (`data/raw`)
- generated candidate cells: 205 (`data/annotations/candidate_cells.csv`)
- reviewed/labeled cells for training: 3 (`data/annotations/labeled_cell_features.csv`)
- cornified: 1
- epithelial: 1
- leukocyte: 1

## Segmentation

- parameters:
  - `min_distance: 10`
  - `max_peaks: 500`
  - `seed_threshold_rel: 0.2`
- debug overlays:
  - full set: `results/debug/segmentation_overlay/`
  - selected samples:
    - `report/figures/segmentation_samples/4-1 10x_seg_overlay.jpg`
    - `report/figures/segmentation_samples/4-10 40x_seg_overlay.jpg`
    - `report/figures/segmentation_samples/4-5 40x_seg_overlay.jpg`
    - `report/figures/segmentation_samples/4-7 40x_seg_overlay.jpg`
    - `report/figures/segmentation_samples/4-8 10x_seg_overlay.jpg`
- notes:
  - strong under-detection in some images (`4-2 10x: 1 cell`, `4-7 10x: 2 cells`)
  - segmentation sensitivity appears highly image-dependent (10x vs 40x, stain/background variation)

## Cell Classifier

- model target: RandomForest (`scripts/train_cell_classifier.py`)
- training status: blocked for proper train/test split due insufficient labels per class
  - error: each class has only 1 sample, stratified split cannot run
- exploratory metric (LOOCV on 3 labeled cells, not statistically meaningful):
  - accuracy: `0.0`
  - confusion matrix: `results/confusion_matrix/rf_loocv_confusion_matrix.csv`
  - figure: `results/confusion_matrix/rf_loocv_confusion_matrix.png`
- main confusion:
  - cornified -> epithelial
  - leukocyte -> epithelial

## Stage Inference

- input summary: `results/predictions/image_summary.csv`
- stage counts:
  - estrus: 0
  - metestrus: 0
  - diestrus: 0
  - proestrus: 18
- ratio pattern:
  - all images predicted `cornified_ratio=0`, `epithelial_ratio=1`, `leukocyte_ratio=0`
- interpretation:
  - current pipeline is collapsing to a single cell type/stage; this is likely driven by insufficient labeled data and unstable segmentation/cell-type boundaries

Per-image predictions:

| image_id   | stage     |   cornified_ratio |   epithelial_ratio |   leukocyte_ratio |
|:-----------|:----------|------------------:|-------------------:|------------------:|
| 4-1 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-1 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-10 10x   | Proestrus |                 0 |                  1 |                 0 |
| 4-10 40x   | Proestrus |                 0 |                  1 |                 0 |
| 4-2 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-2 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-3 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-3 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-4 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-4 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-5 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-5 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-7 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-7 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-8 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-8 40x    | Proestrus |                 0 |                  1 |                 0 |
| 4-9 10x    | Proestrus |                 0 |                  1 |                 0 |
| 4-9 40x    | Proestrus |                 0 |                  1 |                 0 |

## Failure Cases

- `4-2 10x`: severe under-detection (1 candidate cell)
- `4-7 10x`: severe under-detection (2 candidate cells)
- global failure mode: all 18 images inferred as Proestrus

## Why This Baseline Failed

1. Insufficient labeled data:
   - only 3 reviewed cells (1 per class), so classifier training/validation is not statistically valid.
2. Segmentation instability:
   - detection count varies heavily by image, with strong under-detection in specific fields.
3. Leukocyte signal loss:
   - missed small cells directly harms metestrus/diestrus discrimination.
4. Cascading error in interpretable pipeline:
   - weak cell labels -> weak cell typing -> collapsed stage ratios.

## Why We Stop Here (Current Scope Decision)

- This project goal for the current phase is a reproducible prototype, not a publishable final model.
- The next credible improvement requires substantial manual annotation effort.
- Continuing model complexity now (e.g., deeper models/detectors) would not produce trustworthy conclusions.

## If Continued Later

1. Complete reviewed labels for candidate cells (and expand beyond current set).
2. Re-train RandomForest with valid split and class-balanced evaluation.
3. Re-run full stage inference and compare against expert stage labels.
