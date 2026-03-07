# EstrousNet Baseline Results (v4 Final, skip_unet)

Source CSV: `results/bootstrap_v4_final/per_image_stage_summary.csv`

## Key Findings

- Total images: **18**
- Reliable detection (`n_cells > 0`): **15/18**
- Insufficient detection (`n_cells = 0`): **3**
- 10x/40x pair consistency: **3/9**

## Stage Distribution

- Proestrus: 9
- Metestrus: 8
- Estrus: 1
- Diestrus: 0

## Flagged Cases

- Insufficient detection (`n_cells = 0`):
  - `4-3 10x.jpg`
  - `4-4 10x.jpg`
  - `4-9 10x.jpg`
- Over-detection candidate: `4-7 10x.jpg` (`n_cells = 199`)

## Pair Consistency (10x vs 40x)

| Pair | 10x Stage | 40x Stage | Match |
|---|---|---|---|
| 4-1 | Proestrus | Proestrus | Yes |
| 4-10 | Proestrus | Metestrus | No |
| 4-2 | Proestrus | Metestrus | No |
| 4-3 | Proestrus | Metestrus | No |
| 4-4 | Proestrus | Proestrus | Yes |
| 4-5 | Estrus | Metestrus | No |
| 4-7 | Metestrus | Proestrus | No |
| 4-8 | Metestrus | Metestrus | Yes |
| 4-9 | Proestrus | Metestrus | No |

## Per-Image Result Table

| Image | n_cells | Cornified | Epithelial | Leukocyte | Predicted Stage |
|---|---:|---:|---:|---:|---|
| 4-1 10x.jpg | 11 | 3 | 6 | 2 | Proestrus |
| 4-1 40x.jpg | 19 | 15 | 4 | 0 | Proestrus |
| 4-10 10x.jpg | 22 | 10 | 11 | 1 | Proestrus |
| 4-10 40x.jpg | 24 | 14 | 6 | 4 | Metestrus |
| 4-2 10x.jpg | 19 | 12 | 6 | 1 | Proestrus |
| 4-2 40x.jpg | 13 | 6 | 2 | 5 | Metestrus |
| 4-3 10x.jpg | 0 | 0 | 0 | 0 | Proestrus |
| 4-3 40x.jpg | 8 | 3 | 2 | 3 | Metestrus |
| 4-4 10x.jpg | 0 | 0 | 0 | 0 | Proestrus |
| 4-4 40x.jpg | 10 | 5 | 4 | 1 | Proestrus |
| 4-5 10x.jpg | 2 | 2 | 0 | 0 | Estrus |
| 4-5 40x.jpg | 4 | 2 | 1 | 1 | Metestrus |
| 4-7 10x.jpg | 199 | 107 | 61 | 31 | Metestrus |
| 4-7 40x.jpg | 149 | 87 | 48 | 14 | Proestrus |
| 4-8 10x.jpg | 4 | 2 | 1 | 1 | Metestrus |
| 4-8 40x.jpg | 11 | 6 | 3 | 2 | Metestrus |
| 4-9 10x.jpg | 0 | 0 | 0 | 0 | Proestrus |
| 4-9 40x.jpg | 15 | 9 | 2 | 4 | Metestrus |

## Limitation Text (Report-ready)

> This pipeline serves as a proof-of-concept for zero-annotation estrous stage classification. Reliable detection was achieved in 15 of 18 images, with 3 low-contrast 10x images yielding insufficient cell candidates. Inter-magnification agreement was 3/9, indicating field-of-view sampling bias and residual segmentation instability.