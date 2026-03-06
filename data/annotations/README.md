# Cell Annotation Schema

Primary annotation file: `cells.csv`

## Required columns

- `image_id`: image stem or unique image identifier.
- `x`: cell center x-coordinate in pixels.
- `y`: cell center y-coordinate in pixels.
- `cell_type`: one of `cornified`, `epithelial`, `leukocyte`.
- `area`: estimated cell area in pixels.
- `circularity`: shape score in `[0, 1]`.
- `has_nucleus`: `1` if visible nucleus, else `0`.

## Example

```csv
image_id,x,y,cell_type,area,circularity,has_nucleus
4-1 10x,132,88,cornified,340,0.71,0
4-1 10x,221,102,leukocyte,45,0.95,1
4-1 10x,310,120,epithelial,180,0.82,1
```

## Notes

- Keep label names lowercase for consistency.
- Use the same coordinate convention across all images.
- Start with 100-200 cells per class for the first supervised model.

## Semi-Auto Files

- `candidate_cells.csv`: auto-generated candidates from segmentation with patch paths.
- `reviewed_cells.csv`: reviewed labels (`cornified`, `epithelial`, `leukocyte`).
- `labeled_cell_features.csv`: exported training table for `train_cell_classifier.py`.

`candidate_cells.csv` / `reviewed_cells.csv` include:

- `image_id`
- `cell_id`
- `x`
- `y`
- `area`
- `circularity`
- `mean_intensity`
- `mean_saturation`
- `nucleus_score`
- `patch_path`
- `cell_type`
- `review_status`
