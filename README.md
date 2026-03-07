# EstrousNet

Automatic estrous stage inference from rat vaginal smear microscopy images.

This repo is a **proof-of-concept biomedical pipeline**:

`microscopy image -> pseudo cell mask -> cell type counts -> ratio-based stage inference`

## Current Baseline (v4 final)

- Images: `18`
- Mode: `skip_unet` (Stage 1 pseudo-mask used directly)
- Reliable detections: `15/18`
- Insufficient detection: `3/18` (`4-3 10x`, `4-4 10x`, `4-9 10x`)
- Stage distribution: `Proestrus 9`, `Metestrus 8`, `Estrus 1`, `Diestrus 0`
- 10x/40x consistency: `3/9`

Detailed table and limitation text:
- [report/baseline_results_v4_final.md](report/baseline_results_v4_final.md)

## Run (Recommended)

```bash
python scripts/run_pseudolabel_bootstrap.py \
  --img_dir data/raw \
  --out_dir results/bootstrap_v4_final \
  --skip_unet \
  --area_min 500 \
  --area_max 15000 \
  --circularity_min 0.30
```

Outputs:

- `results/bootstrap_v4_final/overlays/`
- `results/bootstrap_v4_final/per_image_stage_summary.csv`
- `results/bootstrap_v4_final/cell_level_features.csv`

## Notes

- This is a pilot study baseline, not a final validated model.
- No strong image-level ground truth set is included yet.
- Main limitation is detection instability across magnification/contrast.
