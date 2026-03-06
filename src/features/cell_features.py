from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def _nucleus_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """Estimate nucleus presence by dark-core fraction inside one cell."""
    pixels = gray[mask]
    if pixels.size == 0:
        return 0.0
    threshold = np.percentile(pixels, 25)
    return float(np.mean(pixels <= threshold))


def extract_cell_features(image_bgr: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Extract morphology and intensity features for each segmented cell."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    rows: list[dict[str, float | int]] = []
    for region in regionprops(labels, intensity_image=gray):
        mask = labels == region.label
        area = float(region.area)
        perimeter = float(region.perimeter) if region.perimeter > 0 else 1.0
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        circularity = max(0.0, min(circularity, 1.0))

        min_row, min_col, max_row, max_col = region.bbox

        rows.append(
            {
                "label_id": int(region.label),
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "mean_intensity": float(np.mean(gray[mask])),
                "mean_saturation": float(np.mean(hsv[..., 1][mask])),
                "nucleus_score": _nucleus_score(gray, mask),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
                "bbox_min_row": int(min_row),
                "bbox_min_col": int(min_col),
                "bbox_max_row": int(max_row),
                "bbox_max_col": int(max_col),
            }
        )

    return pd.DataFrame(rows)

