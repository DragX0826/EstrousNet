from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def segment_cells(
    image_bgr: np.ndarray,
    min_area: int = 80,
    max_area: int = 20000,
    min_distance: int = 14,
    max_peaks: int = 1200,
    seed_threshold_rel: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment cells using threshold + watershed.

    Returns:
        labels: integer label map (0 is background)
        binary_mask: uint8 mask with foreground as 255
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = binary > 0
    if not np.any(mask):
        return np.zeros_like(gray, dtype=np.int32), binary

    distance = ndi.distance_transform_edt(mask)
    max_distance = float(np.max(distance))
    if max_distance <= 0:
        return np.zeros_like(gray, dtype=np.int32), binary

    seed_mask = distance >= (max_distance * seed_threshold_rel)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=(mask & seed_mask).astype(np.uint8),
        num_peaks=max_peaks,
        exclude_border=False,
    )

    markers = np.zeros_like(gray, dtype=np.int32)
    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx

    if np.max(markers) == 0:
        markers, _ = ndi.label(mask)
    else:
        markers, _ = ndi.label(markers > 0)

    raw_labels = watershed(-distance, markers, mask=mask)

    filtered_labels = np.zeros_like(raw_labels, dtype=np.int32)
    next_label = 1
    for region in regionprops(raw_labels):
        if min_area <= region.area <= max_area:
            filtered_labels[raw_labels == region.label] = next_label
            next_label += 1

    return filtered_labels, binary
