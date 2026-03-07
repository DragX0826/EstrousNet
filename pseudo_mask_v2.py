import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, segmentation


def generate_pseudo_mask(
    img_bgr: np.ndarray,
    area_min: int = 500,
    area_max: int = 40000,
    circularity_min: float = 0.30,
) -> np.ndarray:
    """
    Returns uint8 binary mask (0 or 255) of detected cells.

    Strategy:
      A) Large pale-pink cornified cells  -> HSV cytoplasm range + large blob filter
      B) Nucleated cells (epithelial/WBC) -> dark nucleus detection + dilation
      C) Watershed split on A where blobs are very large (likely clusters)
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    margin = 15

    # A. Cytoplasm mask (pale pink / light purple cornified cells)
    # Hue wraps: pinkish = high hue (150-179) OR low hue (0-15)
    cyto_hi = cv2.inRange(hsv, np.array([158, 6, 195]), np.array([179, 80, 255]))
    cyto_lo = cv2.inRange(hsv, np.array([0, 6, 195]), np.array([8, 80, 255]))
    cyto = cv2.bitwise_or(cyto_hi, cyto_lo)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cyto = cv2.morphologyEx(cyto, cv2.MORPH_CLOSE, kernel21, iterations=4)
    cyto = cv2.morphologyEx(cyto, cv2.MORPH_OPEN, kernel15, iterations=1)
    cyto = ndi.binary_fill_holes(cyto > 0).astype(np.uint8) * 255

    # B. Nucleus mask (dark purple/magenta nuclei)
    nuc_hsv = cv2.inRange(hsv, np.array([120, 40, 20]), np.array([175, 255, 180]))
    nuc_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 60, 80]))
    nucleus = cv2.bitwise_or(nuc_hsv, nuc_dark)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, kernel5, iterations=1)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # C. Watershed split for large cytoplasm blobs
    cyto_labels = measure.label(cyto > 0)
    split_mask = np.zeros((h, w), dtype=np.uint8)

    for region in measure.regionprops(cyto_labels):
        blob = (cyto_labels == region.label).astype(np.uint8)
        if region.area < area_min:
            continue

        # Small-enough blob -> keep as single cell
        if region.area < area_max * 0.6:
            split_mask[blob > 0] = 255
            continue

        # Large blob -> watershed split using nucleus seeds
        minr, minc, maxr, maxc = region.bbox
        pad = 5
        r1, r2 = max(0, minr - pad), min(h, maxr + pad)
        c1, c2 = max(0, minc - pad), min(w, maxc + pad)

        roi_blob = blob[r1:r2, c1:c2]
        roi_nuc = (nucleus[r1:r2, c1:c2] > 0) & (roi_blob > 0)

        dist = ndi.distance_transform_edt(roi_blob)
        seed_map = np.zeros_like(roi_blob, dtype=np.int32)

        nuc_labels, n_seeds = ndi.label(roi_nuc)
        if n_seeds == 0:
            from skimage.feature import peak_local_max

            peaks = peak_local_max(dist, min_distance=20, labels=roi_blob.astype(bool))
            for i, (pr, pc) in enumerate(peaks, start=1):
                seed_map[pr, pc] = i
            n_seeds = len(peaks)
        else:
            seed_map = nuc_labels

        if n_seeds == 0:
            split_mask[r1:r2, c1:c2][roi_blob > 0] = 255
            continue

        bg_seed = n_seeds + 1
        border = np.zeros_like(roi_blob, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        seed_map[border & (roi_blob == 0)] = bg_seed

        ws = segmentation.watershed(-dist, seed_map, mask=roi_blob.astype(bool))
        for label_id in range(1, n_seeds + 1):
            cell_region = (ws == label_id).astype(np.uint8)
            cell_area = int(cell_region.sum())
            if area_min <= cell_area <= area_max:
                split_mask[r1:r2, c1:c2][cell_region > 0] = 255

    # D. Small nucleated cells supplement
    nuc_dilated = cv2.dilate(nucleus, kernel5, iterations=2)
    nuc_cells = ndi.binary_fill_holes(nuc_dilated > 0).astype(np.uint8) * 255

    # E. Merge
    combined = cv2.bitwise_or(split_mask, nuc_cells)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # F. Final filter
    out = np.zeros((h, w), dtype=np.uint8)
    labels = measure.label(combined > 0)
    for region in measure.regionprops(labels):
        area = region.area
        if area < area_min or area > area_max:
            continue
        perim = region.perimeter if region.perimeter > 0 else 1
        circ = 4 * np.pi * area / (perim**2)
        if circ < circularity_min:
            continue
        minr, minc, maxr, maxc = region.bbox
        if minr < margin or minc < margin or maxr > h - margin or maxc > w - margin:
            continue
        out[labels == region.label] = 255
    return out


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/4-2 40x.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        sys.exit(1)

    mask = generate_pseudo_mask(img)
    regions = measure.regionprops(measure.label(mask > 0))
    areas = [r.area for r in regions]
    print(f"n_cells : {len(areas)}")
    if areas:
        print(f"area    : min={min(areas)}  median={int(np.median(areas))}  max={max(areas)}")

    overlay = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 80), 2)
    out_path = os.path.splitext(img_path)[0] + "_pseudo_test.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"overlay -> {out_path}")
