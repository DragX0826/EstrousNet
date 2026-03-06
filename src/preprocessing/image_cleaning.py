from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image_bgr: np.ndarray, fast_mode: bool = True) -> np.ndarray:
    """Apply contrast normalization and denoising for microscopy images."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    if fast_mode:
        denoised = cv2.medianBlur(enhanced_bgr, 3)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 3, 3, 7, 21)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    return normalized
