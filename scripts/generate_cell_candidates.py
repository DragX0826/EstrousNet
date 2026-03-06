from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.cell_features import extract_cell_features
from src.preprocessing.image_cleaning import preprocess_image
from src.segmentation.cell_segmentation import segment_cells
from src.utils.io import ensure_dir, list_images, read_image, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate cells and patch images for semi-automatic annotation."
    )
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Image folder.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/annotations/candidate_cells.csv",
        help="Output CSV containing candidate cells.",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default="data/annotations/patches",
        help="Directory for cropped cell patches.",
    )
    parser.add_argument("--min_area", type=int, default=80)
    parser.add_argument("--max_area", type=int, default=20000)
    parser.add_argument("--min_distance", type=int, default=14)
    parser.add_argument("--max_peaks", type=int, default=1200)
    parser.add_argument("--seed_threshold_rel", type=float, default=0.35)
    parser.add_argument("--patch_margin", type=int, default=8)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Process only the first N images (0 means all).",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save segmentation debug overlay for each processed image.",
    )
    parser.add_argument(
        "--overlay_dir",
        type=str,
        default="results/debug/segmentation_overlay",
        help="Output directory for segmentation overlays when --save_overlay is enabled.",
    )
    return parser.parse_args()


def crop_patch(image, min_row: int, min_col: int, max_row: int, max_col: int, margin: int):
    h, w = image.shape[:2]
    r1 = max(0, int(min_row) - margin)
    c1 = max(0, int(min_col) - margin)
    r2 = min(h, int(max_row) + margin)
    c2 = min(w, int(max_col) + margin)
    return image[r1:r2, c1:c2], (r1, c1, r2, c2)


def build_segmentation_overlay(image_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

    cell_count = int(np.max(labels))
    cv2.rectangle(overlay, (8, 8), (330, 36), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        f"Detected cells: {cell_count}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    patch_dir = ensure_dir(args.patch_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = ensure_dir(args.overlay_dir) if args.save_overlay else None

    image_paths = list_images(input_dir)
    if not image_paths and input_dir == Path("data/raw"):
        image_paths = list_images("data")
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_dir}")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    rows: list[dict[str, str | int | float]] = []
    for image_path in image_paths:
        image = read_image(image_path)
        preprocessed = preprocess_image(image)
        labels, _ = segment_cells(
            preprocessed,
            min_area=args.min_area,
            max_area=args.max_area,
            min_distance=args.min_distance,
            max_peaks=args.max_peaks,
            seed_threshold_rel=args.seed_threshold_rel,
        )
        if overlay_dir is not None:
            image_id = image_path.stem
            overlay = build_segmentation_overlay(image, labels)
            write_image(overlay_dir / f"{image_id}_seg_overlay.jpg", overlay)

        features_df = extract_cell_features(preprocessed, labels)
        if features_df.empty:
            continue

        image_id = image_path.stem
        image_patch_dir = ensure_dir(patch_dir / image_id)

        for row in features_df.to_dict(orient="records"):
            label_id = int(row["label_id"])
            cell_id = f"{image_id}_{label_id:04d}"
            patch, (r1, c1, r2, c2) = crop_patch(
                image,
                min_row=int(row["bbox_min_row"]),
                min_col=int(row["bbox_min_col"]),
                max_row=int(row["bbox_max_row"]),
                max_col=int(row["bbox_max_col"]),
                margin=args.patch_margin,
            )
            if patch.size == 0:
                continue

            patch_path = image_patch_dir / f"{cell_id}.jpg"
            write_image(patch_path, patch)

            rows.append(
                {
                    "image_id": image_id,
                    "cell_id": cell_id,
                    "x": float(row["centroid_col"]),
                    "y": float(row["centroid_row"]),
                    "area": float(row["area"]),
                    "circularity": float(row["circularity"]),
                    "mean_intensity": float(row["mean_intensity"]),
                    "mean_saturation": float(row["mean_saturation"]),
                    "nucleus_score": float(row["nucleus_score"]),
                    "patch_path": str(patch_path).replace("\\", "/"),
                    "bbox_min_row": int(r1),
                    "bbox_min_col": int(c1),
                    "bbox_max_row": int(r2),
                    "bbox_max_col": int(c2),
                    "cell_type": "",
                    "review_status": "pending",
                }
            )

    columns = [
        "image_id",
        "cell_id",
        "x",
        "y",
        "area",
        "circularity",
        "mean_intensity",
        "mean_saturation",
        "nucleus_score",
        "patch_path",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row",
        "bbox_max_col",
        "cell_type",
        "review_status",
    ]
    candidates_df = pd.DataFrame(rows, columns=columns)
    if not candidates_df.empty:
        candidates_df.sort_values(["image_id", "cell_id"], inplace=True)
    candidates_df.to_csv(output_csv, index=False)

    print(f"Generated {len(candidates_df)} candidate cells.")
    print(f"CSV saved to: {output_csv}")
    print(f"Patches saved to: {patch_dir}")


if __name__ == "__main__":
    main()
