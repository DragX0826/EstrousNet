from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.cell_features import extract_cell_features
from src.models.cell_classifier import classify_cells
from src.models.stage_classifier import infer_stage, load_stage_rules
from src.preprocessing.image_cleaning import preprocess_image
from src.segmentation.cell_segmentation import segment_cells
from src.utils.io import ensure_dir, list_images, read_image, write_image
from src.utils.visualization import create_overlay, save_stage_distribution_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EstrousNet minimal interpretable pipeline.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Input image folder.")
    parser.add_argument("--output_dir", type=str, default="results", help="Output folder.")
    parser.add_argument(
        "--cell_model",
        type=str,
        default=None,
        help="Optional trained RandomForest .joblib model path.",
    )
    parser.add_argument("--min_area", type=int, default=80, help="Minimum segmented cell area.")
    parser.add_argument("--max_area", type=int, default=20000, help="Maximum segmented cell area.")
    parser.add_argument(
        "--min_distance",
        type=int,
        default=14,
        help="Minimum local-max distance for watershed seeds.",
    )
    parser.add_argument(
        "--max_peaks",
        type=int,
        default=1200,
        help="Maximum watershed seed count per image.",
    )
    parser.add_argument(
        "--seed_threshold_rel",
        type=float,
        default=0.35,
        help="Relative distance-transform threshold for seed generation.",
    )
    parser.add_argument(
        "--stage_rules",
        type=str,
        default="config/stage_rules.yaml",
        help="YAML file for stage inference rules.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Process only the first N images (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    image_paths = list_images(input_dir)
    if not image_paths and input_dir == Path("data/raw"):
        image_paths = list_images("data")

    if not image_paths:
        raise FileNotFoundError(f"No microscopy images found under: {input_dir}")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    predictions_dir = ensure_dir(output_dir / "predictions")
    overlays_dir = ensure_dir(output_dir / "overlays")
    figures_dir = ensure_dir(output_dir / "figures")
    stage_rules = load_stage_rules(args.stage_rules)

    summary_rows: list[dict[str, float | int | str]] = []
    all_cells = []

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

        features_df = extract_cell_features(preprocessed, labels)
        classified_df = classify_cells(features_df, model_path=args.cell_model)

        counts = (
            classified_df["cell_type"].value_counts().to_dict()
            if not classified_df.empty
            else {}
        )
        stage_result = infer_stage(counts, rules=stage_rules)

        image_id = image_path.stem
        if not classified_df.empty:
            classified_df = classified_df.copy()
            classified_df["image_id"] = image_id
            all_cells.append(classified_df)
            classified_df.to_csv(predictions_dir / f"cells_{image_id}.csv", index=False)

        overlay = create_overlay(
            image_bgr=image,
            labels=labels,
            cells_df=classified_df,
            stage=stage_result["stage"],
            counts=counts,
        )
        write_image(overlays_dir / f"{image_id}_overlay.jpg", overlay)

        summary_rows.append(
            {
                "image_id": image_id,
                "stage": stage_result["stage"],
                "total_cells": int(sum(counts.values())),
                "cornified_count": int(counts.get("Cornified epithelial cell", 0)),
                "epithelial_count": int(counts.get("Nucleated epithelial cell", 0)),
                "leukocyte_count": int(counts.get("Leukocyte", 0)),
                "cornified_ratio": stage_result["cornified_ratio"],
                "epithelial_ratio": stage_result["epithelial_ratio"],
                "leukocyte_ratio": stage_result["leukocyte_ratio"],
                "source_path": str(image_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("image_id")
    summary_df.to_csv(predictions_dir / "image_summary.csv", index=False)

    if all_cells:
        all_cells_df = pd.concat(all_cells, ignore_index=True)
        all_cells_df.to_csv(predictions_dir / "all_cells.csv", index=False)

    save_stage_distribution_plot(summary_df, figures_dir / "stage_distribution.png")
    print(f"Processed {len(image_paths)} images.")
    print(f"Summary saved: {predictions_dir / 'image_summary.csv'}")


if __name__ == "__main__":
    main()
