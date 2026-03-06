from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cell_classifier import normalize_cell_type_label


VALID_INPUT_LABELS = {"cornified", "epithelial", "leukocyte"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed cell annotations into training feature CSV."
    )
    parser.add_argument(
        "--reviewed_csv",
        type=str,
        default="data/annotations/reviewed_cells.csv",
        help="Reviewed candidate CSV from review_cell_candidates.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/annotations/labeled_cell_features.csv",
        help="Output CSV for model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviewed_csv = Path(args.reviewed_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not reviewed_csv.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {reviewed_csv}")

    df = pd.read_csv(reviewed_csv)
    if "cell_type" not in df.columns:
        raise ValueError("Reviewed CSV must contain `cell_type`.")

    labels = df["cell_type"].fillna("").astype(str).str.strip().str.lower()
    filtered = df[labels.isin(VALID_INPUT_LABELS)].copy()
    if filtered.empty:
        raise ValueError("No labeled rows found with labels: cornified/epithelial/leukocyte")

    filtered["cell_type"] = filtered["cell_type"].map(normalize_cell_type_label)

    required = ["area", "circularity", "nucleus_score", "mean_intensity", "cell_type"]
    for column in required:
        if column not in filtered.columns:
            raise ValueError(f"Reviewed CSV missing required column: {column}")

    if "mean_saturation" not in filtered.columns:
        filtered["mean_saturation"] = 0.0

    export_columns = [
        "area",
        "circularity",
        "nucleus_score",
        "mean_intensity",
        "mean_saturation",
        "cell_type",
    ]
    exported = filtered[export_columns].copy()
    exported.to_csv(output_csv, index=False)

    print(f"Exported labeled features: {len(exported)} rows")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()
