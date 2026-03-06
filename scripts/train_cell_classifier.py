from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cell_classifier import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest cell-type classifier.")
    parser.add_argument(
        "--labeled_features_csv",
        type=str,
        required=True,
        help="CSV containing features and `cell_type` labels.",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="results/models/cell_type_rf.joblib",
        help="Path for exported model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_random_forest(args.labeled_features_csv, args.output_model_path)
    print(f"Model saved to: {result['model_path']}")
    print("Classification report:")
    print(result["report"])


if __name__ == "__main__":
    main()
