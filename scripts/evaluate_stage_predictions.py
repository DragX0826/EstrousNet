"""
Evaluate stage predictions against manually labeled ground truth.

Usage:
  python scripts/evaluate_stage_predictions.py \
    --pred_csv results/bootstrap_final/per_image_stage_summary.csv \
    --gt_csv data/annotations/image_stage_ground_truth.csv \
    --out_dir results/evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VALID_STAGES = {"proestrus", "estrus", "metestrus", "diestrus"}


def normalize_stage(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image", "predicted_stage"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    out = df[["image", "predicted_stage"]].copy()
    out["predicted_stage"] = out["predicted_stage"].map(normalize_stage)
    return out


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image", "ground_truth_stage"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")
    out = df[["image", "ground_truth_stage"]].copy()
    out["ground_truth_stage"] = out["ground_truth_stage"].map(normalize_stage)
    out = out[out["ground_truth_stage"] != ""]
    return out


def main(pred_csv: Path, gt_csv: Path, out_dir: Path) -> None:
    pred = load_predictions(pred_csv)
    gt = load_ground_truth(gt_csv)

    merged = pred.merge(gt, on="image", how="inner")
    if merged.empty:
        raise SystemExit(
            "No overlapping labeled rows. Fill ground_truth_stage and ensure image names match."
        )

    valid_gt = merged["ground_truth_stage"].isin(VALID_STAGES)
    valid_pred = merged["predicted_stage"].isin(VALID_STAGES)
    keep = valid_gt & valid_pred
    filtered = merged[keep].copy()

    if filtered.empty:
        raise SystemExit(
            "No rows with valid stages. Allowed: proestrus, estrus, metestrus, diestrus."
        )

    filtered["correct"] = filtered["predicted_stage"] == filtered["ground_truth_stage"]
    accuracy = float(filtered["correct"].mean())

    labels = ["proestrus", "estrus", "metestrus", "diestrus"]
    cm = pd.crosstab(
        filtered["ground_truth_stage"],
        filtered["predicted_stage"],
        rownames=["ground_truth"],
        colnames=["predicted"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "evaluation_merged.csv"
    cm_path = out_dir / "confusion_matrix.csv"
    metrics_path = out_dir / "metrics.json"

    filtered.to_csv(merged_path, index=False)
    cm.to_csv(cm_path)

    metrics = {
        "n_evaluated": int(len(filtered)),
        "n_labeled_input": int(len(gt)),
        "accuracy": round(accuracy, 4),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Evaluated rows: {metrics['n_evaluated']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved merged rows: {merged_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stage predictions against manual labels")
    parser.add_argument(
        "--pred_csv",
        default="results/bootstrap_final/per_image_stage_summary.csv",
        help="Prediction CSV with columns: image, predicted_stage",
    )
    parser.add_argument(
        "--gt_csv",
        default="data/annotations/image_stage_ground_truth.csv",
        help="Ground-truth CSV with columns: image, ground_truth_stage",
    )
    parser.add_argument(
        "--out_dir",
        default="results/evaluation",
        help="Output directory for metrics and confusion matrix",
    )
    args = parser.parse_args()
    main(Path(args.pred_csv), Path(args.gt_csv), Path(args.out_dir))
