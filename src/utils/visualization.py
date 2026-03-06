from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_MAP = {
    "Cornified epithelial cell": (40, 180, 255),   # orange
    "Nucleated epithelial cell": (60, 200, 60),    # green
    "Leukocyte": (255, 120, 120),                  # blue-red tint
}


def create_overlay(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    cells_df: pd.DataFrame,
    stage: str,
    counts: dict[str, int],
) -> np.ndarray:
    overlay = image_bgr.copy()

    for _, row in cells_df.iterrows():
        label_id = int(row["label_id"])
        cell_type = row["cell_type"]
        mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLOR_MAP.get(cell_type, (255, 255, 255))
        cv2.drawContours(overlay, contours, -1, color, 1)

    legend = (
        f"Stage: {stage} | "
        f"Cornified: {counts.get('Cornified epithelial cell', 0)} | "
        f"Epithelial: {counts.get('Nucleated epithelial cell', 0)} | "
        f"Leukocyte: {counts.get('Leukocyte', 0)}"
    )
    cv2.rectangle(overlay, (8, 8), (min(overlay.shape[1] - 8, 900), 40), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        legend,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def save_stage_distribution_plot(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_df.empty or "stage" not in summary_df.columns:
        return

    stage_counts = summary_df["stage"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    stage_counts.plot(kind="bar", color="#2d7fb8")
    plt.title("Predicted Stage Distribution")
    plt.xlabel("Stage")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

