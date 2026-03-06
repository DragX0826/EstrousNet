from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "results" / "examples"


def _ensure_dirs() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def make_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    steps = [
        "Microscopy\nImage",
        "Cell\nSegmentation",
        "Feature\nExtraction",
        "Cell Type\nClassification",
        "Stage\nInference",
    ]
    xs = np.linspace(0.1, 0.9, len(steps))
    y = 0.5

    for i, (x, text) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2fb", edgecolor="#2d7fb8", linewidth=1.5),
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.06, y),
                xytext=(x + 0.06, y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2d7fb8"),
                xycoords=ax.transAxes,
            )

    ax.set_title("EstrousNet Pipeline", fontsize=16, pad=20)
    out = EXAMPLES_DIR / "pipeline_diagram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_segmentation_example() -> Path:
    raw_path = ROOT / "data" / "raw" / "4-10 40x.jpg"
    overlay_path = ROOT / "results" / "debug" / "segmentation_overlay" / "4-10 40x_seg_overlay.jpg"

    raw = _read_rgb(raw_path)
    overlay = _read_rgb(overlay_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(raw)
    axes[0].set_title("Raw Microscopy")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")

    fig.suptitle("Example of Detected Cells in Vaginal Smear Microscopy", fontsize=14)
    fig.tight_layout()
    out = EXAMPLES_DIR / "segmentation_example.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_feature_distribution() -> Path:
    candidates_csv = ROOT / "data" / "annotations" / "candidate_cells.csv"
    df = pd.read_csv(candidates_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df["area"], bins=30, color="#2d7fb8", alpha=0.9)
    axes[0].set_title("Cell Area Distribution")
    axes[0].set_xlabel("Area (pixels)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["circularity"], bins=30, color="#f28e2b", alpha=0.9)
    axes[1].set_title("Cell Circularity Distribution")
    axes[1].set_xlabel("Circularity")
    axes[1].set_ylabel("Count")

    fig.suptitle("Cell Feature Distributions (Candidate Cells)", fontsize=14)
    fig.tight_layout()
    out = EXAMPLES_DIR / "feature_distribution.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_stage_inference_example() -> Path:
    summary_csv = ROOT / "results" / "predictions" / "image_summary.csv"
    df = pd.read_csv(summary_csv)

    # Pick a representative image with more detected cells.
    row = df.sort_values("total_cells", ascending=False).iloc[0]
    image_id = row["image_id"]
    stage = row["stage"]
    ratios = [row["cornified_ratio"], row["epithelial_ratio"], row["leukocyte_ratio"]]
    labels = ["Cornified", "Epithelial", "Leukocyte"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, ratios, color=["#4e79a7", "#59a14f", "#e15759"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title(f"Stage Inference Example: {image_id} -> {stage}")
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out = EXAMPLES_DIR / "stage_inference_example.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    _ensure_dirs()
    outputs = [
        make_pipeline_diagram(),
        make_segmentation_example(),
        make_feature_distribution(),
        make_stage_inference_example(),
    ]
    for out in outputs:
        print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()

