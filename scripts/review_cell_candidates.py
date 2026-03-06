from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_image


KEY_TO_LABEL = {
    ord("1"): "cornified",
    ord("2"): "epithelial",
    ord("3"): "leukocyte",
    ord("0"): "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive cell annotation review tool for candidate cell patches."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/annotations/candidate_cells.csv",
        help="Input candidate CSV from generate_cell_candidates.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/annotations/reviewed_cells.csv",
        help="Reviewed CSV with assigned cell_type labels.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Optional start index in pending rows.",
    )
    parser.add_argument(
        "--text_mode",
        action="store_true",
        help="Use terminal input mode instead of OpenCV window.",
    )
    return parser.parse_args()


def draw_text(image, row: pd.Series, pending_idx: int, pending_total: int):
    canvas = image.copy()
    lines = [
        f"[{pending_idx + 1}/{pending_total}] cell_id={row['cell_id']}",
        f"image_id={row['image_id']}",
        "Keys: 1=cornified  2=epithelial  3=leukocyte  0=clear",
        "s=skip  q=save and quit",
    ]
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return canvas


def load_review_table(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        return pd.read_csv(output_csv)
    return pd.read_csv(input_csv)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists() and not output_csv.exists():
        raise FileNotFoundError(
            f"Missing CSV. Expected {input_csv} or existing review file {output_csv}."
        )

    df = load_review_table(input_csv, output_csv)
    if "cell_type" not in df.columns:
        df["cell_type"] = ""
    if "review_status" not in df.columns:
        df["review_status"] = "pending"

    pending_mask = df["cell_type"].fillna("").str.strip() == ""
    pending_indices = df.index[pending_mask].tolist()
    if not pending_indices:
        print("No pending cells to label.")
        df.to_csv(output_csv, index=False)
        return

    start = max(0, min(args.start_index, len(pending_indices) - 1))
    i = start
    window_name = "EstrousNet Cell Reviewer"
    if not args.text_mode:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 700)

    while i < len(pending_indices):
        row_idx = pending_indices[i]
        row = df.loc[row_idx]
        patch_path = Path(str(row["patch_path"]))
        if not patch_path.exists():
            df.loc[row_idx, "review_status"] = "missing_patch"
            i += 1
            continue

        if args.text_mode:
            print(f"[{i + 1}/{len(pending_indices)}] {row['cell_id']} -> {patch_path}")
            print("Input label: 1=cornified 2=epithelial 3=leukocyte 0=clear s=skip q=quit")
            key_input = input("> ").strip().lower()
            if key_input == "q":
                key = ord("q")
            elif key_input == "s":
                key = ord("s")
            elif key_input in {"0", "1", "2", "3"}:
                key = ord(key_input)
            else:
                continue
        else:
            patch = read_image(patch_path)
            canvas = draw_text(patch, row, i, len(pending_indices))
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(0) & 0xFF

        if key in KEY_TO_LABEL:
            label = KEY_TO_LABEL[key]
            df.loc[row_idx, "cell_type"] = label
            df.loc[row_idx, "review_status"] = "labeled" if label else "cleared"
            i += 1
        elif key in (ord("s"), ord("S")):
            df.loc[row_idx, "review_status"] = "skipped"
            i += 1
        elif key in (ord("q"), ord("Q"), 27):
            break

        if i % 20 == 0:
            df.to_csv(output_csv, index=False)

    if not args.text_mode:
        cv2.destroyAllWindows()
    df.to_csv(output_csv, index=False)
    labeled = int((df["cell_type"].fillna("").str.strip() != "").sum())
    print(f"Saved review file: {output_csv}")
    print(f"Labeled cells: {labeled}/{len(df)}")


if __name__ == "__main__":
    main()
