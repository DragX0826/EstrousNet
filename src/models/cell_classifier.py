from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "area",
    "circularity",
    "nucleus_score",
    "mean_intensity",
    "mean_saturation",
]

CELL_TYPE_CANONICAL = {
    "cornified": "Cornified epithelial cell",
    "cornified epithelial cell": "Cornified epithelial cell",
    "cornified_epithelial_cell": "Cornified epithelial cell",
    "epithelial": "Nucleated epithelial cell",
    "nucleated epithelial cell": "Nucleated epithelial cell",
    "nucleated_epithelial_cell": "Nucleated epithelial cell",
    "leukocyte": "Leukocyte",
    "leukocytes": "Leukocyte",
}


def normalize_cell_type_label(label: str) -> str:
    key = str(label).strip().lower()
    return CELL_TYPE_CANONICAL.get(key, str(label))


def _rule_based_cell_type(row: pd.Series) -> str:
    # Leukocytes are usually small, round, and nucleus-dominant.
    if row["area"] < 260 and row["circularity"] > 0.55 and row["nucleus_score"] > 0.22:
        return "Leukocyte"

    # Cornified cells are often larger and weakly nucleated.
    if row["area"] > 800 and row["nucleus_score"] < 0.15:
        return "Cornified epithelial cell"

    return "Nucleated epithelial cell"


def classify_cells(features_df: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """Classify each cell as one of three cytology types."""
    if features_df.empty:
        return features_df.assign(cell_type=pd.Series(dtype=str))

    result = features_df.copy()
    if model_path and Path(model_path).exists():
        model = joblib.load(model_path)
        result["cell_type"] = model.predict(result[FEATURE_COLUMNS])
    else:
        result["cell_type"] = result.apply(_rule_based_cell_type, axis=1)
    result["cell_type"] = result["cell_type"].map(normalize_cell_type_label)
    return result


def train_random_forest(
    labeled_features_csv: str,
    output_model_path: str = "results/models/cell_type_rf.joblib",
    random_state: int = 42,
) -> dict[str, str]:
    """
    Train a RandomForest cell-type classifier.

    Expects a CSV with feature columns + `cell_type`.
    """
    data = pd.read_csv(labeled_features_csv)
    if "cell_type" not in data.columns:
        raise ValueError("Input CSV must contain a `cell_type` column.")

    x = data[FEATURE_COLUMNS]
    y = data["cell_type"].map(normalize_cell_type_label)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=False)

    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    return {"model_path": str(output_path), "report": report}
