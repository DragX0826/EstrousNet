from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_RULES: dict[str, Any] = {
    "default_stage": "Proestrus",
    "priority": ["Estrus", "Diestrus", "Metestrus"],
    "stages": {
        "Estrus": {"cornified_ratio_min": 0.70, "leukocyte_ratio_max": 0.20},
        "Diestrus": {"leukocyte_ratio_min": 0.60, "epithelial_ratio_max": 0.30},
        "Metestrus": {"leukocyte_ratio_min": 0.30},
        "Proestrus": {},
    },
}


def load_stage_rules(config_path: str | None = None) -> dict[str, Any]:
    if not config_path:
        return DEFAULT_RULES

    path = Path(config_path)
    if not path.exists():
        return DEFAULT_RULES

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    rules = {
        "default_stage": loaded.get("default_stage", DEFAULT_RULES["default_stage"]),
        "priority": loaded.get("priority", DEFAULT_RULES["priority"]),
        "stages": loaded.get("stages", DEFAULT_RULES["stages"]),
    }
    return rules


def _rule_passed(
    stage_rules: dict[str, float],
    cornified_ratio: float,
    epithelial_ratio: float,
    leukocyte_ratio: float,
) -> bool:
    if "cornified_ratio_min" in stage_rules and cornified_ratio < stage_rules["cornified_ratio_min"]:
        return False
    if "cornified_ratio_max" in stage_rules and cornified_ratio > stage_rules["cornified_ratio_max"]:
        return False

    if "epithelial_ratio_min" in stage_rules and epithelial_ratio < stage_rules["epithelial_ratio_min"]:
        return False
    if "epithelial_ratio_max" in stage_rules and epithelial_ratio > stage_rules["epithelial_ratio_max"]:
        return False

    if "leukocyte_ratio_min" in stage_rules and leukocyte_ratio < stage_rules["leukocyte_ratio_min"]:
        return False
    if "leukocyte_ratio_max" in stage_rules and leukocyte_ratio > stage_rules["leukocyte_ratio_max"]:
        return False
    return True


def infer_stage(cell_counts: dict[str, int], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer estrous stage from cell-type counts."""
    cornified = cell_counts.get("Cornified epithelial cell", 0)
    epithelial = cell_counts.get("Nucleated epithelial cell", 0)
    leukocyte = cell_counts.get("Leukocyte", 0)

    total = max(cornified + epithelial + leukocyte, 1)
    cornified_ratio = cornified / total
    epithelial_ratio = epithelial / total
    leukocyte_ratio = leukocyte / total

    config = rules or DEFAULT_RULES
    stage = str(config.get("default_stage", "Proestrus"))
    stages = config.get("stages", {})
    priority = config.get("priority", [])

    for candidate in priority:
        candidate_rules = stages.get(candidate, {})
        if _rule_passed(candidate_rules, cornified_ratio, epithelial_ratio, leukocyte_ratio):
            stage = str(candidate)
            break

    return {
        "stage": stage,
        "cornified_ratio": cornified_ratio,
        "epithelial_ratio": epithelial_ratio,
        "leukocyte_ratio": leukocyte_ratio,
    }
