"""Machine-readable row diagnostics for topology decoder investigation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .path_result import CurvePathResult


def diagnostic_rows(result: CurvePathResult) -> list[dict]:
    rows = []
    observation = result.observation_score_by_row
    transition = result.transition_score_by_row
    for row in range(result.x_by_row.size):
        rows.append({
            "row": row,
            "visible_x": float(result.x_by_row[row]),
            "unwrapped_x": float(result.unwrapped_x_by_row[row]),
            "wrap_index": int(result.wrap_index_by_row[row]),
            "slope": float(result.slope_by_row[row]),
            "observation_score": float(observation[row]) if observation is not None else None,
            "transition_score": float(transition[row]) if transition is not None else None,
            "confidence": float(result.confidence_by_row[row]),
        })
    return rows


def write_diagnostic_csv(path: Path | str, result: CurvePathResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = diagnostic_rows(result)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["row"])
        writer.writeheader()
        writer.writerows(rows)
    return path

