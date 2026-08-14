"""Persistence helpers for pilot results.

Results are stored as JSON (one file per (seed, threshold) condition) plus a
flat CSV summary that aggregates every condition into one table for plotting.
The condition variable is the fixed firing threshold; spike activity is a
MEASURED OUTCOME, so everything downstream is keyed by OBSERVED activity.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ConditionRecord:
    """One completed (seed, threshold) pilot condition."""

    seed: int
    threshold: float
    observed_activity_per_task: list[float]
    mean_observed_activity: float
    accuracy_matrix: list[list[float]]
    final_avg_accuracy: float
    per_task_forgetting: list[float]
    mean_forgetting: float
    overlap_cka: float
    overlap_cosine: float
    dead_network: bool = False
    dataset: str = "mnist"
    train_losses: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dirs(results_dir: str) -> dict[str, str]:
    """Create the results/pilot/{raw,metrics,plots,checkpoints,logs} tree."""
    subdirs = {}
    for name in ("raw", "metrics", "plots", "checkpoints", "logs"):
        path = os.path.join(results_dir, name)
        os.makedirs(path, exist_ok=True)
        subdirs[name] = path
    return subdirs


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf to null so the JSON is portable."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def save_condition(raw_dir: str, record: ConditionRecord) -> str:
    """Write one condition's JSON. Returns the file path."""
    fname = f"seed{record.seed}_{record.dataset}_theta{record.threshold:.3g}.json"
    path = os.path.join(raw_dir, fname)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(record.to_dict()), fh, indent=2)
    return path


def load_conditions(raw_dir: str) -> list[ConditionRecord]:
    """Load every condition JSON in raw_dir into ConditionRecords."""
    records: list[ConditionRecord] = []
    if not os.path.isdir(raw_dir):
        return records
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as fh:
            data = json.load(fh)
        records.append(ConditionRecord(**data))
    return records


_SUMMARY_COLUMNS = (
    "seed",
    "dataset",
    "threshold",
    "mean_observed_activity",
    "final_avg_accuracy",
    "mean_forgetting",
    "overlap_cka",
    "overlap_cosine",
    "dead_network",
)


def save_summary_csv(metrics_dir: str, records: list[ConditionRecord]) -> str:
    """Write a flat one-row-per-condition CSV for plotting/inspection."""
    path = os.path.join(metrics_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_SUMMARY_COLUMNS)
        for r in records:
            writer.writerow([getattr(r, col) for col in _SUMMARY_COLUMNS])
    return path
