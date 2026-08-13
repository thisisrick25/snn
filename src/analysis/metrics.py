"""Continual-learning metrics derived from the accuracy matrix.

The accuracy matrix ``A`` is lower-triangular: ``A[i][j]`` is the accuracy on
task ``j`` after training task ``i`` (rows = training stage, cols = evaluated
task). Entries with ``j > i`` are NaN because that task had not been learned yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ContinualMetrics:
    """Summary metrics for one continual-learning run."""

    final_avg_accuracy: float
    per_task_forgetting: list[float]  # F_j per task, NaN for the last task
    mean_forgetting: float


def _n_tasks(accuracy_matrix: list[list[float]]) -> int:
    n = len(accuracy_matrix)
    if n == 0:
        raise ValueError("accuracy_matrix is empty")
    return n


def final_avg_accuracy(accuracy_matrix: list[list[float]]) -> float:
    """Mean accuracy over all tasks after the final task (last row)."""
    n = _n_tasks(accuracy_matrix)
    final_row = accuracy_matrix[n - 1]
    return sum(final_row) / n


def per_task_forgetting(accuracy_matrix: list[list[float]]) -> list[float]:
    """F_j = max_i A[i][j] - A[final][j], over stages i where task j was learned.

    The last task has no forgetting (it was learned last), so its entry is NaN.
    """
    n = _n_tasks(accuracy_matrix)
    final_row = accuracy_matrix[n - 1]
    forgetting: list[float] = []
    for j in range(n):
        if j == n - 1:
            forgetting.append(float("nan"))
            continue
        # Stages i >= j have a valid (non-NaN) measurement of task j.
        history = [accuracy_matrix[i][j] for i in range(j, n) if not math.isnan(accuracy_matrix[i][j])]
        best = max(history)
        forgetting.append(best - final_row[j])
    return forgetting


def mean_forgetting(accuracy_matrix: list[list[float]]) -> float:
    """Average forgetting over all tasks except the last."""
    f = per_task_forgetting(accuracy_matrix)
    valid = [v for v in f if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def compute_metrics(accuracy_matrix: list[list[float]]) -> ContinualMetrics:
    """Compute all continual-learning metrics from the accuracy matrix."""
    return ContinualMetrics(
        final_avg_accuracy=final_avg_accuracy(accuracy_matrix),
        per_task_forgetting=per_task_forgetting(accuracy_matrix),
        mean_forgetting=mean_forgetting(accuracy_matrix),
    )
