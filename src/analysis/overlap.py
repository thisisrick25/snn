"""Representational overlap measures between task representations.

Two measures are provided:

* Cosine similarity between task-mean representations (bounded in [-1, 1]).
* Linear Centered Kernel Alignment (CKA) between representation matrices
  (bounded in [0, 1]).

Both are computed pairwise across tasks; a single scalar summary per run is the
mean over all task pairs.
"""

from __future__ import annotations

import itertools

import torch


def cosine_overlap(rep_a: torch.Tensor, rep_b: torch.Tensor) -> float:
    """Cosine similarity between the mean representations of two tasks."""
    mean_a = rep_a.mean(dim=0)
    mean_b = rep_b.mean(dim=0)
    denom = mean_a.norm() * mean_b.norm()
    if denom == 0:
        return 0.0
    value = float(torch.dot(mean_a, mean_b) / denom)
    # Clamp to guard against floating-point drift outside the valid range.
    return max(-1.0, min(1.0, value))


def _center(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - matrix.mean(dim=0, keepdim=True)


def linear_cka(rep_a: torch.Tensor, rep_b: torch.Tensor) -> float:
    """Linear CKA between two representation matrices [n_samples, features].

    Requires the same number of samples (same fixed probe subset) in both.
    """
    if rep_a.shape[0] != rep_b.shape[0]:
        raise ValueError(
            f"CKA needs matching sample counts, got {rep_a.shape[0]} and {rep_b.shape[0]}"
        )
    x = _center(rep_a)
    y = _center(rep_b)
    # Hilbert-Schmidt Independence Criterion (linear kernel) via cross-covariance.
    hsic_xy = (x.t() @ y).pow(2).sum()
    hsic_xx = (x.t() @ x).pow(2).sum()
    hsic_yy = (y.t() @ y).pow(2).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    value = float(hsic_xy / denom)
    return max(0.0, min(1.0, value))


def mean_pairwise_overlap(reps: list[torch.Tensor], measure: str = "cka") -> float:
    """Mean overlap over all unordered task pairs.

    ``measure`` is ``"cka"`` (linear CKA) or ``"cosine"``.
    """
    if len(reps) < 2:
        raise ValueError("need at least two task representations for overlap")
    fn = linear_cka if measure == "cka" else cosine_overlap
    values = [fn(reps[a], reps[b]) for a, b in itertools.combinations(range(len(reps)), 2)]
    return sum(values) / len(values)
