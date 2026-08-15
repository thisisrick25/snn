from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def holm_bonferroni(pvals: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(pvals, dtype=float)
    adjusted = np.empty_like(values, dtype=float)
    flat = values.ravel()
    order = np.argsort(flat)
    sorted_p = flat[order]
    running = 0.0
    sorted_adj = np.empty_like(sorted_p, dtype=float)
    m = sorted_p.size
    for rank, pval in enumerate(sorted_p, start=1):
        running = max(running, float((m - rank + 1) * pval))
        sorted_adj[rank - 1] = min(running, 1.0)
    restored = np.empty_like(sorted_adj, dtype=float)
    restored[order] = sorted_adj
    adjusted[...] = restored.reshape(values.shape)
    return adjusted


def benjamini_hochberg(pvals: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(pvals, dtype=float)
    adjusted = np.empty_like(values, dtype=float)
    flat = values.ravel()
    order = np.argsort(flat)
    sorted_p = flat[order]
    m = sorted_p.size
    sorted_adj = np.empty_like(sorted_p, dtype=float)
    running = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        running = min(running, float(m * sorted_p[index] / rank))
        sorted_adj[index] = min(running, 1.0)
    restored = np.empty_like(sorted_adj, dtype=float)
    restored[order] = sorted_adj
    adjusted[...] = restored.reshape(values.shape)
    return adjusted
