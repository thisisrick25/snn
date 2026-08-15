"""Formal mediation analysis: does representational overlap mediate the
activity -> forgetting relationship, or do overlap and forgetting merely
co-vary with activity?

The pilot found that as spike activity rises, CKA overlap falls AND forgetting
falls. All three move together, so a raw overlap-forgetting correlation cannot
distinguish genuine mediation (activity -> less overlap -> less forgetting) from
spurious co-variation (activity independently drives both). This module fits the
standard three-regression mediation model to separate them.

Variables (one row per (seed, condition)):
    X = observed activity (treatment)
    M = representational overlap, e.g. CKA (candidate mediator)
    Y = forgetting (outcome)

Paths:
    total effect c   : Y ~ X
    a-path a         : M ~ X
    b-path b, direct c' : Y ~ X + M   (coef on M = b, coef on X = c')
    indirect effect  : a * b, with a percentile bootstrap CI

All variables are z-scored so the coefficients are on a common (standardized)
scale. Implemented with numpy/scipy only (no statsmodels dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MediationResult:
    """Outcome of one mediation analysis (standardized coefficients)."""

    n: int
    total_effect_c: float          # X -> Y
    a_path: float                  # X -> M
    b_path: float                  # M -> Y | X
    direct_effect_c_prime: float   # X -> Y | M
    indirect_effect: float         # a * b (point estimate)
    indirect_ci_low: float         # bootstrap percentile CI
    indirect_ci_high: float
    proportion_mediated: float     # a*b / c (NaN if c ~ 0)


def _standardize(v: np.ndarray) -> np.ndarray:
    std = v.std()
    if std == 0:
        return v - v.mean()
    return (v - v.mean()) / std


def _ols_slopes(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return OLS slope coefficients (excluding intercept) for ``y ~ design``.

    ``design`` holds the predictor columns without an intercept; an intercept
    column of ones is prepended here. Uses least squares.
    """
    x = np.column_stack([np.ones(len(y)), design])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef[1:]  # drop intercept


def _fit_paths(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit the mediation paths on already-standardized inputs.

    Returns (c, a, b, c_prime).
    """
    c = float(_ols_slopes(x.reshape(-1, 1), y)[0])       # Y ~ X
    a = float(_ols_slopes(x.reshape(-1, 1), m)[0])       # M ~ X
    bc = _ols_slopes(np.column_stack([x, m]), y)          # Y ~ X + M
    c_prime = float(bc[0])
    b = float(bc[1])
    return c, a, b, c_prime


def mediation_analysis(
    activity: np.ndarray,
    overlap: np.ndarray,
    forgetting: np.ndarray,
    *,
    n_boot: int = 5000,
    random_state: int = 0,
) -> MediationResult:
    """Run standardized mediation of activity -> overlap -> forgetting.

    Args:
        activity, overlap, forgetting: 1-D arrays of equal length (one entry per
            (seed, condition)).
        n_boot: bootstrap resamples for the indirect-effect CI.
        random_state: RNG seed for reproducibility.

    Returns:
        MediationResult with standardized path coefficients and a 95% percentile
        bootstrap CI on the indirect effect.
    """
    x = np.asarray(activity, dtype=float)
    m = np.asarray(overlap, dtype=float)
    y = np.asarray(forgetting, dtype=float)
    if not (len(x) == len(m) == len(y)):
        raise ValueError("activity, overlap, forgetting must have equal length")
    n = len(x)
    if n < 3:
        raise ValueError(f"need at least 3 observations for mediation, got {n}")

    xs, ms, ys = _standardize(x), _standardize(m), _standardize(y)
    c, a, b, c_prime = _fit_paths(xs, ms, ys)
    indirect = a * b

    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        pick = rng.choice(idx, size=n, replace=True)
        bx, bm, by = _standardize(x[pick]), _standardize(m[pick]), _standardize(y[pick])
        _, ba, bb, _ = _fit_paths(bx, bm, by)
        boot[i] = ba * bb
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    proportion = indirect / c if abs(c) > 1e-8 else float("nan")

    return MediationResult(
        n=n,
        total_effect_c=c,
        a_path=a,
        b_path=b,
        direct_effect_c_prime=c_prime,
        indirect_effect=indirect,
        indirect_ci_low=float(ci_low),
        indirect_ci_high=float(ci_high),
        proportion_mediated=float(proportion),
    )


from src.analysis.confirmatory_estimator import confirmatory_mediation_analysis  # noqa: E402
from src.analysis.confirmatory_types import (  # noqa: E402
    BootstrapSummary,
    ConfirmatorySettingResult,
    ConfirmatoryStudyResult,
    PointEstimates,
)
