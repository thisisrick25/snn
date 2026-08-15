from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.analysis.confirmatory_types import ConfirmatoryRow, OlsFit, SettingArrays
from src.analysis.mediation import _standardize


def setting_label(dataset: str, arch: str) -> str | None:
    if dataset == "mnist" and arch == "mlp":
        return "MNIST-MLP"
    if dataset == "cifar10" and arch == "conv_snn":
        return "CIFAR-Conv-SNN"
    return None


def rows_for_mediator(rows: Sequence[Mapping[str, str]], mediator: str) -> list[ConfirmatoryRow]:
    key = "overlap_cka" if mediator == "cka" else "overlap_cosine"
    parsed: list[ConfirmatoryRow] = []
    for row in rows:
        setting = setting_label(row["dataset"], row["arch"])
        if setting is None:
            continue
        parsed.append(
            ConfirmatoryRow(
                seed=row["seed"],
                setting=setting,
                mechanism=row["mechanism"],
                threshold=float(row["threshold"]),
                x=float(row["mean_observed_activity"]),
                mediator=float(row[key]),
                y=float(row["mean_forgetting"]),
            )
        )
    return parsed


def arrays_for_setting(rows: Sequence[ConfirmatoryRow]) -> SettingArrays:
    thresholds = np.array([row.threshold for row in rows], dtype=float)
    mechanisms = np.array([row.mechanism for row in rows], dtype=np.str_)
    return SettingArrays(
        seed=np.array([row.seed for row in rows], dtype=np.str_),
        mechanism=mechanisms,
        level=_levels(thresholds, mechanisms),
        x=_standardize(np.array([row.x for row in rows], dtype=float)),
        mediator=_standardize(np.array([row.mediator for row in rows], dtype=float)),
        y=_standardize(np.array([row.y for row in rows], dtype=float)),
    )


def model_design(arrays: SettingArrays, model: str) -> tuple[list[str], NDArray[np.float64]]:
    z1 = (arrays.mechanism == "kwta_window").astype(float)
    z2 = (arrays.mechanism == "activity_reg").astype(float)
    level_names, level_matrix = _level_dummies(arrays.level)
    base_names = ["intercept", *level_names, "Z_kwta", "Z_reg"]
    base_cols = [np.ones(arrays.x.size), *[level_matrix[:, i] for i in range(level_matrix.shape[1])], z1, z2]
    match model:
        case "mediator":
            names = [*base_names, "X", "X_Z_kwta", "X_Z_reg"]
            cols = [*base_cols, arrays.x, arrays.x * z1, arrays.x * z2]
        case "outcome":
            names = [*base_names, "X", "X_Z_kwta", "X_Z_reg", "M", "M_Z_kwta", "M_Z_reg"]
            cols = [*base_cols, arrays.x, arrays.x * z1, arrays.x * z2, arrays.mediator, arrays.mediator * z1, arrays.mediator * z2]
        case "total":
            names = [*base_names, "X", "X_Z_kwta", "X_Z_reg"]
            cols = [*base_cols, arrays.x, arrays.x * z1, arrays.x * z2]
        case "shape":
            names = [*base_names, "X", "X2"]
            cols = [*base_cols, arrays.x, arrays.x * arrays.x]
        case _:
            raise UnknownModelError(model)
    return names, np.column_stack(cols).astype(float)


def fit_named_ols(names: Sequence[str], design: NDArray[np.float64], y: NDArray[np.float64]) -> OlsFit:
    kept = _independent_columns(design)
    used = design[:, kept]
    coef, _, rank, _ = np.linalg.lstsq(used, y, rcond=None)
    values = {name: 0.0 for name in names}
    for idx, value in zip(kept, coef, strict=True):
        values[names[idx]] = float(value)
    dropped = tuple(name for idx, name in enumerate(names) if idx not in kept)
    return OlsFit(
        coefficients=values,
        rank=int(rank),
        n_columns=len(names),
        condition_number=_condition_number(used),
        dropped_columns=dropped,
    )


def vif_for_mediator(arrays: SettingArrays) -> float:
    names, design = model_design(arrays, "outcome")
    mediator_index = names.index("M")
    other = np.delete(design, mediator_index, axis=1)
    fit = fit_named_ols([name for name in names if name != "M"], other, design[:, mediator_index])
    predicted = np.zeros(design.shape[0], dtype=float)
    for idx, name in enumerate(name for name in names if name != "M"):
        predicted += other[:, idx] * fit.coefficients[name]
    residual = design[:, mediator_index] - predicted
    sse = float(np.sum(residual * residual))
    total = design[:, mediator_index] - float(np.mean(design[:, mediator_index]))
    sst = float(np.sum(total * total))
    if sst <= 0.0:
        return float("inf")
    r2 = max(0.0, min(1.0, 1.0 - sse / sst))
    if r2 >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - r2))


class UnknownModelError(ValueError):
    def __init__(self, model: str) -> None:
        super().__init__(f"unknown confirmatory model: {model}")


def _levels(thresholds: NDArray[np.float64], mechanisms: NDArray[np.str_]) -> NDArray[np.int64]:
    levels = np.zeros(thresholds.size, dtype=np.int64)
    for mechanism in np.unique(mechanisms):
        mask = mechanisms == mechanism
        unique = sorted(float(value) for value in np.unique(thresholds[mask]))
        index = {value: idx for idx, value in enumerate(unique)}
        for row_index in np.flatnonzero(mask):
            levels[row_index] = index[float(thresholds[row_index])]
    return levels


def _level_dummies(level: NDArray[np.int64]) -> tuple[list[str], NDArray[np.float64]]:
    unique = sorted(int(value) for value in np.unique(level))
    if len(unique) <= 1:
        return [], np.zeros((level.size, 0), dtype=float)
    names = [f"L_{value}" for value in unique[1:]]
    cols = [(level == value).astype(float) for value in unique[1:]]
    return names, np.column_stack(cols).astype(float)


def _independent_columns(design: NDArray[np.float64]) -> list[int]:
    kept: list[int] = []
    rank = 0
    for idx in range(design.shape[1]):
        trial = [*kept, idx]
        trial_rank = int(np.linalg.matrix_rank(design[:, trial]))
        if trial_rank > rank:
            kept.append(idx)
            rank = trial_rank
    return kept


def _condition_number(design: NDArray[np.float64]) -> float:
    if design.size == 0:
        return float("inf")
    singular = np.linalg.svd(design, compute_uv=False)
    if singular.size == 0 or singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])
