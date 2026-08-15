from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.analysis.confirmatory_design import arrays_for_setting, fit_named_ols, model_design, rows_for_mediator, vif_for_mediator
from src.analysis.confirmatory_types import (
    MECHANISM_EPS,
    MECHANISMS,
    BootstrapStore,
    BootstrapSummary,
    ConfirmatoryRow,
    ConfirmatorySettingResult,
    ConfirmatoryStudyResult,
    FloatMap,
    OlsFit,
    PointEstimates,
    SettingArrays,
)


def confirmatory_mediation_analysis(
    rows: Sequence[Mapping[str, str]],
    *,
    mediator: str = "cka",
    n_boot: int = 10_000,
    random_state: int = 0,
) -> ConfirmatoryStudyResult:
    parsed = rows_for_mediator(rows, mediator)
    grouped = _group_by_setting(parsed)
    settings: dict[str, ConfirmatorySettingResult] = {}
    rng = np.random.default_rng(random_state)
    for setting, setting_rows in grouped.items():
        arrays = arrays_for_setting(setting_rows)
        estimates = estimate_setting(arrays)
        bootstrap = bootstrap_setting(arrays, n_boot=n_boot, rng=rng)
        settings[setting] = ConfirmatorySettingResult(
            setting=setting,
            n_rows=len(setting_rows),
            n_seed_clusters=len(np.unique(arrays.seed)),
            estimates=estimates,
            bootstrap=bootstrap,
        )
    theta_values = [result.estimates.theta_bar for result in settings.values()]
    theta_bar_study = float(np.mean(theta_values)) if theta_values else float("nan")
    return ConfirmatoryStudyResult(mediator=mediator, settings=settings, theta_bar_study=theta_bar_study)


def estimate_setting(arrays: SettingArrays) -> PointEstimates:
    mediator_fit = _fit_model(arrays, "mediator", arrays.mediator)
    outcome_fit = _fit_model(arrays, "outcome", arrays.y)
    total_fit = _fit_model(arrays, "total", arrays.y)
    shape_fit = _fit_model(arrays, "shape", arrays.y)
    counts = {mechanism: int(np.sum(arrays.mechanism == mechanism)) for mechanism in MECHANISMS}
    a = _mechanism_paths(mediator_fit.coefficients, "X", "X_Z_kwta", "X_Z_reg", counts)
    b = _mechanism_paths(outcome_fit.coefficients, "M", "M_Z_kwta", "M_Z_reg", counts)
    c_prime = _mechanism_paths(outcome_fit.coefficients, "X", "X_Z_kwta", "X_Z_reg", counts)
    c = _mechanism_paths(total_fit.coefficients, "X", "X_Z_kwta", "X_Z_reg", counts)
    theta = {mechanism: a[mechanism] * b[mechanism] for mechanism in MECHANISMS}
    theta_bar = _available_mean(theta)
    c_bar = _available_mean(c)
    diagnostics = _diagnostics(arrays, mediator_fit, outcome_fit, total_fit, shape_fit)
    return PointEstimates(
        a=a,
        b=b,
        c_prime=c_prime,
        c=c,
        theta=theta,
        pm=_proportion_mediated(theta, c),
        theta_bar=theta_bar,
        c_bar=c_bar,
        u2=float(shape_fit.coefficients["X2"]),
        imm={"kwta_window": theta["kwta_window"] - theta["threshold"], "activity_reg": theta["activity_reg"] - theta["threshold"]},
        fits={"mediator": mediator_fit, "outcome": outcome_fit, "total": total_fit, "shape": shape_fit},
        diagnostics=diagnostics,
    )


def bootstrap_setting(arrays: SettingArrays, *, n_boot: int, rng: np.random.Generator) -> BootstrapSummary:
    seeds = np.unique(arrays.seed)
    store = _empty_store()
    failed = 0
    for _ in range(n_boot):
        draw = rng.choice(seeds, size=seeds.size, replace=True)
        index = np.concatenate([np.flatnonzero(arrays.seed == seed) for seed in draw])
        sample = _take_arrays(arrays, index)
        if _missing_mechanism(sample):
            failed += 1
            continue
        estimates = estimate_setting(sample)
        _append_estimates(store, estimates)
    ci = {name: _percentile_ci(values) for name, values in store.items() if values}
    p_value = {name: _bootstrap_p(values) for name, values in store.items() if values}
    return BootstrapSummary(
        ci=ci,
        p_value=p_value,
        failed_draws=failed,
        successful_draws=n_boot - failed,
        high_failure_rate=failed > n_boot / 2,
    )


def _fit_model(arrays: SettingArrays, model: str, y: np.ndarray) -> OlsFit:
    names, design = model_design(arrays, model)
    # Pilot collinearity can make planned fixed effects redundant; keep a full-rank subset.
    return fit_named_ols(names, design, y)


def _mechanism_paths(coef: FloatMap, base: str, kwta_delta: str, reg_delta: str, counts: dict[str, int]) -> FloatMap:
    paths = {
        "threshold": float(coef[base]),
        "kwta_window": float(coef[base] + coef[kwta_delta]),
        "activity_reg": float(coef[base] + coef[reg_delta]),
    }
    return {mechanism: paths[mechanism] if counts[mechanism] > 0 else float("nan") for mechanism in MECHANISMS}


def _proportion_mediated(theta: FloatMap, c: FloatMap) -> FloatMap:
    pm: FloatMap = {}
    for mechanism in MECHANISMS:
        compatible = theta[mechanism] * c[mechanism] > 0.0 and abs(c[mechanism]) > MECHANISM_EPS
        pm[mechanism] = float(theta[mechanism] / c[mechanism]) if compatible else float("nan")
    return pm


def _available_mean(values: FloatMap) -> float:
    observed = np.asarray([values[mechanism] for mechanism in MECHANISMS], dtype=float)
    if bool(np.all(np.isnan(observed))):
        return float("nan")
    return float(np.nanmean(observed))


def _diagnostics(
    arrays: SettingArrays,
    mediator_fit: OlsFit,
    outcome_fit: OlsFit,
    total_fit: OlsFit,
    shape_fit: OlsFit,
) -> dict[str, int | float | bool | str | list[str] | dict[str, int] | dict[str, list[str]]]:
    vif = vif_for_mediator(arrays)
    kappa = max(fit.condition_number for fit in (mediator_fit, outcome_fit, total_fit, shape_fit))
    mechanism_counts = {mechanism: int(np.sum(arrays.mechanism == mechanism)) for mechanism in MECHANISMS}
    flags = [flag for flag, active in {
        "low_seed_clusters": len(np.unique(arrays.seed)) < 8,
        "high_vif_m": vif > 10.0,
        "severe_vif_m": vif > 20.0,
        "severe_condition_number": kappa > 100.0,
        "missing_mechanism": any(count == 0 for count in mechanism_counts.values()),
    }.items() if active]
    return {
        "VIF_M": float(vif),
        "condition_number_kappa": float(kappa),
        "mechanism_counts": mechanism_counts,
        "ranks": {name: fit.rank for name, fit in {"mediator": mediator_fit, "outcome": outcome_fit, "total": total_fit, "shape": shape_fit}.items()},
        "n_columns": {name: fit.n_columns for name, fit in {"mediator": mediator_fit, "outcome": outcome_fit, "total": total_fit, "shape": shape_fit}.items()},
        "dropped_columns": {name: list(fit.dropped_columns) for name, fit in {"mediator": mediator_fit, "outcome": outcome_fit, "total": total_fit, "shape": shape_fit}.items()},
        "flags": flags,
    }


def _empty_store() -> BootstrapStore:
    names = ["theta_bar", "c_bar", "u2"]
    names.extend(f"theta_{mechanism}" for mechanism in MECHANISMS)
    names.extend(f"c_{mechanism}" for mechanism in MECHANISMS)
    names.extend(f"c_prime_{mechanism}" for mechanism in MECHANISMS)
    names.extend(("IMM_kwta_window", "IMM_activity_reg"))
    return {name: [] for name in names}


def _append_estimates(store: BootstrapStore, estimates: PointEstimates) -> None:
    store["theta_bar"].append(estimates.theta_bar)
    store["c_bar"].append(estimates.c_bar)
    store["u2"].append(estimates.u2)
    for mechanism in MECHANISMS:
        store[f"theta_{mechanism}"].append(estimates.theta[mechanism])
        store[f"c_{mechanism}"].append(estimates.c[mechanism])
        store[f"c_prime_{mechanism}"].append(estimates.c_prime[mechanism])
    store["IMM_kwta_window"].append(estimates.imm["kwta_window"])
    store["IMM_activity_reg"].append(estimates.imm["activity_reg"])


def _group_by_setting(rows: Sequence[ConfirmatoryRow]) -> dict[str, list[ConfirmatoryRow]]:
    grouped: dict[str, list[ConfirmatoryRow]] = {}
    for row in rows:
        grouped.setdefault(row.setting, []).append(row)
    return grouped


def _take_arrays(arrays: SettingArrays, index: np.ndarray) -> SettingArrays:
    return SettingArrays(
        seed=arrays.seed[index],
        mechanism=arrays.mechanism[index],
        level=arrays.level[index],
        x=arrays.x[index],
        mediator=arrays.mediator[index],
        y=arrays.y[index],
    )


def _missing_mechanism(arrays: SettingArrays) -> bool:
    return any(np.sum(arrays.mechanism == mechanism) == 0 for mechanism in MECHANISMS)


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    # Percentile CI is primary because seed clusters are too few for stable BCa.
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_p(values: Sequence[float]) -> float:
    samples = np.asarray(values, dtype=float)
    lower = (1 + int(np.sum(samples <= 0.0))) / (samples.size + 1)
    upper = (1 + int(np.sum(samples >= 0.0))) / (samples.size + 1)
    return min(2.0 * min(lower, upper), 1.0)
