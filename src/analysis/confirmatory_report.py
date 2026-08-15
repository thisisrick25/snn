from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from src.analysis.confirmatory_types import ConfirmatorySettingResult, ConfirmatoryStudyResult, MECHANISMS
from src.analysis.multiple_comparisons import benjamini_hochberg, holm_bonferroni

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def confirmatory_json(primary: ConfirmatoryStudyResult, cosine: ConfirmatoryStudyResult | None = None) -> dict[str, JsonValue]:
    primary_tests = _primary_tests(primary)
    _attach_adjustments(primary_tests)
    report: dict[str, JsonValue] = {
        "mediator": primary.mediator,
        "standardization_note": "X, M, and Y z-scored within setting before fitting.",
        "theta_bar_study": _json_float(primary.theta_bar_study),
        "exploratory_note": _exploratory_note(primary),
        "primary_family": primary_tests,
        "settings": {name: _setting_json(result, primary_tests) for name, result in primary.settings.items()},
        "secondary_families": _secondary_families(primary, cosine),
    }
    return report


def _primary_tests(study: ConfirmatoryStudyResult) -> list[dict[str, JsonValue]]:
    tests: list[dict[str, JsonValue]] = []
    for setting, result in study.settings.items():
        tests.extend(
            [
                _test_row(setting, "H2_c_bar", "c_bar", result),
                _test_row(setting, "H3_u2", "u2", result),
                _test_row(setting, "H4_theta_bar", "theta_bar", result),
            ]
        )
    return tests


def _test_row(setting: str, hypothesis: str, key: str, result: ConfirmatorySettingResult) -> dict[str, JsonValue]:
    value = getattr(result.estimates, key)
    return {
        "setting": setting,
        "hypothesis": hypothesis,
        "estimand": key,
        "estimate": _json_float(float(value)),
        "ci": _ci_json(result.bootstrap.ci.get(key)),
        "p_raw": _json_float(_p_value(result, key)),
    }


def _attach_adjustments(tests: list[dict[str, JsonValue]]) -> None:
    pvals = np.asarray([_numeric_p(row["p_raw"]) for row in tests], dtype=float)
    holm = holm_bonferroni(pvals)
    bh = benjamini_hochberg(pvals)
    for row, holm_p, bh_p in zip(tests, holm, bh, strict=True):
        row["p_holm"] = _json_float(float(holm_p))
        row["p_bh"] = _json_float(float(bh_p))


def _setting_json(result: ConfirmatorySettingResult, primary_tests: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    primary_by_key = {row["estimand"]: row for row in primary_tests if row["setting"] == result.setting}
    return {
        "n_rows": result.n_rows,
        "n_seed_clusters": result.n_seed_clusters,
        "standardization_note": "within-setting z-score over live rows",
        "mechanisms": {mechanism: _mechanism_json(result, mechanism) for mechanism in MECHANISMS},
        "theta_bar_s": _estimand_json(result, "theta_bar", primary_by_key),
        "c_bar_s_H2": _estimand_json(result, "c_bar", primary_by_key),
        "u2_H3": _estimand_json(result, "u2", primary_by_key),
        "IMM": {
            "kwta_window": _estimand_json(result, "IMM_kwta_window", {}),
            "activity_reg": _estimand_json(result, "IMM_activity_reg", {}),
        },
        "diagnostics": _diagnostics_json(result),
    }


def _mechanism_json(result: ConfirmatorySettingResult, mechanism: str) -> dict[str, JsonValue]:
    return {
        "a": _json_float(result.estimates.a[mechanism]),
        "b": _json_float(result.estimates.b[mechanism]),
        "c_prime": _estimand_json(result, f"c_prime_{mechanism}", {}),
        "c": _estimand_json(result, f"c_{mechanism}", {}),
        "theta": _estimand_json(result, f"theta_{mechanism}", {}),
        "PM_descriptive": _json_float(result.estimates.pm[mechanism]),
    }


def _estimand_json(result: ConfirmatorySettingResult, key: str, adjusted: dict[JsonValue, dict[str, JsonValue]]) -> dict[str, JsonValue]:
    point = _point_for_key(result, key)
    row = adjusted.get(key, {})
    return {
        "estimate": _json_float(point),
        "ci": _ci_json(result.bootstrap.ci.get(key)),
        "p_raw": row.get("p_raw", _json_float(_p_value(result, key))),
        "p_holm": row.get("p_holm"),
        "p_bh": row.get("p_bh"),
    }


def _secondary_families(primary: ConfirmatoryStudyResult, cosine: ConfirmatoryStudyResult | None) -> dict[str, JsonValue]:
    families: dict[str, JsonValue] = {
        "mechanism_indirect_exploratory": _family_rows(primary, [f"theta_{mechanism}" for mechanism in MECHANISMS]),
        "moderation_exploratory": _family_rows(primary, ["IMM_kwta_window", "IMM_activity_reg"]),
    }
    if cosine is not None:
        families["overlap_cosine_secondary_mediator_exploratory"] = _primary_tests_with_adjustments(cosine)
    return families


def _family_rows(study: ConfirmatoryStudyResult, keys: list[str]) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for setting, result in study.settings.items():
        for key in keys:
            rows.append({"setting": setting, "estimand": key, "estimate": _json_float(_point_for_key(result, key)), "ci": _ci_json(result.bootstrap.ci.get(key)), "p_raw": _json_float(_p_value(result, key))})
    _attach_adjustments(rows)
    return rows


def _primary_tests_with_adjustments(study: ConfirmatoryStudyResult) -> list[dict[str, JsonValue]]:
    rows = _primary_tests(study)
    _attach_adjustments(rows)
    return rows


def _point_for_key(result: ConfirmatorySettingResult, key: str) -> float:
    if key == "theta_bar":
        return result.estimates.theta_bar
    if key == "c_bar":
        return result.estimates.c_bar
    if key == "u2":
        return result.estimates.u2
    for mechanism in MECHANISMS:
        if key == f"theta_{mechanism}":
            return result.estimates.theta[mechanism]
        if key == f"c_{mechanism}":
            return result.estimates.c[mechanism]
        if key == f"c_prime_{mechanism}":
            return result.estimates.c_prime[mechanism]
    if key == "IMM_kwta_window":
        return result.estimates.imm["kwta_window"]
    if key == "IMM_activity_reg":
        return result.estimates.imm["activity_reg"]
    return float("nan")


def _p_value(result: ConfirmatorySettingResult, key: str) -> float:
    return result.bootstrap.p_value.get(key, 1.0)


def _numeric_p(value: JsonValue) -> float:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return 1.0


def _ci_json(value: tuple[float, float] | None) -> JsonValue:
    if value is None:
        return None
    return [_json_float(value[0]), _json_float(value[1])]


def _json_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _diagnostics_json(result: ConfirmatorySettingResult) -> dict[str, JsonValue]:
    diagnostics = {key: _jsonable(value) for key, value in result.estimates.diagnostics.items()}
    raw_flags = diagnostics.get("flags")
    flags = [str(item) for item in raw_flags] if isinstance(raw_flags, list) else []
    if result.bootstrap.high_failure_rate:
        flags.append("high_bootstrap_failure_rate")
    diagnostics["flags"] = flags
    diagnostics["failed_bootstrap_draws"] = result.bootstrap.failed_draws
    diagnostics["successful_bootstrap_draws"] = result.bootstrap.successful_draws
    diagnostics["high_bootstrap_failure_rate"] = result.bootstrap.high_failure_rate
    return diagnostics


def _jsonable(value: object) -> JsonValue:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_float(float(value))
    if isinstance(value, float):
        return _json_float(value)
    if isinstance(value, int | str | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _exploratory_note(study: ConfirmatoryStudyResult) -> bool:
    return any(result.n_seed_clusters < 8 for result in study.settings.values())
