from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

MECHANISMS: tuple[str, str, str] = ("threshold", "kwta_window", "activity_reg")
SETTING_LABELS: tuple[str, str] = ("MNIST-MLP", "CIFAR-Conv-SNN")
MECHANISM_EPS = 1.0e-12

type FloatMap = dict[str, float]
type StringFloatMap = dict[str, FloatMap]
type DiagnosticValue = int | float | bool | str | list[str] | dict[str, int] | dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class ConfirmatoryRow:
    seed: str
    setting: str
    mechanism: str
    threshold: float
    x: float
    mediator: float
    y: float


@dataclass(frozen=True, slots=True)
class OlsFit:
    coefficients: FloatMap
    rank: int
    n_columns: int
    condition_number: float
    dropped_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SettingArrays:
    seed: NDArray[np.str_]
    mechanism: NDArray[np.str_]
    level: NDArray[np.int64]
    x: NDArray[np.float64]
    mediator: NDArray[np.float64]
    y: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PointEstimates:
    a: FloatMap
    b: FloatMap
    c_prime: FloatMap
    c: FloatMap
    theta: FloatMap
    pm: FloatMap
    theta_bar: float
    c_bar: float
    u2: float
    imm: FloatMap
    fits: dict[str, OlsFit]
    diagnostics: dict[str, DiagnosticValue]


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    ci: dict[str, tuple[float, float]]
    p_value: FloatMap
    failed_draws: int
    successful_draws: int
    high_failure_rate: bool


@dataclass(frozen=True, slots=True)
class ConfirmatorySettingResult:
    setting: str
    n_rows: int
    n_seed_clusters: int
    estimates: PointEstimates
    bootstrap: BootstrapSummary


@dataclass(frozen=True, slots=True)
class ConfirmatoryStudyResult:
    mediator: str
    settings: dict[str, ConfirmatorySettingResult]
    theta_bar_study: float


type BootstrapStore = dict[str, list[float]]
