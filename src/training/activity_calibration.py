"""Per-target lambda search for the activity-regularization mechanism.

The activity-regularization penalty is a soft nudge, so a single fixed penalty
weight (lambda) cannot land every target activity level. This module searches
lambda for one target: it briefly trains an activity_reg model on task 0,
measures the resulting combined activity, and multiplicatively adjusts lambda
(raise it when activity overshoots the target, lower it when it undershoots)
until the observed activity is within a tolerance band or the iteration budget
is exhausted. It returns the chosen lambda; the caller writes it into the config
so build_model / build_conv_model pick it up for the real run.

Because the penalty acts through the loss, calibration is only meaningful with a
few real training epochs; under a one-epoch smoke test it will not converge and
should be treated as a plumbing check only.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.training.seeds import set_seed
from src.training.train_task import train_task
from src.training.instrumentation import measure_activity


_LAMBDA_MIN = 1e-4
_LAMBDA_MAX = 1e4


@dataclass
class CalibrationResult:
    """Outcome of a lambda search for one target activity."""

    target: float
    chosen_lambda: float
    observed_activity: float
    iterations: int


def calibrate_lambda(
    cfg: dict,
    build_model_fn,
    train_loader,
    activity_loader,
    *,
    target: float,
    device: str = "cpu",
    warmup_epochs: int = 2,
    max_iters: int = 5,
    tolerance: float = 0.02,
    activity_max_batches: int = 8,
) -> CalibrationResult:
    """Search activity_reg_lambda so observed activity approaches ``target``.

    Args:
        cfg: pilot config dict (read for model hyperparameters; not mutated here).
        build_model_fn: build_model or build_conv_model; called as fn(cfg, target).
        train_loader: task-0 loader used for the warm-up passes.
        activity_loader: loader used to measure observed activity.
        target: desired combined active-neuron fraction.
        device: 'cpu' or 'cuda'.
        warmup_epochs: real training epochs per lambda trial.
        max_iters: maximum lambda adjustments.
        tolerance: acceptable absolute gap between observed and target activity.
        activity_max_batches: batches used to estimate activity.

    Returns:
        CalibrationResult with the chosen lambda and the activity it produced.
    """
    timesteps = int(cfg["timesteps"])
    lr = float(cfg["lr"])
    lam = float(cfg.get("activity_reg_lambda", 1.0))

    observed = float("nan")
    used_iters = 0
    for used_iters in range(1, max_iters + 1):
        trial_cfg = dict(cfg)
        trial_cfg["activity_reg_lambda"] = lam

        set_seed(int(cfg["seeds"][0]) if "seeds" in cfg else 0)
        model = build_model_fn(trial_cfg, target)
        model.to(device)
        train_task(model, train_loader, task_id=0, epochs=warmup_epochs, lr=lr, device=device)
        summary = measure_activity(
            model, activity_loader, task_id=0, timesteps=timesteps,
            max_batches=activity_max_batches, device=device,
        )
        observed = summary.combined_activity

        if abs(observed - target) <= tolerance:
            break

        # Overshoot (too much activity) -> stronger penalty; undershoot -> weaker.
        ratio = (observed + 1e-6) / (target + 1e-6)
        lam = min(_LAMBDA_MAX, max(_LAMBDA_MIN, lam * ratio))

    return CalibrationResult(
        target=float(target),
        chosen_lambda=float(lam),
        observed_activity=float(observed),
        iterations=used_iters,
    )
