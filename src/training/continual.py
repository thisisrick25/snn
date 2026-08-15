"""Naive sequential continual learning over the Split-MNIST tasks.

Trains tasks in order with no forgetting-mitigation mechanism and, after each
task, evaluates every task learned so far. The result is a lower-triangular
accuracy matrix ``A`` where ``A[i, j]`` is the accuracy on task ``j`` after
training task ``i`` (rows = training stage, cols = evaluated task).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.training.evaluate import evaluate_task
from src.training.instrumentation import measure_activity
from src.training.train_task import train_task


@dataclass
class ContinualResult:
    """Outcome of one naive sequential run."""

    accuracy_matrix: list[list[float]]  # A[i][j], NaN where j > i (not yet learned)
    train_losses: list[list[float]] = field(default_factory=list)
    observed_activity_per_task: list[float] = field(default_factory=list)


def run_naive_sequential(
    model,
    train_loaders,
    test_loaders,
    *,
    epochs: int,
    lr: float,
    timesteps: int,
    activity_max_batches: int | None = 8,
    device: str = "cpu",
    control: dict | None = None,
    progress: str | None = None,
) -> ContinualResult:
    """Run naive sequential CL and fill the accuracy matrix.

    The threshold must already be set and frozen on ``model`` before this call.
    """
    n_tasks = len(train_loaders)
    nan = float("nan")
    accuracy_matrix = [[nan for _ in range(n_tasks)] for _ in range(n_tasks)]
    train_losses: list[list[float]] = []
    observed_activity: list[float] = []

    for i in range(n_tasks):
        task_progress = f"{progress} task {i + 1}/{n_tasks}" if progress is not None else None
        losses = train_task(model, train_loaders[i], task_id=i, epochs=epochs, lr=lr,
                            device=device, control=control, progress=task_progress)
        train_losses.append(losses)

        # Log observed activity on the just-trained task (uses frozen threshold).
        summary = measure_activity(
            model, test_loaders[i], task_id=i, timesteps=timesteps,
            max_batches=activity_max_batches, device=device,
        )
        observed_activity.append(summary.combined_activity)

        # Evaluate every task learned so far (j <= i).
        for j in range(i + 1):
            accuracy_matrix[i][j] = evaluate_task(model, test_loaders[j], task_id=j, device=device)

    return ContinualResult(
        accuracy_matrix=accuracy_matrix,
        train_losses=train_losses,
        observed_activity_per_task=observed_activity,
    )
