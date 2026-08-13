"""Evaluate per-task accuracy for the LIF-SNN.

Predictions are the argmax of the spike-count logits produced by the task's own
output head (task-incremental setting: the task id selects the head).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def evaluate_task(model, loader, task_id: int, device: str = "cpu") -> float:
    """Return classification accuracy on one task's test loader."""
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, task_id)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum())
        total += int(y.shape[0])

    if was_training:
        model.train()

    if total == 0:
        raise ValueError("Empty evaluation loader.")
    return correct / total
