"""Train the LIF-SNN on a single task.

Naive sequential training: the optimizer only ever sees the current task's data.
The firing threshold is assumed already calibrated and frozen by the caller.
"""

from __future__ import annotations

import torch
from torch import nn


def train_task(model, loader, task_id: int, epochs: int, lr: float,
               device: str = "cpu") -> list[float]:
    """Train ``model`` on one task for ``epochs`` and return per-epoch mean loss.

    Uses Adam and cross-entropy over the spike-count logits of the task's head.
    Batches are moved to ``device``; the model is assumed already on ``device``.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []

    model.train()
    for _ in range(epochs):
        running = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, task_id)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1
        epoch_losses.append(running / max(n_batches, 1))

    return epoch_losses
