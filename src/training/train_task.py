"""Train the LIF-SNN on a single task.

Naive sequential training: the optimizer only ever sees the current task's data.
The firing threshold is assumed already calibrated and frozen by the caller.
"""

from __future__ import annotations

import torch
from torch import nn

from src.training.progress import iter_progress


def _frozen_grad_params(model):
    """Yield the fc1/fc2 weight+bias grads for the confound freeze/norm hooks.

    Confound controls (§3.3.4/§3.3.6) act on the two hidden FC layers, the same
    layers the sparsity mechanisms and the activity metric operate on.
    """
    for layer in (model.fc1, model.fc2):
        for p in (layer.weight, layer.bias):
            if p.grad is not None:
                yield p


def _apply_block_freeze(model, frozen_fraction: float) -> None:
    """Zero the gradient rows of a fixed random subset of fc1/fc2 output neurons.

    Freezing whole output neurons (rows of the weight matrix + bias entries)
    isolates capacity-partitioning: it removes the same *count* of updatable units
    a sparse network leaves unupdated, but as contiguous neuron blocks rather than
    a distributed sparse code (§3.3.6). The frozen set is drawn once and cached on
    the model so it is stable across tasks.
    """
    for layer in (model.fc1, model.fc2):
        width = layer.weight.shape[0]
        attr = f"_frozen_idx_{id(layer)}"
        idx = getattr(model, attr, None)
        if idx is None:
            n_frozen = int(round(frozen_fraction * width))
            g = torch.Generator(device="cpu").manual_seed(width)
            idx = torch.randperm(width, generator=g)[:n_frozen]
            setattr(model, attr, idx)
        if idx.numel() == 0:
            continue
        if layer.weight.grad is not None:
            layer.weight.grad[idx] = 0.0
        if layer.bias.grad is not None:
            layer.bias.grad[idx] = 0.0


def train_task(model, loader, task_id: int, epochs: int, lr: float,
               device: str = "cpu", *, control: dict | None = None,
               progress: str | None = None) -> list[float]:
    """Train ``model`` on one task for ``epochs`` and return per-epoch mean loss.

    Uses Adam and cross-entropy over the spike-count logits of the task's head.
    Batches are moved to ``device``; the model is assumed already on ``device``.

    ``control`` optionally selects a confound-control condition applied before each
    optimizer step:
    - ``{"kind": "update_norm", "max_norm": float}`` clips the total gradient norm
      to isolate "fewer/smaller updates" (§3.3.4).
    - ``{"kind": "block_freeze", "fraction": float}`` zeros the grads of a fixed
      random subset of hidden neurons to isolate capacity-partitioning (§3.3.6).
    - activation-dropout (§3.3.5) is a model-level control, not applied here.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []
    kind = control.get("kind") if control else None

    model.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        batches = loader
        if progress is not None:
            batches = iter_progress(loader, prefix=f"{progress} epoch {epoch + 1}/{epochs}")
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, task_id)
            loss = criterion(logits, y)
            penalty = getattr(model, "last_activity_penalty", None)
            if penalty is not None:
                loss = loss + penalty
            loss.backward()
            if control is not None and kind == "update_norm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), control["max_norm"])
            elif control is not None and kind == "block_freeze":
                _apply_block_freeze(model, control["fraction"])
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1
        epoch_losses.append(running / max(n_batches, 1))

    return epoch_losses
