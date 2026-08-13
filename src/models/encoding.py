"""Input encoding for the LIF-SNN pilot.

The pilot uses direct (rate-style) current injection: the flattened MNIST image
is fed unchanged as input current at every timestep. This is the simplest,
most reproducible scheme and avoids the variance of Poisson spike encoding.
"""

from __future__ import annotations

import torch


def repeat_over_time(x: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Repeat a static input across the time dimension.

    Args:
        x: input batch of shape ``[batch, features]``.
        timesteps: number of simulation steps ``T``.

    Returns:
        Tensor of shape ``[T, batch, features]`` where each timestep is a copy
        of ``x`` (direct current injection).
    """
    if x.dim() != 2:
        raise ValueError(f"expected [batch, features] input, got shape {tuple(x.shape)}")
    return x.unsqueeze(0).expand(timesteps, *x.shape)
