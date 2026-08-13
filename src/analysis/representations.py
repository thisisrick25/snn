"""Extract hidden-layer-2 spike-count representations for overlap analysis.

Representations are taken from a FIXED held-out subset per task so that the same
inputs are compared across all sparsity levels and seeds. The representation of a
sample is the hidden-layer-2 spike counts summed over the simulation window,
i.e. ``ForwardTraces.h2_spike_counts`` of shape ``[batch, hidden2]``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset


def fixed_subset_loader(dataset, n_samples: int, batch_size: int = 128) -> DataLoader:
    """Deterministic first-``n_samples`` subset loader (no shuffling).

    Using a fixed, unshuffled subset guarantees the same probe inputs across all
    conditions, which is required for a fair overlap comparison.
    """
    n = min(n_samples, len(dataset))
    subset = Subset(dataset, list(range(n)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def extract_representation(model, loader, task_id: int, device: str = "cpu") -> torch.Tensor:
    """Return hidden-layer-2 spike counts for every sample in ``loader``.

    Output shape: ``[n_samples, hidden2]``. Model is put in eval mode (train
    state restored afterwards) and weights are not updated. Representations are
    moved back to CPU so downstream overlap/CKA analysis stays on CPU regardless
    of where the model ran.
    """
    was_training = model.training
    model.eval()
    reps: list[torch.Tensor] = []
    try:
        for batch in loader:
            x = batch[0].to(device)
            _, traces = model(x, task_id, return_traces=True)
            if traces is None:
                raise RuntimeError("model did not return traces; pass return_traces=True")
            reps.append(traces.h2_spike_counts.detach().cpu())
    finally:
        if was_training:
            model.train()
    if not reps:
        raise ValueError("loader produced no batches")
    return torch.cat(reps, dim=0)
