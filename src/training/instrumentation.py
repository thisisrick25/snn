"""Spike-activity instrumentation for the LIF-SNN pilot.

The pilot's calibration target metric is the *active-neuron percentage*: the
fraction of hidden neurons that spike at least once over the full simulation
window, measured across BOTH hidden layers combined (denominator = sum of the
two hidden widths). We also expose per-layer activity, total spike count, and
average spike rate for logging.

All functions consume the per-sample spike-count tensors produced by
``LifSnn.forward(..., return_traces=True)`` (shape ``[batch, hidden]`` where each
entry is the number of spikes that neuron emitted over the T-step window).
A neuron is "active" for a sample if its spike count is > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.lif_snn import ForwardTraces


@dataclass
class ActivitySummary:
    """Aggregated spike-activity statistics for one batch (or accumulated set).

    Attributes:
        combined_activity: fraction of hidden neurons (both layers pooled) that
            spike >= 1 time over the window, averaged over samples. In [0, 1].
        layer1_activity: same fraction restricted to hidden layer 1.
        layer2_activity: same fraction restricted to hidden layer 2.
        total_spike_count: total number of spikes emitted across both hidden
            layers, summed over the batch.
        mean_spike_rate: average spikes per hidden neuron per timestep.
    """

    combined_activity: float
    layer1_activity: float
    layer2_activity: float
    total_spike_count: float
    mean_spike_rate: float


def _fraction_active(spike_counts: torch.Tensor) -> float:
    """Mean over samples of (fraction of neurons with spike count > 0).

    Args:
        spike_counts: tensor of shape [batch, hidden], non-negative spike counts.

    Returns:
        Scalar in [0, 1].
    """
    if spike_counts.ndim != 2:
        raise ValueError(
            f"expected spike_counts of shape [batch, hidden], got {tuple(spike_counts.shape)}"
        )
    active = (spike_counts > 0).float()  # [batch, hidden]
    per_sample = active.mean(dim=1)  # fraction of neurons active, per sample
    return float(per_sample.mean().item())


def summarize_traces(traces: ForwardTraces, timesteps: int) -> ActivitySummary:
    """Compute an :class:`ActivitySummary` from one forward pass' traces.

    Args:
        traces: ForwardTraces with h1_spike_counts and h2_spike_counts, each
            shaped [batch, hidden_i], holding spikes-over-window per neuron.
        timesteps: number of simulation steps T (for the spike-rate denominator).

    Returns:
        ActivitySummary. ``combined_activity`` pools both layers so the
        denominator is (hidden1 + hidden2) neurons, matching the pilot's
        active-neuron-percentage definition.
    """
    h1 = traces.h1_spike_counts
    h2 = traces.h2_spike_counts
    if h1.ndim != 2 or h2.ndim != 2:
        raise ValueError("trace tensors must be 2D [batch, hidden]")
    if h1.shape[0] != h2.shape[0]:
        raise ValueError("h1 and h2 must share the batch dimension")

    combined = torch.cat([h1, h2], dim=1)  # [batch, hidden1 + hidden2]

    batch = combined.shape[0]
    n_hidden = combined.shape[1]
    total_spikes = float(combined.sum().item())
    # spikes per neuron per timestep, averaged over the batch
    mean_rate = total_spikes / (batch * n_hidden * timesteps)

    return ActivitySummary(
        combined_activity=_fraction_active(combined),
        layer1_activity=_fraction_active(h1),
        layer2_activity=_fraction_active(h2),
        total_spike_count=total_spikes,
        mean_spike_rate=mean_rate,
    )


@torch.no_grad()
def measure_activity(model, loader, task_id: int, timesteps: int, max_batches: int | None = None, device: str = "cpu") -> ActivitySummary:
    """Measure average activity of ``model`` over ``loader`` in eval mode.

    Runs the model without gradient tracking or weight updates and averages the
    activity statistics across batches. Used both for calibration (finding the
    threshold that hits a target activity) and for logging observed activity
    during the continual-learning run.

    Args:
        model: a LifSnn instance.
        loader: DataLoader yielding (x [batch, input_dim], y).
        task_id: which task head to route through (activity is head-independent,
            but forward requires a valid head index).
        timesteps: T, for the spike-rate denominator.
        max_batches: if set, stop after this many batches (keeps calibration fast).

    Returns:
        An ActivitySummary averaged across the visited batches.
    """
    was_training = model.training
    model.eval()

    combined_sum = 0.0
    l1_sum = 0.0
    l2_sum = 0.0
    total_spikes = 0.0
    rate_sum = 0.0
    n = 0
    for i, (x, _y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        _logits, traces = model(x, task_id, return_traces=True)
        s = summarize_traces(traces, timesteps)
        combined_sum += s.combined_activity
        l1_sum += s.layer1_activity
        l2_sum += s.layer2_activity
        total_spikes += s.total_spike_count
        rate_sum += s.mean_spike_rate
        n += 1

    if was_training:
        model.train()

    if n == 0:
        raise ValueError("loader yielded no batches")

    return ActivitySummary(
        combined_activity=combined_sum / n,
        layer1_activity=l1_sum / n,
        layer2_activity=l2_sum / n,
        total_spike_count=total_spikes,  # summed over visited batches
        mean_spike_rate=rate_sum / n,
    )
