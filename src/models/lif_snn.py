"""Leaky integrate-and-fire SNN for the Split-MNIST pilot.

Architecture: 784 -> Linear -> LIF(256) -> Linear -> LIF(256) -> per-task Linear head.

Design decisions (see RESEARCH_REPORT.md / EXPERIMENT_PROTOCOL.md):
- Direct/rate encoding: the static 784-vector is injected as current every timestep.
- Two hidden LIF layers of 256 neurons each, snntorch ``Leaky`` with fast-sigmoid surrogate.
- Output readout = spike-count argmax: the per-task linear head is applied to the
  summed hidden-layer-2 spikes over the T-window; cross-entropy is computed on that.
- The forward pass can optionally return hidden spike traces for instrumentation
  (activity %) and representation extraction (hidden-layer-2 spike counts for overlap).

Two sparsity mechanisms, selected by ``sparsity_mode``:
- ``"threshold"``: one global scalar firing threshold shared across both LIF layers.
  A fixed threshold does NOT reliably control the over-window active fraction in a
  trained SNN (weights grow until activity re-saturates), so this mode is kept only
  as a baseline/fallback.
- ``"kwta_window"``: per-sample, per-layer winner-take-all with a winner set fixed
  for the whole T-window. A no-grad scoring pass ranks neurons by their summed
  membrane drive over the window and keeps the top-k; the real pass gates spikes and
  membrane to that fixed set every timestep. This guarantees the over-window active
  fraction per layer is <= k/width (an upper bound, not exact equality, since a kept
  neuron may still never cross threshold), giving a controlled monotone sparsity axis.
  Always analyse by the MEASURED activity, using k as the controlled condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from .encoding import repeat_over_time
from .heads import build_heads


@dataclass
class ForwardTraces:
    """Optional diagnostic outputs from a forward pass."""

    h1_spike_counts: torch.Tensor  # [batch, hidden1] spikes summed over time
    h2_spike_counts: torch.Tensor  # [batch, hidden2] spikes summed over time


class LifSnn(nn.Module):
    """Feedforward LIF-SNN with per-task binary heads and a global threshold."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_tasks: int,
        n_classes_per_task: int,
        beta: float,
        threshold: float,
        timesteps: int,
        v_reset: float = 0.0,
        sparsity_mode: str = "threshold",
        k_per_layer: list[int] | None = None,
        activity_reg_lambda: float = 0.0,
        activity_target: float = 0.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError(f"pilot expects exactly 2 hidden layers, got {hidden_dims}")
        if sparsity_mode not in ("threshold", "kwta_window", "activity_reg"):
            raise ValueError(f"unknown sparsity_mode {sparsity_mode!r}")

        self.timesteps = int(timesteps)
        self._threshold = float(threshold)
        self.beta = float(beta)
        self.sparsity_mode = sparsity_mode
        # k_per_layer[i] = number of neurons allowed to fire in hidden layer i under
        # kwta_window mode. None otherwise.
        self.k_per_layer = list(k_per_layer) if k_per_layer is not None else None
        # activity_reg mode: penalise squared deviation of the mean FC firing rate
        # from a target, so the optimiser is pushed toward (not just below) a level.
        self.activity_reg_lambda = float(activity_reg_lambda)
        self.activity_target = float(activity_target)
        # Set each forward pass; the trainer adds it to the loss in activity_reg mode.
        self.last_activity_penalty: torch.Tensor = torch.zeros(())
        # activation_dropout confound control: random per-unit dropout on the FC
        # spike-count features, matched to a target active fraction on a dense model.
        self.dropout_p = float(dropout_p)

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.lif1 = snn.Leaky(
            beta=self.beta, threshold=self._threshold, spike_grad=spike_grad, reset_mechanism="zero"
        )
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lif2 = snn.Leaky(
            beta=self.beta, threshold=self._threshold, spike_grad=spike_grad, reset_mechanism="zero"
        )
        self.heads = build_heads(hidden_dims[1], n_tasks, n_classes_per_task)

    @property
    def threshold(self) -> float:
        """Current global firing threshold shared by all LIF layers."""
        return self._threshold

    def set_threshold(self, value: float) -> None:
        """Set the global scalar threshold on every LIF layer (used by calibration).

        snntorch registers ``threshold`` as a buffer, so we write into the existing
        buffer tensor in-place rather than assigning a Python float.
        """
        self._threshold = float(value)
        with torch.no_grad():
            self.lif1.threshold.fill_(float(value))
            self.lif2.threshold.fill_(float(value))

    def forward(
        self, x: torch.Tensor, task_id: int, return_traces: bool = False
    ) -> tuple[torch.Tensor, ForwardTraces | None]:
        """Run the SNN over T timesteps and read out with the task head.

        Args:
            x: input batch ``[batch, input_dim]``.
            task_id: which per-task head to use (task-incremental).
            return_traces: if True, also return hidden spike-count traces.

        Returns:
            (logits, traces) where logits is ``[batch, n_classes_per_task]`` from the
            spike-count readout, and traces is ``ForwardTraces`` or ``None``.
        """
        current = repeat_over_time(x, self.timesteps)  # [T, batch, input_dim]

        if self.sparsity_mode == "kwta_window":
            mask1, mask2 = self._winner_masks(current)
        else:
            mask1 = mask2 = None

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        h1_sum = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        h2_sum = torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)

        for t in range(self.timesteps):
            cur1 = self.fc1(current[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            if mask1 is not None:
                spk1 = spk1 * mask1
                mem1 = mem1 * mask1
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            if mask2 is not None:
                spk2 = spk2 * mask2
                mem2 = mem2 * mask2
            h1_sum = h1_sum + spk1
            h2_sum = h2_sum + spk2

        if self.dropout_p > 0.0:
            h1_sum = torch.nn.functional.dropout(h1_sum, p=self.dropout_p, training=self.training)
            h2_sum = torch.nn.functional.dropout(h2_sum, p=self.dropout_p, training=self.training)

        logits = self.heads[task_id](h2_sum)  # spike-count readout

        if self.sparsity_mode == "activity_reg":
            mean_rate = torch.cat([h1_sum, h2_sum], dim=1).mean() / self.timesteps
            self.last_activity_penalty = self.activity_reg_lambda * (mean_rate - self.activity_target) ** 2
        else:
            self.last_activity_penalty = torch.zeros((), device=x.device)

        traces = ForwardTraces(h1_spike_counts=h1_sum, h2_spike_counts=h2_sum) if return_traces else None
        return logits, traces

    @torch.no_grad()
    def _winner_masks(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a fixed per-sample winner set for each hidden layer.

        Scores neurons by their summed membrane potential over the whole T-window in
        a gradient-free pass, then keeps the top-k per layer. The returned masks are
        detached {0,1} tensors of shape ``[batch, width]``; multiplying spikes and
        membrane by them each timestep gates the network to those winners without
        being a differentiable parameter (top-k is a routing decision).
        """
        assert self.k_per_layer is not None
        k1, k2 = self.k_per_layer

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        score1 = torch.zeros(current.shape[1], self.fc1.out_features, device=current.device)
        score2 = torch.zeros(current.shape[1], self.fc2.out_features, device=current.device)

        for t in range(self.timesteps):
            cur1 = self.fc1(current[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            score1 = score1 + mem1
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            score2 = score2 + mem2

        return self._topk_mask(score1, k1), self._topk_mask(score2, k2)

    @staticmethod
    def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        """Return a {0,1} mask [batch, width] keeping the top-k scores per row."""
        width = scores.shape[1]
        k = max(1, min(int(k), width))
        mask = torch.zeros_like(scores)
        idx = scores.topk(k, dim=1).indices
        mask.scatter_(1, idx, 1.0)
        return mask


def build_model(cfg: dict, condition: float) -> LifSnn:
    """Construct a LifSnn from a pilot config dict for one sweep condition.

    ``condition`` is interpreted by ``sparsity_mode``:
    - threshold mode: the firing threshold value.
    - kwta_window mode: the target active fraction, converted to a per-layer k
      (round(fraction * width), floored at 1).
    """
    mode = cfg.get("sparsity_mode", "threshold")
    hidden_dims = list(cfg["hidden_dims"])
    activity_reg_lambda = 0.0
    activity_target = 0.0

    if mode == "kwta_window":
        k_per_layer = [max(1, round(condition * w)) for w in hidden_dims]
        threshold = float(cfg.get("threshold", 1.0))
    elif mode == "activity_reg":
        k_per_layer = None
        threshold = float(cfg.get("threshold", 1.0))
        activity_reg_lambda = float(cfg.get("activity_reg_lambda", 1.0))
        activity_target = condition
    else:
        k_per_layer = None
        threshold = condition

    return LifSnn(
        input_dim=cfg["input_dim"],
        hidden_dims=hidden_dims,
        n_tasks=len(cfg["tasks"]),
        n_classes_per_task=cfg["n_classes_per_task"],
        beta=cfg["beta"],
        threshold=threshold,
        timesteps=cfg["timesteps"],
        v_reset=cfg.get("v_reset", 0.0),
        sparsity_mode=mode,
        k_per_layer=k_per_layer,
        activity_reg_lambda=activity_reg_lambda,
        activity_target=activity_target,
        dropout_p=float(cfg.get("control_dropout_p", 0.0)),
    )
