"""Convolutional spiking network for the Split-CIFAR pilot.

A spiking conv frontend (3 Conv2d + LIF + MaxPool blocks) extracts features from
raw [B, 3, 32, 32] images, which then feed the SAME two 256-unit fully-connected
LIF hidden layers and per-task heads as the MLP model (LifSnn). The k-WTA sparsity
mechanism and the active-neuron activity metric operate ONLY on those two FC hidden
layers, exactly as in the MLP path, so the sparsity axis is directly comparable
across MNIST-MLP, CIFAR-MLP, and CIFAR-Conv runs. The conv frontend is purely a
feature extractor: it is not traced, not gated by k-WTA, and not counted in activity.

Input is fed with direct/rate encoding: the same image is presented at every one
of the T simulation timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from .heads import build_heads

_CONV_FEATURE_DIM = 64 * 4 * 4  # 64 channels at 4x4 after three 2x pools from 32x32
_HIDDEN = 256


@dataclass
class ForwardTraces:
    """Hidden-layer spike counts (summed over time) for the two FC layers."""

    h1_spike_counts: torch.Tensor  # [batch, 256]
    h2_spike_counts: torch.Tensor  # [batch, 256]


class ConvSnn(nn.Module):
    """Spiking conv frontend + two FC LIF hidden layers with per-task heads."""

    def __init__(
        self,
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
        if sparsity_mode not in ("threshold", "kwta_window", "activity_reg"):
            raise ValueError(f"unknown sparsity_mode {sparsity_mode!r}")

        self.timesteps = int(timesteps)
        self._threshold = float(threshold)
        self.beta = float(beta)
        self.sparsity_mode = sparsity_mode
        self.k_per_layer = k_per_layer
        self.activity_reg_lambda = float(activity_reg_lambda)
        self.activity_target = float(activity_target)
        # Set each forward pass; the trainer adds it to the loss (activity_reg mode).
        self.last_activity_penalty: torch.Tensor = torch.zeros(())
        # Section 3.3.5 activation-dropout confound control: random per-unit dropout
        # on the FC spike-count features of a dense model, matched to a target active
        # fraction, to isolate "fewer units active" from sparse coding.
        self.dropout_p = float(dropout_p)

        spike_grad = surrogate.fast_sigmoid()

        def _lif() -> snn.Leaky:
            return snn.Leaky(
                beta=self.beta,
                threshold=self._threshold,
                spike_grad=spike_grad,
                reset_mechanism="zero",
            )

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.lifc1 = _lif()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.lifc2 = _lif()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lifc3 = _lif()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(_CONV_FEATURE_DIM, _HIDDEN)
        self.lif1 = _lif()
        self.fc2 = nn.Linear(_HIDDEN, _HIDDEN)
        self.lif2 = _lif()
        self.heads = build_heads(_HIDDEN, n_tasks, n_classes_per_task)

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, value: float) -> None:
        """Set the global firing threshold on every LIF layer (conv + FC)."""
        self._threshold = float(value)
        with torch.no_grad():
            for lif in (self.lifc1, self.lifc2, self.lifc3, self.lif1, self.lif2):
                lif.threshold.fill_(float(value))

    def _conv_step(self, x_t, memc1, memc2, memc3):
        """One timestep through the spiking conv frontend -> [B, 1024] features."""
        spk, memc1 = self.lifc1(self.conv1(x_t), memc1)
        z = self.pool(spk)
        spk, memc2 = self.lifc2(self.conv2(z), memc2)
        z = self.pool(spk)
        spk, memc3 = self.lifc3(self.conv3(z), memc3)
        z = self.pool(spk)
        return z.flatten(1), memc1, memc2, memc3

    @staticmethod
    def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        width = scores.shape[1]
        k = max(1, min(int(k), width))
        mask = torch.zeros_like(scores)
        idx = scores.topk(k, dim=1).indices
        mask.scatter_(1, idx, 1.0)
        return mask

    @torch.no_grad()
    def _winner_masks(self, x: torch.Tensor):
        """Pick fixed per-sample FC winner sets from summed membrane over the window."""
        assert self.k_per_layer is not None
        k1, k2 = self.k_per_layer
        memc1 = self.lifc1.init_leaky()
        memc2 = self.lifc2.init_leaky()
        memc3 = self.lifc3.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        score1 = x.new_zeros((x.shape[0], _HIDDEN))
        score2 = x.new_zeros((x.shape[0], _HIDDEN))
        for _ in range(self.timesteps):
            features, memc1, memc2, memc3 = self._conv_step(x, memc1, memc2, memc3)
            spk1, mem1 = self.lif1(self.fc1(features), mem1)
            score1 = score1 + mem1
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            score2 = score2 + mem2
        return self._topk_mask(score1, k1), self._topk_mask(score2, k2)

    def forward(self, x: torch.Tensor, task_id: int, return_traces: bool = False):
        if x.dim() != 4:
            raise ValueError(f"expected [batch, 3, 32, 32] input, got {tuple(x.shape)}")

        h1_mask = h2_mask = None
        if self.sparsity_mode == "kwta_window":
            h1_mask, h2_mask = self._winner_masks(x)

        memc1 = self.lifc1.init_leaky()
        memc2 = self.lifc2.init_leaky()
        memc3 = self.lifc3.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        h1_sum = x.new_zeros((x.shape[0], _HIDDEN))
        h2_sum = x.new_zeros((x.shape[0], _HIDDEN))

        for _ in range(self.timesteps):
            features, memc1, memc2, memc3 = self._conv_step(x, memc1, memc2, memc3)
            spk1, mem1 = self.lif1(self.fc1(features), mem1)
            if h1_mask is not None:
                spk1 = spk1 * h1_mask
                mem1 = mem1 * h1_mask
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            if h2_mask is not None:
                spk2 = spk2 * h2_mask
                mem2 = mem2 * h2_mask
            h1_sum = h1_sum + spk1
            h2_sum = h2_sum + spk2

        if self.dropout_p > 0.0:
            h1_sum = torch.nn.functional.dropout(h1_sum, p=self.dropout_p, training=self.training)
            h2_sum = torch.nn.functional.dropout(h2_sum, p=self.dropout_p, training=self.training)

        logits = self.heads[task_id](h2_sum)

        if self.sparsity_mode == "activity_reg":
            mean_rate = torch.cat([h1_sum, h2_sum], dim=1).mean() / self.timesteps
            self.last_activity_penalty = self.activity_reg_lambda * (mean_rate - self.activity_target) ** 2
        else:
            self.last_activity_penalty = torch.zeros((), device=x.device)

        traces = ForwardTraces(h1_spike_counts=h1_sum, h2_spike_counts=h2_sum) if return_traces else None
        return logits, traces


def build_conv_model(cfg: dict, condition: float) -> ConvSnn:
    """Construct a ConvSnn from a pilot config dict and a swept condition value."""
    mode = cfg.get("sparsity_mode", "threshold")
    activity_reg_lambda = 0.0
    activity_target = 0.0
    if mode == "kwta_window":
        k_per_layer = [max(1, round(condition * _HIDDEN)) for _ in range(2)]
        threshold = float(cfg.get("threshold", 1.0))
    elif mode == "activity_reg":
        k_per_layer = None
        threshold = float(cfg.get("threshold", 1.0))
        activity_reg_lambda = float(cfg.get("activity_reg_lambda", 1.0))
        activity_target = condition
    else:
        k_per_layer = None
        threshold = condition
    return ConvSnn(
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
