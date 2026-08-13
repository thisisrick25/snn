"""The five pilot figures (EXPERIMENT_PROTOCOL.md Section 6.1).

All x-axes use OBSERVED activity, never the calibration target. Each plot is a
pure function of the loaded ConditionRecords, so plots can be regenerated
without retraining via scripts/make_plots.py.

Figures:
  (a) accuracy matrix heatmap, one per condition
  (b) observed activity vs final average accuracy
  (c) observed activity vs mean forgetting
  (d) observed activity vs representational overlap (CKA)
  (e) representational overlap (CKA) vs mean forgetting
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from .io import ConditionRecord  # noqa: E402


def _finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    fx, fy = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if isinstance(x, float) and math.isnan(x):
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        fx.append(x)
        fy.append(y)
    return fx, fy


def plot_accuracy_matrices(records: list[ConditionRecord], plots_dir: str) -> list[str]:
    """(a) One heatmap of the lower-triangular accuracy matrix per condition."""
    paths = []
    for r in records:
        mat = r.accuracy_matrix
        n = len(mat)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        # NaN (upper triangle) renders as blank via masked array behaviour
        data = [[(v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float("nan")) for v in row] for row in mat]
        im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
        ax.set_xlabel("evaluated task j")
        ax.set_ylabel("after training task i")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        dead = " | DEAD" if getattr(r, "dead_network", False) else ""
        ax.set_title(
            f"seed {r.seed} | obs act {r.mean_observed_activity:.3f}\n"
            f"(theta {r.threshold:.3g}){dead}"
        )
        for i in range(n):
            for j in range(n):
                v = data[i][j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
        fig.tight_layout()
        path = os.path.join(plots_dir, f"accmatrix_seed{r.seed}_theta{r.threshold:.3g}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def _scatter(records, x_attr, y_attr, xlabel, ylabel, title, fname, plots_dir):
    xs = [getattr(r, x_attr) for r in records]
    ys = [getattr(r, y_attr) for r in records]
    fx, fy = _finite_pairs(xs, ys)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(fx, fy, c="#2b6cb0", s=40, edgecolors="white", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_activity_vs_final_accuracy(records, plots_dir):
    """(b)."""
    return _scatter(records, "mean_observed_activity", "final_avg_accuracy",
                    "observed active-neuron fraction", "final average accuracy",
                    "Activity vs final average accuracy",
                    "activity_vs_final_accuracy.png", plots_dir)


def plot_activity_vs_forgetting(records, plots_dir):
    """(c)."""
    return _scatter(records, "mean_observed_activity", "mean_forgetting",
                    "observed active-neuron fraction", "mean forgetting",
                    "Activity vs mean forgetting",
                    "activity_vs_forgetting.png", plots_dir)


def plot_activity_vs_overlap(records, plots_dir):
    """(d) overlap = CKA."""
    return _scatter(records, "mean_observed_activity", "overlap_cka",
                    "observed active-neuron fraction", "representational overlap (CKA)",
                    "Activity vs representational overlap",
                    "activity_vs_overlap.png", plots_dir)


def plot_overlap_vs_forgetting(records, plots_dir):
    """(e) overlap = CKA."""
    return _scatter(records, "overlap_cka", "mean_forgetting",
                    "representational overlap (CKA)", "mean forgetting",
                    "Representational overlap vs forgetting",
                    "overlap_vs_forgetting.png", plots_dir)


def make_all_plots(records: list[ConditionRecord], plots_dir: str) -> list[str]:
    """Generate all five pilot figures. Returns list of written paths."""
    os.makedirs(plots_dir, exist_ok=True)
    paths: list[str] = []
    paths.extend(plot_accuracy_matrices(records, plots_dir))
    paths.append(plot_activity_vs_final_accuracy(records, plots_dir))
    paths.append(plot_activity_vs_forgetting(records, plots_dir))
    paths.append(plot_activity_vs_overlap(records, plots_dir))
    paths.append(plot_overlap_vs_forgetting(records, plots_dir))
    return paths
