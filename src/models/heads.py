"""Task-specific binary output heads for task-incremental learning.

Each Split-MNIST task gets its own linear head. At train/eval time the caller
selects the head for the current task by index (task-incremental setting: the
task label is available at inference).
"""

from __future__ import annotations

import torch.nn as nn


def build_heads(hidden_dim: int, n_tasks: int, n_classes_per_task: int) -> nn.ModuleList:
    """Create one linear head per task.

    Args:
        hidden_dim: width of the final hidden layer feeding the heads.
        n_tasks: number of continual-learning tasks.
        n_classes_per_task: output classes per task (2 for Split-MNIST binary tasks).

    Returns:
        A ``ModuleList`` of ``n_tasks`` linear layers, indexed by task id.
    """
    return nn.ModuleList(
        [nn.Linear(hidden_dim, n_classes_per_task) for _ in range(n_tasks)]
    )
