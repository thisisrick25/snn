"""Split-MNIST construction: 5 binary tasks with per-task label remapping.

Task t uses two digit classes; labels are remapped to {0, 1} so every task shares
the same 2-way head structure. Task order is fixed for reproducibility.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .transforms import mnist_transform

# Fixed Split-MNIST task pairs (EXPERIMENT_PROTOCOL.md section 2.2).
DEFAULT_TASKS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def _remap_subset(dataset: datasets.MNIST, pair: tuple[int, int]) -> TensorDataset:
    """Extract the two classes of `pair` and remap their labels to {0, 1}.

    Materializes to a TensorDataset so the flattened inputs and remapped labels
    are fixed and cheap to iterate. Applies the dataset transform manually to the
    raw uint8 image tensors so we avoid per-item Subset indexing.
    """
    lo, hi = pair
    targets: torch.Tensor = dataset.targets
    mask = (targets == lo) | (targets == hi)
    idx = mask.nonzero(as_tuple=True)[0].tolist()

    xs = torch.stack([dataset[i][0] for i in idx])  # transform -> [N, 784]
    ys_raw = targets[mask]
    ys = (ys_raw == hi).long()  # lo -> 0, hi -> 1
    return TensorDataset(xs, ys)


def build_split_mnist(
    root: str,
    tasks: list[tuple[int, int]] | None = None,
    batch_size: int = 128,
    download: bool = True,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """Return (train_loaders, test_loaders), one DataLoader per task.

    Train loaders shuffle; test loaders do not.
    """
    if tasks is None:
        tasks = DEFAULT_TASKS

    tfm = mnist_transform()
    train_full = datasets.MNIST(root=root, train=True, download=download, transform=tfm)
    test_full = datasets.MNIST(root=root, train=False, download=download, transform=tfm)

    train_loaders: list[DataLoader] = []
    test_loaders: list[DataLoader] = []
    for pair in tasks:
        train_ds = _remap_subset(train_full, pair)
        test_ds = _remap_subset(test_full, pair)
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders
