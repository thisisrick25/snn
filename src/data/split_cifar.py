"""Split-CIFAR-10 continual-learning benchmark.

Mirrors ``split_mnist`` but for CIFAR-10: ten classes are grouped into five
binary tasks, each remapped to labels {0, 1}. Images are flattened to length
3072 (3x32x32) so the same flatten-MLP LIF model can consume them.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .transforms import cifar10_transform, cifar10_conv_transform

DEFAULT_TASKS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def _remap_subset(dataset, pair: tuple[int, int]) -> TensorDataset:
    """Extract the two classes in ``pair`` and remap labels to {0, 1}.

    ``datasets.CIFAR10.targets`` is a Python list, so it is converted to a
    tensor before masking (unlike MNIST, whose targets are already a tensor).
    """
    lo, hi = pair
    targets = torch.as_tensor(dataset.targets)
    mask = (targets == lo) | (targets == hi)
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    xs = torch.stack([dataset[i][0] for i in idx])
    ys = (targets[mask] == hi).long()
    return TensorDataset(xs, ys)


def build_split_cifar(
    root: str,
    tasks: list[tuple[int, int]] | None = None,
    batch_size: int = 128,
    download: bool = True,
    conv: bool = False,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """Build per-task train/test loaders for Split-CIFAR-10.

    ``conv=False`` flattens images to [3072] for the MLP-LIF; ``conv=True`` keeps
    them as [3,32,32] for the spiking convolutional frontend.
    """
    if tasks is None:
        tasks = DEFAULT_TASKS

    tfm = cifar10_conv_transform() if conv else cifar10_transform()
    train_full = datasets.CIFAR10(root=root, train=True, download=download, transform=tfm)
    test_full = datasets.CIFAR10(root=root, train=False, download=download, transform=tfm)

    train_loaders: list[DataLoader] = []
    test_loaders: list[DataLoader] = []
    for pair in tasks:
        train_loaders.append(
            DataLoader(_remap_subset(train_full, pair), batch_size=batch_size, shuffle=True)
        )
        test_loaders.append(
            DataLoader(_remap_subset(test_full, pair), batch_size=batch_size, shuffle=False)
        )
    return train_loaders, test_loaders
