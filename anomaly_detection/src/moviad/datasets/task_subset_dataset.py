"""
TaskSubsetDataset — Wrapper that selects a subset of a dataset
and applies an additional post-loading transform.

Usage:
    from moviad.datasets.task_subset_dataset import TaskSubsetDataset

    # Takes the first 1000 samples of the dataset and applies a transform
    subset_ds = TaskSubsetDataset(
        dataset=original_dataset,
        indices=range(0, 1000),
        extra_transform=DirtyLensTransform(),
    )
"""

import torch
from torch.utils.data import Dataset
from typing import Callable, Sequence


class TaskSubsetDataset(Dataset):
    """Dataset wrapper that selects a subset and applies extra transforms.

    Works both with datasets that return only images (training)
    and with those that return tuples (image, label, mask, path) for test/val.

    Args:
        dataset: The original dataset to extract the subset from.
        indices: List/sequence of indices to select.
        extra_transform: Additional transform applied to the image tensor
                         (operates on the tensor already transformed by the original dataset).
                         If None, no additional transform is applied.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        extra_transform: Callable | None = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.extra_transform = extra_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        item = self.dataset[self.indices[index]]

        if self.extra_transform is None:
            return item

        # The dataset may return:
        # - only the image (Tensor) during training
        # - a tuple (image, label, mask, path) during test/val
        if isinstance(item, torch.Tensor):
            return self.extra_transform(item)
        elif isinstance(item, (tuple, list)):
            # The first element is always the image
            items = list(item)
            items[0] = self.extra_transform(items[0])
            return tuple(items)
        else:
            return item

    def __repr__(self) -> str:
        return (
            f"TaskSubsetDataset(n_samples={len(self)}, "
            f"extra_transform={self.extra_transform})"
        )
