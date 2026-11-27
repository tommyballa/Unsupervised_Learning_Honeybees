"""
VarroaDomainShiftDataset — Dataset for CL with progressive domain shift.

Extends VarroaDataset to automatically create 3 tasks with different
acquisition conditions, starting from the same dataset:

    - varroa_clean:      Original images (no transform)
    - varroa_dirty_lens: Blur + spots (simulates a dirty lens)
    - varroa_low_light:  Low brightness + noise (simulates a lighting change)

The dataset is split internally into 3 equal parts using a fixed
seed to ensure reproducibility. Each category corresponds to a
different portion of the dataset with its own transform.

Usage with ContinualDataset:
    from moviad.datasets.varroa.varroa_domain_shift_dataset import VarroaDomainShiftDataset
    from moviad.scenarios.continual.continual_dataset import ContinualDataset

    continual_dataset = ContinualDataset(
        dataset_args,
        VarroaDomainShiftDataset,
        categories=["varroa_clean", "varroa_dirty_lens", "varroa_low_light"],
    )
"""

import random as _random

import torch

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.domain_shift_transforms import DirtyLensTransform, LowLightTransform
from moviad.utilities.configurations import Split


# Fixed seed for splitting the dataset — ensures the same images always
# end up in the same task regardless of the split.
_PARTITION_SEED = 42

# Number of parts to split the dataset into
_N_PARTITIONS = 3

# Mapping category → (partition index, transform to apply)
_TASK_DEFINITIONS = {
    "varroa_clean": {
        "partition_index": 0,
        "transform": None,
    },
    "varroa_dirty_lens": {
        "partition_index": 1,
        "transform": DirtyLensTransform(
            blur_kernel=13, blur_sigma=4.0, border_ratio=0.25,
            num_spots=15, spot_radius_range=(20, 50), spot_opacity=0.6,
        ),
    },
    "varroa_low_light": {
        "partition_index": 2,
        "transform": LowLightTransform(
            brightness_factor=0.3, contrast_factor=0.5, noise_std=0.15,
        ),
    },
}


class VarroaDomainShiftDataset(VarroaDataset):
    """Varroa dataset with automatic domain shift.

    When instantiated with one of the domain-shift categories
    (varroa_clean, varroa_dirty_lens, varroa_low_light), it loads the whole
    dataset and then automatically selects only the subset
    corresponding to its own partition, applying the right
    transform.

    Args:
        dataset_arguments: Dataset arguments (path, size, transforms).
        category: One of "varroa_clean", "varroa_dirty_lens", "varroa_low_light".
        split: The split to load (TRAIN, VAL, TEST).
    """

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ) -> None:
        if category not in _TASK_DEFINITIONS:
            raise ValueError(
                f"Category '{category}' not supported. "
                f"Use one of: {list(_TASK_DEFINITIONS.keys())}"
            )

        self._domain_shift_category = category
        task_def = _TASK_DEFINITIONS[category]
        self._extra_transform = task_def["transform"]
        self._partition_index = task_def["partition_index"]

        # Load the WHOLE aggregated dataset ("varroa") through the base class
        super().__init__(dataset_arguments, category="varroa", split=split)

        # After loading, select only the right partition
        self._select_partition()

    def _select_partition(self):
        """Splits the sample DataFrame into N equal parts and selects
        only the one matching its own partition_index."""
        n_total = len(self.samples)

        # Generate the indices shuffled with a fixed seed
        indices = list(range(n_total))
        rng = _random.Random(_PARTITION_SEED)
        rng.shuffle(indices)

        # Compute the partition bounds
        chunk_size = n_total // _N_PARTITIONS
        start = self._partition_index * chunk_size
        # The last task also takes the remainder
        if self._partition_index < _N_PARTITIONS - 1:
            end = start + chunk_size
        else:
            end = n_total

        selected_indices = indices[start:end]

        # Filter the DataFrame keeping only the partition's samples
        self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)

        print(
            f"  [{self._domain_shift_category}] {self.split.value}: "
            f"{len(self.samples)}/{n_total} images selected, "
            f"transform={self._extra_transform}"
        )

    def __getitem__(self, index: int):
        """Returns the item with the domain-shift transform applied."""
        item = super().__getitem__(index)

        if self._extra_transform is None:
            return item

        # The dataset may return:
        # - only the image (Tensor) during training
        # - a tuple (image, label, mask, path) during test/val
        if isinstance(item, torch.Tensor):
            return self._extra_transform(item)
        elif isinstance(item, (tuple, list)):
            items = list(item)
            items[0] = self._extra_transform(items[0])
            return tuple(items)
        else:
            return item

    @staticmethod
    def get_categories() -> list:
        """Returns the 3 domain-shift categories."""
        return list(_TASK_DEFINITIONS.keys())

    def __repr__(self) -> str:
        return (
            f"VarroaDomainShiftDataset(category={self._domain_shift_category}, "
            f"split={self.split}, n_samples={len(self)}, "
            f"transform={self._extra_transform})"
        )
