"""
VarroaDataset — Dataset for varroa detection on bees (Ceschi).

Dataset layout on disk:

    Training:
        .../train_single_bee/train_single_bee_clustered/categoria_1/
            ├── bee_00000_conf0.96.png
            ├── bee_00011_conf0.93.png
            └── ...
        All images are normal (no CSV needed).

    Test:
        .../test_single_bee/anomaly_dataset_synthetic/
            ├── labels.csv          (bee_id, label, image_path, mask_path)
            ├── normal/             healthy images
            ├── anomalous/          images with varroa
            └── masks/              pixel-level masks for the anomalous images

    labels.csv:
        bee_id,label,image_path,mask_path
        - label: 0=normal, 1=anomalous
        - mask_path: empty for the normal images

Two usage modes:
    1. Single task:   category="varroa"  → uses the hardcoded paths
    2. Per-video CL:  no longer supported (single dataset)
"""

from moviad.datasets.varroa.paths import DATA_ROOT
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split, LabelName


# Default paths — overridden by dataset_arguments.dataset_path when provided
_DEFAULT_TRAIN_DIR = Path(
    DATA_ROOT + "/processed/train_single_bee_batch3+4/cleaned_bees"
)
_DEFAULT_TEST_DIR = Path(
    DATA_ROOT + "/processed/test_single_bee_batch3+4_varroa/images"
)

# Reduced test set: all anomalous images + a fixed subset of normal ones
# (512x512 maps for the whole test set do not fit in reasonable time/RAM).
# None = use all normal images; DatasetArguments.full_test=True disables it for a script.
_MAX_TEST_NORMALS = 5000
_TEST_SUBSAMPLE_SEED = 0

# Special category that aggregates all the data
_AGGREGATE_CATEGORY = "varroa"


class VarroaDataset(VADDataset):
    """Dataset for varroa detection on bees (Ceschi).

    - TRAIN: loads all images from categoria_1 (all normal).
    - TEST/VAL: loads from anomaly_dataset_synthetic with labels.csv,
      using the pixel-level masks for the anomalous images.
    """

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ) -> None:
        super().__init__(dataset_arguments, category, split)

        self.samples: pd.DataFrame | None = None
        # Pre-converted lists for fast indexing (avoid pd.iloc overhead)
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.mask_paths: list[str] = []
        self._load_dataset()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_dataset(self):
        """Loads the samples according to the split."""
        if self.split == Split.TRAIN:
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        """Loads all images from categoria_1 as normal samples."""
        if self.dataset_arguments.dataset_path:
            train_dir = Path(self.dataset_arguments.dataset_path)
        else:
            train_dir = _DEFAULT_TRAIN_DIR

        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}"
            )

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        image_files = sorted([
            f for f in train_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise RuntimeError(
                f"No images found in {train_dir}"
            )

        rows = []
        for img_path in image_files:
            rows.append({
                "image_path": str(img_path),
                "label_index": LabelName.NORMAL.value,
                "mask_path": "",
            })

        self.samples = pd.DataFrame(rows)
        # Pre-convert to lists for fast __getitem__ indexing
        self.image_paths = self.samples["image_path"].tolist()
        self.labels = self.samples["label_index"].tolist()
        self.mask_paths = self.samples["mask_path"].tolist()
        print(
            f"[VarroaDataset] TRAIN: {len(self.samples)} normal images "
            f"from {train_dir.name}"
        )

    def _load_test(self):
        """Loads the samples from anomaly_dataset_synthetic using labels.csv."""
        test_path = getattr(self.dataset_arguments, "test_path", None)
        test_dir = Path(test_path) if test_path else _DEFAULT_TEST_DIR
        csv_path = test_dir / "labels.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        # Deterministic subset of the normal images (the same for all models)
        normal_idx = df.index[df["label"] == 0]
        full_test = getattr(self.dataset_arguments, "full_test", False)
        if not full_test and _MAX_TEST_NORMALS is not None and len(normal_idx) > _MAX_TEST_NORMALS:
            rng = np.random.default_rng(_TEST_SUBSAMPLE_SEED)
            dropped = rng.choice(
                normal_idx, size=len(normal_idx) - _MAX_TEST_NORMALS, replace=False
            )
            df = df.drop(index=dropped)

        rows = []
        for _, row in df.iterrows():
            label_int = int(row["label"])
            label_index = (
                LabelName.NORMAL if label_int == 0
                else LabelName.ABNORMAL
            )

            # The paths in the CSV may point to the old directory;
            # rebuild the correct paths from the actual directory.
            bee_id = row["bee_id"]
            if label_int == 0:
                img_path = str(test_dir / "normal" / f"{bee_id}.png")
                mask_path = ""
            else:
                img_path = str(test_dir / "anomalous" / f"{bee_id}.png")
                mask_path = str(test_dir / "masks" / f"{bee_id}_mask.png")

            rows.append({
                "image_path": img_path,
                "label_index": label_index.value,
                "mask_path": mask_path,
            })

        if not rows:
            raise RuntimeError(
                f"No samples found in {csv_path}"
            )

        self.samples = pd.DataFrame(rows)
        # Pre-convert to lists for fast __getitem__ indexing
        self.image_paths = self.samples["image_path"].tolist()
        self.labels = self.samples["label_index"].tolist()
        self.mask_paths = self.samples["mask_path"].tolist()

        n_normal = (self.samples.label_index == LabelName.NORMAL.value).sum()
        n_anomal = (self.samples.label_index == LabelName.ABNORMAL.value).sum()
        print(
            f"[VarroaDataset] TEST: {len(self.samples)} total samples "
            f"({n_normal} normal, {n_anomal} anomalous)"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.samples is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Returns:
            If split == TRAIN:
                image (Tensor): (C, H, W)
            Otherwise:
                image (Tensor): (C, H, W)
                label (int): 0 or 1
                mask (Tensor): (1, H, W) — pixel-level mask or zero mask
                path (str): absolute path of the image
        """
        img_path = self.image_paths[index]

        image = Image.open(img_path).convert("RGB")
        image = self.transform_image(image)

        if self.split == Split.TRAIN:
            return image

        label = self.labels[index]
        mask_path = self.mask_paths[index]

        if label == LabelName.ABNORMAL.value and mask_path:
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            # Binarize the mask (threshold 0.5)
            mask = (mask > 0.5).int()
        else:
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size)

        return (image, label, mask.int(), img_path)

    # ------------------------------------------------------------------
    # Abstract methods from VADDataset
    # ------------------------------------------------------------------

    def split_dataset(self, train_size, valid_size):
        """Unused — the splits are already defined by the layout on disk."""
        pass

    def contaminate(self, ratio: float, seed: int = 42) -> int:
        """Not implemented for VarroaDataset."""
        raise NotImplementedError("Contamination not supported for VarroaDataset")

    def compute_contamination_ratio(self) -> float:
        """Computes the fraction of anomalous samples."""
        if self.samples is None:
            return 0.0
        n_abnormal = (self.samples.label_index == LabelName.ABNORMAL.value).sum()
        return n_abnormal / len(self.samples)

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    @staticmethod
    def get_categories() -> list:
        """Returns the list of available categories."""
        return [_AGGREGATE_CATEGORY]
