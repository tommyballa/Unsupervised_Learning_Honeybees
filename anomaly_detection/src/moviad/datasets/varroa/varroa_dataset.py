"""
VarroaDataset — Dataset per il rilevamento di varroa su api (Ceschi).

Struttura del dataset su disco:

    Training:
        .../train_single_bee/train_single_bee_clustered/categoria_1/
            ├── bee_00000_conf0.96.png
            ├── bee_00011_conf0.93.png
            └── ...
        Tutte le immagini sono normali (nessun CSV necessario).

    Test:
        .../test_single_bee/anomaly_dataset_synthetic/
            ├── labels.csv          (bee_id, label, image_path, mask_path)
            ├── normal/             immagini sane
            ├── anomalous/          immagini con varroa
            └── masks/              maschere pixel-level per le anomale

    labels.csv:
        bee_id,label,image_path,mask_path
        - label: 0=normale, 1=anomala
        - mask_path: vuoto per le normali

Due modalità di utilizzo:
    1. Singola task:  category="varroa"  → usa i path hardcoded
    2. Per-video CL:  non più supportato (dataset singolo)
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

# Test set alleggerito: tutte le anomale + un sottoinsieme fisso di normali
# (le mappe a 512x512 di tutto il test set non stanno in tempi/RAM ragionevoli).
# None = usa tutte le normali; DatasetArguments.full_test=True lo disattiva per uno script.
_MAX_TEST_NORMALS = 5000
_TEST_SUBSAMPLE_SEED = 0

# Categoria speciale che aggrega tutti i dati
_AGGREGATE_CATEGORY = "varroa"


class VarroaDataset(VADDataset):
    """Dataset per il rilevamento di varroa su api (Ceschi).

    - TRAIN: carica tutte le immagini da categoria_1 (tutte normali).
    - TEST/VAL: carica da anomaly_dataset_synthetic con labels.csv,
      usando le maschere pixel-level per le immagini anomale.
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
        """Carica i sample in base allo split."""
        if self.split == Split.TRAIN:
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        """Carica tutte le immagini da categoria_1 come sample normali."""
        if self.dataset_arguments.dataset_path:
            train_dir = Path(self.dataset_arguments.dataset_path)
        else:
            train_dir = _DEFAULT_TRAIN_DIR

        if not train_dir.exists():
            raise FileNotFoundError(
                f"Directory di training non trovata: {train_dir}"
            )

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        image_files = sorted([
            f for f in train_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise RuntimeError(
                f"Nessuna immagine trovata in {train_dir}"
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
            f"[VarroaDataset] TRAIN: {len(self.samples)} immagini normali "
            f"da {train_dir.name}"
        )

    def _load_test(self):
        """Carica i sample da anomaly_dataset_synthetic usando labels.csv."""
        test_path = getattr(self.dataset_arguments, "test_path", None)
        test_dir = Path(test_path) if test_path else _DEFAULT_TEST_DIR
        csv_path = test_dir / "labels.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Labels CSV non trovato: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        # Sottoinsieme deterministico delle normali (stesso per tutti i modelli)
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

            # I path nel CSV potrebbero puntare alla vecchia directory;
            # ricostruiamo i path corretti dalla directory effettiva.
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
                f"Nessun sample trovato in {csv_path}"
            )

        self.samples = pd.DataFrame(rows)
        # Pre-convert to lists for fast __getitem__ indexing
        self.image_paths = self.samples["image_path"].tolist()
        self.labels = self.samples["label_index"].tolist()
        self.mask_paths = self.samples["mask_path"].tolist()

        n_normal = (self.samples.label_index == LabelName.NORMAL.value).sum()
        n_anomal = (self.samples.label_index == LabelName.ABNORMAL.value).sum()
        print(
            f"[VarroaDataset] TEST: {len(self.samples)} sample totali "
            f"({n_normal} normali, {n_anomal} anomali)"
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
            Se split == TRAIN:
                image (Tensor): (C, H, W)
            Altrimenti:
                image (Tensor): (C, H, W)
                label (int): 0 o 1
                mask (Tensor): (1, H, W) — mask pixel-level o zero mask
                path (str): path assoluto dell'immagine
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
            # Binarizza la mask (threshold 0.5)
            mask = (mask > 0.5).int()
        else:
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size)

        return (image, label, mask.int(), img_path)

    # ------------------------------------------------------------------
    # Abstract methods da VADDataset
    # ------------------------------------------------------------------

    def split_dataset(self, train_size, valid_size):
        """Non utilizzato — gli split sono già definiti dalla struttura su disco."""
        pass

    def contaminate(self, ratio: float, seed: int = 42) -> int:
        """Non implementato per VarroaDataset."""
        raise NotImplementedError("Contamination non supportata per VarroaDataset")

    def compute_contamination_ratio(self) -> float:
        """Calcola la percentuale di sample anomali."""
        if self.samples is None:
            return 0.0
        n_abnormal = (self.samples.label_index == LabelName.ABNORMAL.value).sum()
        return n_abnormal / len(self.samples)

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    @staticmethod
    def get_categories() -> list:
        """Restituisce la lista delle category disponibili."""
        return [_AGGREGATE_CATEGORY]
