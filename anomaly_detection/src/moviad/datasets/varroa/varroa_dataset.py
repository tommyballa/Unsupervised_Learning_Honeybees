"""
VarroaDataset — Dataset per il rilevamento di varroa su api.

Struttura del dataset su disco:
    data/
    ├── train/
    │   ├── gt_one.csv
    │   ├── videos/<video_folder>/<image>.png
    │   └── labels/<video_folder>/<image>.txt
    ├── test/   (stessa struttura)
    └── val/    (stessa struttura)

Formato gt_one.csv (space-separated, senza header):
    <path_relativo> <label> [x1 y1 x2 y2 ...]
    - label: 0=normale, 1=varroa singola, 3=varroa multipla
    - bbox opzionali in gruppi di 4 (possono esserci più bbox per riga)

Due modalità di utilizzo:
    1. Singola task:  category="varroa"  → aggrega tutti i video folder
    2. Per-video CL:  category="<nome_video_folder>" → un solo folder
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split, LabelName


# Video folder presenti nel train set (ordine cronologico → ordine delle task CL)
TRAIN_VIDEO_FOLDERS = (
    "2017-08-28_09-30-00-1_500_dirty_glass",
    "2017-08-28_16-31-55",
    "2017-08-30_15-42-59",
    "2017-09-20_19-24-55",
    "2017-09-25_16-03-38-2",
    "2017-10-17_16-41-10",
)

# Categoria speciale che aggrega tutti i video folder
_AGGREGATE_CATEGORY = "varroa"


class VarroaDataset(VADDataset):
    """Dataset per il rilevamento di varroa su api.

    Supporta sia il caricamento di un singolo video folder (per CL)
    sia l'aggregazione di tutti i folder (per training single-task).
    """

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ) -> None:
        super().__init__(dataset_arguments, category, split)

        self.samples: pd.DataFrame | None = None
        self._cache: dict = {}
        self._load_dataset()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _split_dir(self) -> Path:
        """Restituisce il path della directory dello split (train/test/val)."""
        split_str = self.split.value if hasattr(self.split, "value") else self.split
        # Mappa Split.VAL → "val" (la directory su disco si chiama "val")
        if split_str == "valid":
            split_str = "val"
        return Path(self.dataset_arguments.dataset_path) / split_str

    def _load_dataset(self):
        """Legge gt_one.csv e costruisce il DataFrame dei sample."""
        split_dir = self._split_dir()
        csv_path = split_dir / "gt_one.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Ground truth CSV non trovato: {csv_path}"
            )

        rows: list[dict] = []
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]          # es. "videos/<folder>/<img>.png"
                label_int = int(parts[1])    # 0, 1, o 3

                # Bounding box: gruppi di 4 float
                bbox_coords = [float(x) for x in parts[2:]]
                bboxes = []
                for i in range(0, len(bbox_coords), 4):
                    if i + 3 < len(bbox_coords):
                        bboxes.append(bbox_coords[i : i + 4])

                # Estrai il nome del video folder dal path relativo
                # rel_path = "videos/<video_folder>/<filename>.png"
                path_parts = Path(rel_path).parts
                video_folder = path_parts[1] if len(path_parts) >= 3 else ""

                # Filtra per category
                if self.category != _AGGREGATE_CATEGORY:
                    if video_folder != self.category:
                        continue

                abs_path = str(split_dir / rel_path)
                label_index = LabelName.NORMAL if label_int == 0 else LabelName.ABNORMAL

                rows.append(
                    {
                        "image_path": abs_path,
                        "label_index": label_index.value,
                        "label_raw": label_int,
                        "video_folder": video_folder,
                        "bboxes": bboxes,
                    }
                )

        if not rows:
            raise RuntimeError(
                f"Nessun sample trovato per category='{self.category}' "
                f"in {csv_path}"
            )

        self.samples = pd.DataFrame(rows)

        # Durante il training (unsupervised one-class), si usano SOLO sample normali.
        # Il test e la val mantengono tutti i sample per la valutazione.
        if self.split == Split.TRAIN:
            n_total = len(self.samples)
            self.samples = self.samples[
                self.samples["label_index"] == LabelName.NORMAL.value
            ].reset_index(drop=True)
            n_normal = len(self.samples)
            print(
                f"[VarroaDataset] TRAIN split: {n_normal}/{n_total} sample normali "
                f"(scartati {n_total - n_normal} anomali)"
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
                mask (Tensor): (1, H, W) — mask rettangolare dai bbox
                path (str): path assoluto dell'immagine
        """
        if index in self._cache:
            return self._cache[index]

        row = self.samples.iloc[index]

        image = Image.open(row.image_path).convert("RGB")
        image = self.transform_image(image)

        if self.split == Split.TRAIN:
            self._cache[index] = image
            return image

        label = row.label_index
        path = row.image_path
        bboxes = row.bboxes

        if label == LabelName.ABNORMAL.value and bboxes:
            mask = self._bboxes_to_mask(bboxes)
        else:
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size)

        result = (image, label, mask.int(), path)
        self._cache[index] = result
        return result

    def _bboxes_to_mask(self, bboxes: list[list[float]]) -> torch.Tensor:
        """Genera una mask binaria dai bounding box.

        I bbox nel CSV sono in coordinate pixel dell'immagine originale (160×280).
        La mask viene generata alla dimensione gt_mask_size.
        """
        # Dimensione originale delle immagini varroa
        orig_w, orig_h = 160, 280

        mask_h, mask_w = self.dataset_arguments.gt_mask_size
        mask = torch.zeros(1, mask_h, mask_w)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Scala le coordinate alla dimensione della mask
            sx1 = int(x1 / orig_w * mask_w)
            sy1 = int(y1 / orig_h * mask_h)
            sx2 = int(x2 / orig_w * mask_w)
            sy2 = int(y2 / orig_h * mask_h)

            # Clamp
            sx1 = max(0, min(sx1, mask_w - 1))
            sy1 = max(0, min(sy1, mask_h - 1))
            sx2 = max(0, min(sx2, mask_w))
            sy2 = max(0, min(sy2, mask_h))

            mask[0, sy1:sy2, sx1:sx2] = 1

        return mask



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
    # Categories per il continual learning
    # ------------------------------------------------------------------

    @staticmethod
    def get_categories() -> list:
        """Restituisce i video folder del train come task per il CL."""
        return list(TRAIN_VIDEO_FOLDERS)
