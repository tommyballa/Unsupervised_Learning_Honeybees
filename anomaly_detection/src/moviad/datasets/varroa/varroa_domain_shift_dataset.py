"""
VarroaDomainShiftDataset — Dataset per CL con domain shift progressivo.

Estende VarroaDataset per creare automaticamente 3 task con condizioni
di acquisizione diverse, a partire dallo stesso dataset aggregato:

    - varroa_clean:      Immagini originali (nessuna trasformazione)
    - varroa_dirty_lens: Blur + macchie (simula lente sporca)
    - varroa_low_light:  Bassa luminosità + rumore (simula cambio luce)

Il dataset viene diviso internamente in 3 parti uguali usando un seed
fisso per garantire la riproducibilità. Ogni category corrisponde a
una porzione diversa del dataset con la propria trasformazione.

Utilizzo con ContinualDataset:
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


# Seed fisso per la divisione del dataset — garantisce che le stesse
# immagini finiscano sempre nello stesso task indipendentemente dallo split.
_PARTITION_SEED = 42

# Numero di parti in cui dividere il dataset
_N_PARTITIONS = 3

# Mapping category → (indice partizione, trasformazione da applicare)
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
    """Dataset Varroa con domain shift automatico.

    Quando viene istanziato con una delle category di domain shift
    (varroa_clean, varroa_dirty_lens, varroa_low_light), carica l'intero
    dataset aggregato e poi seleziona automaticamente solo il sottoinsieme
    corrispondente alla propria partizione, applicando la trasformazione
    corretta.

    Args:
        dataset_arguments: Argomenti del dataset (path, size, transforms).
        category: Una tra "varroa_clean", "varroa_dirty_lens", "varroa_low_light".
        split: Lo split da caricare (TRAIN, VAL, TEST).
    """

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ) -> None:
        if category not in _TASK_DEFINITIONS:
            raise ValueError(
                f"Category '{category}' non supportata. "
                f"Usa una tra: {list(_TASK_DEFINITIONS.keys())}"
            )

        self._domain_shift_category = category
        task_def = _TASK_DEFINITIONS[category]
        self._extra_transform = task_def["transform"]
        self._partition_index = task_def["partition_index"]

        # Carica TUTTO il dataset aggregato ("varroa") tramite la classe base
        super().__init__(dataset_arguments, category="varroa", split=split)

        # Dopo il caricamento, seleziona solo la partizione corretta
        self._select_partition()

    def _select_partition(self):
        """Divide il DataFrame dei sample in N parti uguali e seleziona
        solo quella corrispondente al proprio partition_index."""
        n_total = len(self.samples)

        # Genera gli indici mescolati con seed fisso
        indices = list(range(n_total))
        rng = _random.Random(_PARTITION_SEED)
        rng.shuffle(indices)

        # Calcola i confini della partizione
        chunk_size = n_total // _N_PARTITIONS
        start = self._partition_index * chunk_size
        # L'ultimo task prende anche il resto
        if self._partition_index < _N_PARTITIONS - 1:
            end = start + chunk_size
        else:
            end = n_total

        selected_indices = indices[start:end]

        # Filtra il DataFrame mantenendo solo i sample della partizione
        self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)

        print(
            f"  [{self._domain_shift_category}] {self.split.value}: "
            f"{len(self.samples)}/{n_total} immagini selezionate, "
            f"transform={self._extra_transform}"
        )

    def __getitem__(self, index: int):
        """Restituisce l'item applicando la trasformazione di domain shift."""
        item = super().__getitem__(index)

        if self._extra_transform is None:
            return item

        # Il dataset può restituire:
        # - solo l'immagine (Tensor) durante il training
        # - una tupla (image, label, mask, path) durante test/val
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
        """Restituisce le 3 category di domain shift."""
        return list(_TASK_DEFINITIONS.keys())

    def __repr__(self) -> str:
        return (
            f"VarroaDomainShiftDataset(category={self._domain_shift_category}, "
            f"split={self.split}, n_samples={len(self)}, "
            f"transform={self._extra_transform})"
        )
