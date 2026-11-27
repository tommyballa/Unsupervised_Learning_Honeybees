"""
TaskSubsetDataset — Wrapper che seleziona un subset di un dataset
e applica una trasformazione aggiuntiva post-caricamento.

Utilizzo:
    from moviad.datasets.task_subset_dataset import TaskSubsetDataset

    # Prende i primi 1000 sample del dataset e applica una trasformazione
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
    """Dataset wrapper che seleziona un sottoinsieme e applica trasformazioni extra.

    Compatibile sia con dataset che restituiscono solo immagini (training)
    sia con quelli che restituiscono tuple (image, label, mask, path) per test/val.

    Args:
        dataset: Il dataset originale da cui estrarre il subset.
        indices: Lista/sequenza di indici da selezionare.
        extra_transform: Trasformazione aggiuntiva da applicare al tensore immagine
                         (opera sul tensore già trasformato dal dataset originale).
                         Se None, non viene applicata nessuna trasformazione aggiuntiva.
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

        # Il dataset può restituire:
        # - solo l'immagine (Tensor) durante il training
        # - una tupla (image, label, mask, path) durante test/val
        if isinstance(item, torch.Tensor):
            return self.extra_transform(item)
        elif isinstance(item, (tuple, list)):
            # Il primo elemento è sempre l'immagine
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
