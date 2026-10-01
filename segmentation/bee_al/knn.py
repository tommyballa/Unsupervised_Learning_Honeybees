"""Classificatore kNN pesato sulla memory bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .memory_bank import MemoryBank


@dataclass
class Prediction:
    """Distribuzione sulle classi per un insieme di api (righe allineate a ``indices``)."""

    classes: list[str]
    probs: np.ndarray  # (n, n_classi); n_classi può essere 0 o 1

    @property
    def confidence(self) -> np.ndarray:
        if self.probs.shape[1] == 0:
            return np.zeros(len(self.probs))
        return self.probs.max(axis=1)

    @property
    def labels(self) -> list[Optional[str]]:
        if self.probs.shape[1] == 0:
            return [None] * len(self.probs)
        return [self.classes[j] for j in self.probs.argmax(axis=1)]


def knn_predict(
    bank: MemoryBank,
    indices: Optional[Sequence[int]] = None,
    k: int = 10,
    temperature: float = 0.07,
) -> Prediction:
    """Distribuzione sulle classi dai k vicini etichettati, pesati con ``exp(sim / T)``.

    Funziona con qualsiasi numero di classi: con 0 etichette le probabilità hanno 0 colonne,
    con una sola classe valgono tutte 1. Un'ape etichettata non vota per se stessa.
    """
    idx = np.arange(len(bank)) if indices is None else np.asarray(indices, dtype=np.int64)
    classes = bank.classes
    probs = np.zeros((len(idx), len(classes)), dtype=np.float64)
    if not classes or len(idx) == 0:
        return Prediction(classes, probs)

    # Un vicino in più, per poter scartare l'ape stessa quando è etichettata.
    sims, nb = bank.search_labeled(bank.X[idx], k + 1)
    is_self = nb == idx[:, None]
    sims = np.where(is_self, -np.inf, sims.astype(np.float64))
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    sims = np.take_along_axis(sims, order, axis=1)
    nb = np.take_along_axis(nb, order, axis=1)

    valid = np.isfinite(sims)
    best = np.where(valid, sims, -np.inf).max(axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    weights = np.where(valid, np.exp((np.where(valid, sims, 0.0) - best) / temperature), 0.0)

    class_pos = {c: j for j, c in enumerate(classes)}
    nb_class = np.array([class_pos[lbl] for lbl in bank.labels_at(nb.ravel())]).reshape(nb.shape)
    rows = np.repeat(np.arange(len(idx)), nb.shape[1])
    np.add.at(probs, (rows, nb_class.ravel()), weights.ravel())
    total = probs.sum(axis=1, keepdims=True)
    return Prediction(classes, np.divide(probs, total, out=np.zeros_like(probs), where=total > 0))


def nearest_labeled_similarity(bank: MemoryBank, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Similarità coseno tra ogni ape e l'esempio etichettato più vicino (escluso se stessa).

    Vale 0 se non ci sono (altre) api etichettate. ``1 - similarità`` è la "novità" dell'ape:
    alta per le api che non assomigliano a nulla di ciò che è stato etichettato finora.
    """
    idx = np.arange(len(bank)) if indices is None else np.asarray(indices, dtype=np.int64)
    sims, nb = bank.search_labeled(bank.X[idx], 2)
    if sims.shape[1] == 0:
        return np.zeros(len(idx))
    sims = np.where(nb == idx[:, None], -np.inf, sims.astype(np.float64)).max(axis=1)
    return np.where(np.isfinite(sims), sims, 0.0)
