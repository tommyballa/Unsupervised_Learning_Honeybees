"""Weighted kNN classifier on the memory bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .memory_bank import MemoryBank


@dataclass
class Prediction:
    """Class distribution for a set of bees (rows aligned with ``indices``)."""

    classes: list[str]
    probs: np.ndarray  # (n, n_classes); n_classes can be 0 or 1

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
    """Class distribution from the k labelled neighbours, weighted by ``exp(sim / T)``.

    Works with any number of classes: with 0 labels the probabilities have 0 columns,
    with a single class they are all 1. A labelled bee does not vote for itself.
    """
    idx = np.arange(len(bank)) if indices is None else np.asarray(indices, dtype=np.int64)
    classes = bank.classes
    probs = np.zeros((len(idx), len(classes)), dtype=np.float64)
    if not classes or len(idx) == 0:
        return Prediction(classes, probs)

    # One extra neighbour, so the bee itself can be dropped when it is labelled.
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
    """Cosine similarity between each bee and the nearest labelled example (excluding itself).

    It is 0 if there are no (other) labelled bees. ``1 - similarity`` is the bee's "novelty":
    high for bees that look like nothing labelled so far.
    """
    idx = np.arange(len(bank)) if indices is None else np.asarray(indices, dtype=np.int64)
    sims, nb = bank.search_labeled(bank.X[idx], 2)
    if sims.shape[1] == 0:
        return np.zeros(len(idx))
    sims = np.where(nb == idx[:, None], -np.inf, sims.astype(np.float64)).max(axis=1)
    return np.where(np.isfinite(sims), sims, 0.0)
