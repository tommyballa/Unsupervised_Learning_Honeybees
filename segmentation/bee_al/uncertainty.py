"""Misure di incertezza intercambiabili. Valore più alto = ape più incerta, sempre in [0, 1]."""
from __future__ import annotations

from typing import Callable

import numpy as np


def _top2(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if probs.shape[1] == 0:
        zeros = np.zeros(len(probs))
        return zeros, zeros
    if probs.shape[1] == 1:
        return probs[:, 0], np.zeros(len(probs))
    part = np.partition(probs, -2, axis=1)
    return part[:, -1], part[:, -2]


def margin(probs: np.ndarray) -> np.ndarray:
    """1 - (p1 - p2): massima quando le prime due classi sono alla pari."""
    p1, p2 = _top2(probs)
    return 1.0 - (p1 - p2)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Entropia normalizzata per ``log(n_classi)``."""
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return 1.0 - _top2(probs)[0]
    p = np.clip(probs, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=1) / np.log(n_classes)
    # Una riga tutta a zero (nessun vicino etichettato) è la più incerta possibile.
    return np.where(probs.sum(axis=1) > 0, h, 1.0)


def least_confidence(probs: np.ndarray) -> np.ndarray:
    """1 - p1."""
    return 1.0 - _top2(probs)[0]


UNCERTAINTY_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "margin": margin,
    "entropy": entropy,
    "least_confidence": least_confidence,
}


def uncertainty_scores(probs: np.ndarray, kind: str = "margin") -> np.ndarray:
    """Incertezza per riga di ``probs`` (n, n_classi), con la misura scelta."""
    try:
        fn = UNCERTAINTY_FUNCTIONS[kind]
    except KeyError:
        raise ValueError(f"Incertezza sconosciuta: {kind}") from None
    return fn(np.asarray(probs, dtype=np.float64))
