"""Interchangeable uncertainty measures. Higher value = more uncertain bee, always in [0, 1]."""
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
    """1 - (p1 - p2): maximal when the top two classes are tied."""
    p1, p2 = _top2(probs)
    return 1.0 - (p1 - p2)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy normalised by ``log(n_classes)``."""
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return 1.0 - _top2(probs)[0]
    p = np.clip(probs, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=1) / np.log(n_classes)
    # An all-zero row (no labelled neighbour) is as uncertain as it gets.
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
    """Per-row uncertainty of ``probs`` (n, n_classes), using the chosen measure."""
    try:
        fn = UNCERTAINTY_FUNCTIONS[kind]
    except KeyError:
        raise ValueError(f"Unknown uncertainty measure: {kind}") from None
    return fn(np.asarray(probs, dtype=np.float64))
