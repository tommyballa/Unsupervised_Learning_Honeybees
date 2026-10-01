"""Fixture condivise: dati sintetici, nessuna immagine reale e nessuna GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import ALConfig, config_from_dict  # noqa: E402
from bee_al.simulation import make_synthetic  # noqa: E402


@pytest.fixture
def cfg(tmp_path) -> ALConfig:
    """Config piccola che scrive solo sotto ``tmp_path``."""
    return config_from_dict(
        {
            "query": {"batch_size": 10, "cold_start_threshold": 10},
            "validation": {"size": 20},
            "propagation": {"n_neighbors": 8},
        },
        root=tmp_path,
    )


@pytest.fixture
def synth():
    """300 punti, 4 classi sbilanciate, embedding normalizzati L2."""
    X, y = make_synthetic(n=300, dim=32, n_classes=4, imbalance=0.6, noise=0.6, seed=0)
    ids = [f"bee_{i:05d}_conf0.90.png" for i in range(len(X))]
    return X, np.asarray(y), ids
