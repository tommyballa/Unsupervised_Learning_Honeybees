"""Fixture condivise: dati sintetici, nessuna immagine reale e nessuna GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util  # noqa: E402

from bee_al.config import REPO_ROOT, ALConfig, config_from_dict  # noqa: E402
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


@pytest.fixture(scope="session")
def hub_app(tmp_path_factory):
    """Importa app.py da una cartella temporanea: il suo ``results`` relativo finisce lì."""
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("hub_cwd"))
    try:
        spec = importlib.util.spec_from_file_location("hub_app_under_test", REPO_ROOT / "active_learning_tool" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module
