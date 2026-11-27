"""Shared fixtures: synthetic data, no real images and no GPU."""
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
    """Small config that writes only under ``tmp_path``."""
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
    """300 points, 4 imbalanced classes, L2-normalised embeddings."""
    X, y = make_synthetic(n=300, dim=32, n_classes=4, imbalance=0.6, noise=0.6, seed=0)
    ids = [f"bee_{i:05d}_conf0.90.png" for i in range(len(X))]
    return X, np.asarray(y), ids


@pytest.fixture(scope="session")
def hub_app(tmp_path_factory):
    """Import app.py from a temporary folder: its relative ``results`` folder ends up there."""
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


def build_colony_dataset(root, folders=None, discard_every=7, seed=0, extra_cfg=None):
    """Complete fake dataset: frames -> raw bees -> cleaned bees, with the manifests of both scripts.

    ``folders``: ``{folder name: {frame name: raw bees}}``. Numbering restarts from 0 in
    each folder; cleaning discards one bee every ``discard_every`` (without renumbering).
    Returns config, synthetic embeddings (4 classes), ground truth per image_id and frame per image_id.
    """
    import json
    from types import SimpleNamespace

    from bee_al.embeddings import Embeddings

    folders = folders or {
        "tr": {"hiveA_s1_001.JPG": 60, "hiveA_s1_002.JPG": 0, "hiveA_s2_003.JPG": 50, "hiveB_s1_004.JPG": 70, "hiveB_s2_005.JPG": 40},
        "te": {"hiveA_s2_006.JPG": 45, "hiveB_s1_007.JPG": 35, "hiveB_s2_008.JPG": 50},
    }
    ids, paths, frame_of = [], [], {}
    for name, frames in folders.items():
        raw, cleaned = root / name / "raw_bees", root / name / "cleaned_bees"
        raw.mkdir(parents=True)
        cleaned.mkdir(parents=True)
        processed, reasons, counter = {}, {}, 0
        for frame, n in frames.items():
            for _ in range(n):
                filename = f"bee_{counter:05d}_conf0.90.png"
                if counter % discard_every == 0:
                    processed[filename] = "discarded"
                    reasons[filename] = "blur" if counter % 2 else "border"
                else:
                    processed[filename] = "accepted"
                    (cleaned / filename).write_bytes(b"png")
                    ids.append(f"{name}__{filename}")
                    paths.append(str(cleaned / filename))
                    frame_of[ids[-1]] = frame
                counter += 1
        (raw / "_extraction_manifest.json").write_text(json.dumps(
            {"processed_sources": {f: {"bees": n} for f, n in frames.items()}, "next_bee_id": counter}))
        (cleaned / "_cleaning_manifest.json").write_text(json.dumps(
            {"params": {}, "processed_files": processed, "discard_reasons": reasons}))

    X, y = make_synthetic(n=len(ids), dim=32, n_classes=4, imbalance=0.6, noise=0.6, seed=seed)
    data = {
        "paths": {"dataset_dirs": {name: f"{name}/cleaned_bees" for name in folders}},
        "query": {"batch_size": 10, "cold_start_threshold": 10},
        "validation": {"size": 20},
        "propagation": {"n_neighbors": 8},
    }
    for section, values in (extra_cfg or {}).items():
        data.setdefault(section, {}).update(values)
    cfg = config_from_dict(data, root=root)
    return SimpleNamespace(cfg=cfg, emb=Embeddings(ids, paths, X), truth=dict(zip(ids, y)), frame_of=frame_of,
                           folders=folders)


@pytest.fixture
def colony(tmp_path):
    return build_colony_dataset(tmp_path)
