"""Dataset made of several folders with repeated file names."""
import re
from pathlib import Path

import numpy as np
import pytest

from bee_al import hub as al_hub
from bee_al.config import config_from_dict, load_config
from bee_al.contrastive import ContrastiveTrainer
from bee_al.embeddings import Embeddings, extract_embeddings, list_dataset, load_embeddings
from bee_al.engine import ALEngine
from bee_al.retrain import TrainingData
from bee_al.validation import load_or_create_validation_set


@pytest.fixture
def two_folders(tmp_path):
    """Two folders with the same file names (like the train and test cleaned_bees)."""
    for name, n in (("train", 5), ("test", 3)):
        d = tmp_path / name / "cleaned_bees"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"bee_{i:05d}_conf0.90.png").write_bytes(f"{name}-{i}".encode())
        (d / "_cleaning_manifest.json").write_text("{}")
    return config_from_dict(
        {"paths": {"dataset_dirs": {"tr": "train/cleaned_bees", "te": "test/cleaned_bees"}},
         "query": {"batch_size": 2, "cold_start_threshold": 2}, "validation": {"size": 2},
         "propagation": {"n_neighbors": 3}},
        root=tmp_path,
    )


def _encoder(paths):
    return np.stack([np.random.default_rng(sum(Path(p).read_bytes())).normal(size=8) for p in paths])


def test_default_config_uses_the_sam3_folder():
    cfg = load_config()
    sources = dict(cfg.dataset_sources)
    assert set(sources) == {"sam3_b3456__"}                                  # all photos of batches 3-6
    assert sources["sam3_b3456__"].parts[-2:] == ("single_bee3+4+5+6_SAM3", "cleaned_bees")
    assert cfg.embeddings_dir.name == "embeddings_sam3_b3456"
    assert cfg.stage.name == "fase1" and "segmentation" in cfg.stage.title.lower()
    assert cfg.clusters_dir.name == "single_bee3+4+5+6_SAM3_clusters" and cfg.export.mode == "copy"
    assert cfg.clusters_dir.parent == sources["sam3_b3456__"].parents[1]      # next to the other datasets, in its own folder


def test_sources_and_unique_ids(two_folders, tmp_path):
    cfg = two_folders
    assert [p for p, _ in cfg.dataset_sources] == ["tr__", "te__"]
    assert cfg.dataset_dir == tmp_path / "train" / "cleaned_bees"
    ids, paths = list_dataset(cfg.dataset_sources)
    assert len(ids) == len(set(ids)) == 8                                   # 5 + 3, no manifest
    assert ids[0] == "tr__bee_00000_conf0.90.png" and ids[5] == "te__bee_00000_conf0.90.png"
    assert paths[0].name == paths[5].name and paths[0] != paths[5]

    single = config_from_dict({"paths": {"dataset_dir": "train/cleaned_bees"}}, root=tmp_path)
    assert list_dataset(single.dataset_sources)[0][0] == "bee_00000_conf0.90.png"   # a single folder: id = file name
    missing = config_from_dict({"paths": {"dataset_dirs": {"a": "non_esiste", "tr": "train/cleaned_bees"}}}, root=tmp_path)
    assert len(list_dataset(missing.dataset_sources)[0]) == 5              # folder not created yet: skipped
    with pytest.raises(ValueError, match="nome non valido"):
        config_from_dict({"paths": {"dataset_dirs": {"a__b": "x"}}})


def test_extraction_keeps_same_named_files_apart_and_resumes(two_folders, tmp_path):
    cfg = two_folders
    ids, paths = list_dataset(cfg.dataset_sources)
    out = tmp_path / "emb"
    assert extract_embeddings(paths[:6], out, _encoder, image_ids=ids[:6], shard_size=4) == 6
    assert extract_embeddings(paths, out, _encoder, image_ids=ids, shard_size=4) == 2    # resume: only the new ones
    emb = load_embeddings(out)
    assert emb.ids == ids and emb.paths == [str(p) for p in paths]
    assert not np.allclose(emb.X[0], emb.X[5])                              # same name, different bees
    with pytest.raises(ValueError):
        extract_embeddings(paths, tmp_path / "e2", _encoder, image_ids=[p.name for p in paths])  # repeated ids


def test_retrain_job_carries_ids(two_folders):
    ids, paths = list_dataset(two_folders.dataset_sources)
    paths = [str(p) for p in paths]
    labeled = TrainingData(ids[:2], paths[:2], np.zeros((2, 2)), ["A", "B"])
    unlabeled = TrainingData(ids[2:], paths[2:], np.zeros((6, 2)), pseudo_labels=[None] * 6)
    job = ContrastiveTrainer(two_folders).build_job(labeled, unlabeled)
    assert job["all_ids"] == ids and job["all_paths"] == paths


def test_validation_file_of_another_dataset_is_refused(tmp_path):
    path = tmp_path / "validation_set.json"
    old = load_or_create_validation_set(path, [f"bee_{i}.png" for i in range(20)], 5, 0)
    assert len(old) == 5
    with pytest.raises(ValueError, match="altro"):
        load_or_create_validation_set(path, [f"tr__bee_{i}.png" for i in range(20)], 5, 0)


def test_hub_serves_images_from_both_folders(two_folders, monkeypatch, tmp_path, request):
    hub_app = request.getfixturevalue("hub_app")
    cfg = two_folders
    ids, paths = list_dataset(cfg.dataset_sources)
    X = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    engine = ALEngine(cfg, Embeddings(ids, [str(p) for p in paths], X))
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    assert c.get(f"/al-image/{ids[0]}").status_code == 403                   # login required
    c.post("/", data={"email": "t@x.it"})

    assert c.get(f"/al-image/{ids[0]}").data == b"train-0"
    assert c.get(f"/al-image/{ids[5]}").data == b"test-0"                    # same file name, different folder
    assert c.get("/al-image/non_esiste.png").status_code == 404
    assert c.get("/al-image/..%2F..%2Fetc%2Fpasswd").status_code == 404
    paths[1].unlink()
    assert c.get(f"/al-image/{ids[1]}").status_code == 404                   # file removed after extraction

    html = c.get("/al-labeling").get_data(as_text=True)
    shown = re.search(r'name="image_id" value="([^"]+)"', html).group(1)
    assert shown in ids and f"/al-image/{shown}" in html
    c.post("/al-labeling", data={"image_id": shown, "label": "Sana", "action": "label"})
    assert engine.store.events()[-1].image_id == shown and cfg.al_labels_path.parent.name == "results"
