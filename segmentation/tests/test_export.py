"""Current phase shown prominently, only the folders in the config, and closing the phase with the clusters on disk."""
import csv
import json
import os
import shutil
import threading

import numpy as np
import pytest
import yaml

from bee_al import hub as al_hub
from bee_al.config import config_from_dict
from bee_al.embeddings import Embeddings, restrict_to_datasets, save_embeddings
from bee_al.engine import ALEngine
from bee_al.export import cluster_folder, list_exports
from bee_al.snapshot import read_csv

from conftest import build_colony_dataset
from test_engine import label_batch


@pytest.fixture
def phase(tmp_path):
    data = build_colony_dataset(tmp_path, extra_cfg={
        "stage": {"name": "fase1", "title": "Phase 1 of 2 - segmentation quality", "description": "Well or badly segmented?"},
        "export": {"clusters_dir": "datasets/clusters"},
        "labels": {"classes": ["Segmentata Bene", "Segmentata Male"]},
    })
    for image_id, path in zip(data.emb.ids, data.emb.paths):                 # real files, all different
        open(path, "wb").write(image_id.encode())
    data.good = {i: ("Segmentata Bene" if data.truth[i] in ("Class_0", "Class_1") else "Segmentata Male") for i in data.truth}
    data.engine = ALEngine(data.cfg, data.emb)
    for _ in range(4):
        label_batch(data.engine, data.good)
    return data


def test_only_configured_dataset_folders_are_used(tmp_path):
    data = build_colony_dataset(tmp_path)
    only_train = restrict_to_datasets(data.emb, ["tr"])
    assert only_train.ids == [i for i in data.emb.ids if i.startswith("tr__")] and 0 < len(only_train) < len(data.emb)
    assert restrict_to_datasets(data.emb, ["tr", "te"]) is data.emb
    with pytest.raises(ValueError, match="extract_embeddings"):
        restrict_to_datasets(data.emb, ["altro"])

    # embeddings extracted for train + test, config with train only: the hub never sees the test set
    save_embeddings(data.cfg.embeddings_dir, data.emb.ids, data.emb.paths, data.emb.X)
    cfg = data.cfg.to_dict()
    cfg["paths"] = {"dataset_dirs": {"tr": str(tmp_path / "tr" / "cleaned_bees")}, "results_dir": str(data.cfg.results_dir),
                    "embeddings_dir": str(data.cfg.embeddings_dir), "models_dir": str(data.cfg.models_dir)}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    engine = ALEngine.from_config(path)
    assert len(engine.bank) == len(only_train) and all(i.startswith("tr__") for i in engine.bank.ids)
    assert all(i.startswith("tr__") for i in engine.validation_ids) and all(b.image_id.startswith("tr__") for b in engine.batch)


def test_cluster_folder_names():
    assert cluster_folder("Segmentata Bene") == "Segmentata_Bene"
    assert cluster_folder("  a/b: c?  ") == "a_b_c" and cluster_folder("../..") == "senza_nome"
    assert config_from_dict({}).clusters_dir is None
    with pytest.raises(ValueError):
        config_from_dict({"export": {"mode": "sposta"}})


def test_export_writes_config_and_one_folder_per_cluster(phase):
    engine, cfg = phase.engine, phase.cfg
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, "scarta ape")
    engine.exporter.start("fase1_finale", "a@x.it")
    engine.exporter.join(30)
    st = engine.exporter.status()
    assert st["state"] == "idle" and st["error"] == "" and st["progress"]["copied"] == st["progress"]["total"] > 0

    out = cfg.clusters_dir / "fase1_finale"
    assert out == phase.cfg.resolve("datasets/clusters") / "fase1_finale"
    assert {p.name for p in out.iterdir()} == {"Segmentata_Bene", "Segmentata_Male", "clusters.csv", "config.yaml", "meta.json"}
    assert not [p for p in cfg.clusters_dir.iterdir() if p.name.startswith(".")]          # no temporary folder

    snapshot = read_csv(cfg.results_dir / "snapshots" / "fase1_finale" / "predictions.csv")   # same name: frozen snapshot
    table = read_csv(out / "clusters.csv")
    assert [r["image_id"] for r in table] == [r["image_id"] for r in snapshot] == engine.bank.ids
    in_clusters = [r for r in snapshot if r["status"] in ("labeled", "predicted")]
    for folder, label in (("Segmentata_Bene", "Segmentata Bene"), ("Segmentata_Male", "Segmentata Male")):
        files = sorted(p.name for p in (out / folder).iterdir())
        assert files == sorted(r["image_id"] for r in in_clusters if r["final_label"] == label) and files
        one = out / folder / files[0]
        assert one.read_bytes() == files[0].encode() and not one.is_symlink()             # real copy of the image
    by_id = {r["image_id"]: r for r in table}
    assert by_id[item.image_id]["status"] == "unknown" and by_id[item.image_id]["file"] == ""   # "Scarta ape": only in the csv
    assert all(by_id[i]["file"] == "" and by_id[i]["status"] == "validation" for i in engine.validation_ids)
    some = next(r for r in table if r["source"] == "human")
    assert some["file"] == f"{cluster_folder(some['cluster'])}/{some['image_id']}" and some["frame"] and some["original_path"]

    meta = json.loads((out / "meta.json").read_text())
    assert meta["stage"] == "fase1" and meta["annotator"] == "a@x.it" and meta["mode"] == "copy" and meta["missing_files"] == 0
    assert meta["n_bees"] == len(table) and meta["n_in_clusters"] == len(in_clusters)
    assert {k: v["total"] for k, v in meta["clusters"].items()} == {
        "Segmentata Bene": sum(r["final_label"] == "Segmentata Bene" for r in in_clusters),
        "Segmentata Male": sum(r["final_label"] == "Segmentata Male" for r in in_clusters)}
    assert sum(v["human"] for v in meta["clusters"].values()) == 40
    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["stage"]["name"] == "fase1" and saved["query"]["batch_size"] == 10 and saved == cfg.to_dict()
    assert [e["name"] for e in list_exports(cfg)] == ["fase1_finale"] and engine.status()["export"]["exports"][0]["path"] == str(out)

    # labelling can continue: the export stays as it was
    before = (out / "clusters.csv").read_bytes()
    label_batch(engine, phase.good)
    assert (out / "clusters.csv").read_bytes() == before


def test_export_modes_and_errors(phase):
    engine, cfg = phase.engine, phase.cfg
    for mode in ("symlink", "hardlink"):
        cfg.export.mode = mode
        engine.exporter.start(f"prova_{mode}", "a@x.it")
        engine.exporter.join(30)
        assert engine.exporter.state == "idle", engine.exporter.error
        one = next(p for p in (cfg.clusters_dir / f"prova_{mode}" / "Segmentata_Bene").iterdir())
        assert one.is_symlink() == (mode == "symlink") and one.read_bytes() == one.name.encode()
        assert (one.stat().st_nlink > 1) == (mode == "hardlink")
    for bad, text in (("prova_symlink", "already"), ("con spazi", "name"), ("../fuori", "name")):
        with pytest.raises(ValueError, match=text):
            engine.exporter.start(bad, "a@x.it")
    assert len(list_exports(cfg)) == 2

    cfg.export.clusters_dir = ""
    with pytest.raises(ValueError, match="clusters_dir"):
        engine.exporter.start("x", "a@x.it")
    cfg.export.clusters_dir = "datasets/clusters"
    empty = ALEngine(build_colony_dataset(phase.cfg.resolve("altro"), extra_cfg={"export": {"clusters_dir": "c"}}).cfg, phase.emb)
    with pytest.raises(ValueError, match="no labels"):
        empty.exporter.start("x", "a@x.it")

    os.remove(phase.emb.paths[engine.bank.labeled_indices()[0]])               # a missing file: reported, does not break
    cfg.export.mode = "copy"
    engine.exporter.start("con_buco", "a@x.it")
    engine.exporter.join(30)
    assert json.loads((cfg.clusters_dir / "con_buco" / "meta.json").read_text())["missing_files"] == 1


def test_failed_export_leaves_nothing_behind(phase, monkeypatch):
    engine, cfg = phase.engine, phase.cfg
    def boom(*args, **kwargs):
        raise OSError("disco pieno")

    monkeypatch.setattr("bee_al.export.yaml.safe_dump", boom)                  # fails after copying the images
    engine.exporter.start("rotto", "a@x.it")
    engine.exporter.join(30)
    assert engine.exporter.state == "failed" and "disco pieno" in engine.exporter.status()["error"]
    assert not (cfg.clusters_dir / "rotto").exists() and list_exports(cfg) == []
    assert not [p for p in cfg.clusters_dir.iterdir()]



def test_export_freezes_the_snapshot_in_the_background(phase, monkeypatch):
    """On the results disk the snapshot takes minutes: the button must not wait for it."""
    engine, cfg = phase.engine, phase.cfg
    release = threading.Event()
    real = engine.export_snapshot

    def slow_snapshot(name, annotator):
        release.wait(30)
        return real(name, annotator)

    monkeypatch.setattr(engine, "export_snapshot", slow_snapshot)
    engine.exporter.start("lento", "a@x.it")                                    # returns immediately
    assert engine.exporter.state == "running" and engine.exporter.progress["total"] == 0
    release.set()
    engine.exporter.join(30)
    assert engine.exporter.state == "idle", engine.exporter.error
    assert (cfg.clusters_dir / "lento" / "clusters.csv").exists()
    with pytest.raises(ValueError, match="already"):                           # the snapshot already exists
        engine.exporter.start("lento", "a@x.it")
    shutil.rmtree(cfg.clusters_dir / "lento")
    with pytest.raises(ValueError, match="snapshot 'lento' already exists"):
        engine.exporter.start("lento", "a@x.it")

@pytest.fixture
def client(phase, hub_app, monkeypatch, tmp_path):
    monkeypatch.setattr(al_hub.STATE, "engine", phase.engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    c.phase = phase
    return c


def test_every_page_says_which_phase_this_is(client):
    for url in ("/al-grid", "/al-labeling", "/al-labeling?mode=validation", "/al-clusters", "/al-status", "/al-grid?mode=validation"):
        html = client.get(url).get_data(as_text=True)
        assert "Phase 1 of 2 - segmentation quality" in html and "Well or badly segmented?" in html, url
        assert "clusters saved as" not in html
    client.phase.cfg.stage.title = ""                                          # no phase in the config: no band
    assert "Phase 1 of 2" not in client.get("/al-status").get_data(as_text=True)


def test_export_button(client):
    phase = client.phase
    html = client.get("/al-status").get_data(as_text=True)
    assert "Save configuration &amp; export clusters" in html and str(phase.cfg.clusters_dir) in html

    r = client.post("/al-export", data={"name": "nome non valido"})
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert r.status_code == 302 and "Export not started" in page

    gate = threading.Event()
    original = phase.engine.exporter._run
    phase.engine.exporter._run = lambda *a: (gate.wait(10), original(*a))       # holds the thread to observe "running"
    assert client.post("/al-export", data={"name": "fase1_finale"}).status_code == 302
    running = client.get("/al-status").get_data(as_text=True)
    assert "Export running" in running and 'http-equiv="refresh"' in running
    assert "Export not started" in client.get(client.post("/al-export", data={"name": "altro"}).headers["Location"]).get_data(as_text=True)
    gate.set()
    phase.engine.exporter.join(30)

    html = client.get("/al-status").get_data(as_text=True)
    assert "<td>fase1_finale</td>" in html and "Segmentata Bene:" in html and "Export running" not in html
    assert (phase.cfg.clusters_dir / "fase1_finale" / "meta.json").exists()
    assert json.loads((phase.cfg.clusters_dir / "fase1_finale" / "meta.json").read_text())["annotator"] == "t@x.it"
    # the phase band says the clusters have been saved
    assert "clusters saved as fase1_finale" in client.get("/al-grid").get_data(as_text=True)
    assert "/al-export" in {r.rule for r in client.application.url_map.iter_rules()}
