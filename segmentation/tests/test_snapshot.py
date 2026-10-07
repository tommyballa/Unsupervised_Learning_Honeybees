"""Snapshot immutabili: una riga per ape, con stato, predizioni e frame di origine."""
import csv
import json
import subprocess
import sys
from collections import Counter

import pytest
import yaml

from bee_al import hub as al_hub
from bee_al.config import REPO_ROOT
from bee_al.embeddings import save_embeddings
from bee_al.engine import ALEngine, ALError
from bee_al.snapshot import PREDICTION_FIELDS, read_csv, read_meta

from test_engine import label_batch

PATTERN = r"(?P<colony>hive[A-Z])_(?P<session>s\d)"


def labeled_engine(colony, rounds=3, reject=(), pattern=PATTERN):
    colony.cfg.frames.frame_pattern = pattern
    colony.cfg.labels.reject_classes = list(reject)
    engine = ALEngine(colony.cfg, colony.emb)
    for _ in range(rounds):
        label_batch(engine, colony.truth)
    return engine


def test_snapshot_has_every_file_and_one_row_per_bee(colony):
    engine = labeled_engine(colony)
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, "non so")
    engine.refresh()
    out = engine.export_snapshot("round-4_test.v1", "a@x.it")
    assert out == colony.cfg.results_dir / "snapshots" / "round-4_test.v1"
    assert {p.name for p in out.iterdir()} == {"predictions.csv", "frames.csv", "meta.json", "al_labels.csv", "validation_set.json"}

    rows = read_csv(out / "predictions.csv")
    assert list(rows[0].keys()) == PREDICTION_FIELDS
    assert [r["image_id"] for r in rows] == engine.bank.ids

    by_id = {r["image_id"]: r for r in rows}
    status = Counter(r["status"] for r in rows)
    st = engine.status()
    assert status["validation"] == len(engine.validation_ids) == 20
    assert status["unknown"] == st["n_unknown"] == 1 and by_id[item.image_id]["final_label"] == ""
    assert status["labeled"] == st["n_labels"] == 30
    assert status["unassigned"] == st["n_unassigned"] > 0
    assert sum(status.values()) == len(engine.bank)

    clusters = engine.final_clusters()                            # stessi cluster dell'engine
    in_clusters = {m["image_id"]: (label, m["source"]) for label, members in clusters.items() for m in members
                   if m["image_id"] not in engine.validation_ids}   # nello snapshot la validazione ha uno stato suo
    in_snapshot = {r["image_id"]: (r["final_label"], r["source"]) for r in rows if r["status"] in ("labeled", "predicted")}
    assert in_snapshot == in_clusters
    for r in rows:
        if r["status"] == "labeled":
            assert r["source"] == "human" and float(r["confidence"]) == 1.0 and r["final_label"] == colony.truth[r["image_id"]]
        if r["status"] == "predicted":
            assert r["final_label"] == r["propagation_label"] and 0 < float(r["confidence"]) <= 1
            assert r["knn_label"] and r["nearest_labeled_similarity"]
        if r["status"] in ("unassigned", "validation", "unknown"):
            assert r["final_label"] == "" and r["source"] == ""

    meta = read_meta(out)
    assert meta["annotator"] == "a@x.it" and meta["round"] == engine.round and meta["n_bees"] == len(rows)
    assert meta["status_counts"] == {**{s: 0 for s in meta["status_counts"]}, **status}
    assert meta["config"]["query"]["batch_size"] == 10 and meta["config_id"] == colony.cfg.config_id
    assert meta["n_label_events"] == 31 and "predictions_stable" in meta["stopping"]
    assert sum(c["human"] for c in meta["class_counts"].values()) == 30
    assert meta["frame_maps"]["tr"] == {**meta["frame_maps"]["tr"], "found": True, "reliable": True, "frames": 5}
    assert meta["frame_maps"]["tr"]["bees_linked"] == meta["frame_maps"]["tr"]["bees"]
    with open(out / "al_labels.csv") as f:
        assert len(list(csv.DictReader(f))) == 31


def test_snapshot_links_each_bee_to_its_frame_session_and_colony(colony):
    engine = labeled_engine(colony)
    out = engine.export_snapshot("s1", "a@x.it")
    for r in read_csv(out / "predictions.csv"):
        frame = colony.frame_of[r["image_id"]]
        colony_name, session = frame.split("_")[:2]
        assert (r["frame"], r["session"], r["colony"], r["dataset"]) == (frame, session, colony_name, r["image_id"].split("__")[0])
        assert r["filename"] == r["image_id"].split("__")[1] and r["path"].endswith(r["filename"])

    frames = read_csv(out / "frames.csv")
    assert len(frames) == 8 and {f["dataset"] for f in frames} == {"tr", "te"}
    first = frames[0]
    assert (first["frame"], first["raw_bees"], first["cleaned_accepted"], first["cleaned_discarded"]) == ("hiveA_s1_001.JPG", "60", "51", "9")
    assert json.loads(first["discard_reasons"]) == {"blur": 4, "border": 5}
    assert (first["session"], first["colony"]) == ("s1", "hiveA")
    empty = frames[1]
    assert (empty["frame"], empty["raw_bees"], empty["cleaned_accepted"]) == ("hiveA_s1_002.JPG", "0", "0")
    assert sum(int(f["cleaned_accepted"]) for f in frames) == len(colony.emb.ids)


def test_without_pattern_or_reliable_manifest_nothing_is_invented(colony, tmp_path):
    path = tmp_path / "te" / "raw_bees" / "_extraction_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["next_bee_id"] += 5
    path.write_text(json.dumps(manifest))
    with pytest.warns(UserWarning):
        engine = labeled_engine(colony, rounds=1, pattern="")
        out = engine.export_snapshot("s1", "a@x.it")
    rows = read_csv(out / "predictions.csv")
    assert {r["session"] for r in rows} == {"tr", "te"} and {r["colony"] for r in rows} == {""}
    assert all(r["frame"] == "" for r in rows if r["dataset"] == "te")
    assert all(r["frame"] for r in rows if r["dataset"] == "tr")
    info = read_meta(out)["frame_maps"]
    assert info["te"]["reliable"] is False and info["te"]["bees_linked"] == 0 and info["tr"]["reliable"] is True
    assert {f["dataset"] for f in read_csv(out / "frames.csv")} == {"tr"}


def test_snapshots_are_immutable_and_names_are_checked(colony):
    engine = labeled_engine(colony, rounds=1)
    out = engine.export_snapshot("uno", "a@x.it")
    before = (out / "predictions.csv").read_bytes()
    with pytest.raises(ALError, match="already exists"):
        engine.export_snapshot("uno", "b@x.it")
    for bad in ("", "con spazi", "../fuori", "a/b", ".nascosto", "x" * 81, "è"):
        with pytest.raises(ALError):
            engine.export_snapshot(bad, "a@x.it")
    assert (out / "predictions.csv").read_bytes() == before
    snaps = colony.cfg.results_dir / "snapshots"
    assert [p.name for p in snaps.iterdir()] == ["uno"]            # nessuna cartella temporanea rimasta


def test_reproducible_without_new_labels_and_listed(colony):
    engine = labeled_engine(colony)
    a = engine.export_snapshot("a", "a@x.it")
    engine.refresh()                                              # un ricalcolo senza etichette nuove
    b = engine.export_snapshot("b", "a@x.it")
    assert (a / "predictions.csv").read_bytes() == (b / "predictions.csv").read_bytes()

    label_batch(engine, colony.truth)
    c = engine.export_snapshot("c", "a@x.it")
    assert (a / "predictions.csv").read_bytes() != (c / "predictions.csv").read_bytes()
    listed = engine.list_snapshots()
    assert {s["name"] for s in listed} == {"a", "b", "c"} and listed[-1]["name"] == "a"
    assert all(s["n_bees"] == len(colony.emb.ids) and s["model_version"] == "v000" and not s["has_audit"] for s in listed)
    assert engine.status()["snapshots"] == listed


def test_reject_classes_are_marked_rejected(colony):
    engine = labeled_engine(colony, reject=["Class_1"])
    rows = read_csv(engine.export_snapshot("s", "a@x.it") / "predictions.csv")
    rejected = [r for r in rows if r["status"] == "rejected"]
    assert rejected and {r["final_label"] for r in rejected} == {"Class_1"}
    assert {r["source"] for r in rejected} == {"human", "predicted"}
    assert not any(r["final_label"] == "Class_1" for r in rows if r["status"] in ("labeled", "predicted"))


def test_export_script_does_not_add_a_round(colony, tmp_path):
    engine = labeled_engine(colony)
    rounds_before = (colony.cfg.results_dir / "al_rounds.jsonl").read_text()
    save_embeddings(colony.cfg.embeddings_dir, colony.emb.ids, colony.emb.paths, colony.emb.X)
    config_path = tmp_path / "cfg.yaml"
    data = colony.cfg.to_dict()
    data["paths"] = {k: (str(colony.cfg.resolve(v)) if v else "") if isinstance(v, str) else {n: str(colony.cfg.resolve(d)) for n, d in v.items()}
                     for k, v in data["paths"].items()}
    config_path.write_text(yaml.safe_dump(data))

    script = REPO_ROOT / "scripts" / "export_snapshot.py"
    run = subprocess.run([sys.executable, str(script), "--name", "da_script", "--config", str(config_path), "--annotator", "cli"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    out = colony.cfg.results_dir / "snapshots" / "da_script"
    assert read_meta(out)["annotator"] == "cli" and len(read_csv(out / "predictions.csv")) == len(colony.emb.ids)
    assert (colony.cfg.results_dir / "al_rounds.jsonl").read_text() == rounds_before
    engine.export_snapshot("da_engine", "a@x.it")
    key = lambda path: [(r["image_id"], r["status"], r["final_label"], r["frame"]) for r in read_csv(path / "predictions.csv")]
    assert key(out) == key(out.parent / "da_engine")              # stesse predizioni dell'engine dell'hub

    again = subprocess.run([sys.executable, str(script), "--name", "da_script", "--config", str(config_path)], capture_output=True, text=True)
    assert again.returncode != 0 and "already exists" in again.stderr


def test_hub_freezes_and_lists_snapshots(colony, hub_app, monkeypatch, tmp_path):
    engine = labeled_engine(colony, rounds=1)
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    assert c.post("/al-snapshot", data={"name": "x"}).status_code == 302          # senza login non crea nulla
    assert engine.list_snapshots() == []
    c.post("/", data={"email": "t@x.it"})
    assert "No snapshots yet" in c.get("/al-status").get_data(as_text=True)

    r = c.post("/al-snapshot", data={"name": "primo"})
    assert r.status_code == 302
    html = c.get(r.headers["Location"]).get_data(as_text=True)
    assert "created" in html and "<td>primo</td>" in html and "Freeze snapshot" in html
    assert read_meta(colony.cfg.results_dir / "snapshots" / "primo")["annotator"] == "t@x.it"

    for bad in ("primo", "nome non valido"):                                       # errori come messaggio, non 500
        r = c.post("/al-snapshot", data={"name": bad})
        page = c.get(r.headers["Location"])
        assert r.status_code == 302 and page.status_code == 200 and "Snapshot not created" in page.get_data(as_text=True)
    assert len(engine.list_snapshots()) == 1
