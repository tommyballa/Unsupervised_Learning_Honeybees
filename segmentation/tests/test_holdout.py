"""Batch tenuto da parte per validare il modello, e filtro per batch della pagina dei cluster."""
import csv
import re

import numpy as np
import pytest

from bee_al import hub as al_hub
from bee_al.audit import create_audit_set
from bee_al.config import config_from_dict
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine, ALError
from bee_al.label_store import is_training_event


@pytest.fixture
def held_cfg(tmp_path):
    """Due cartelle: ``tr`` (training) e ``b7`` (tenuta da parte)."""
    return config_from_dict(
        {"paths": {"dataset_dirs": {"tr": "tr", "b7": "b7"}},
         "query": {"batch_size": 10, "cold_start_threshold": 10}, "validation": {"size": 20, "holdout_datasets": ["b7"]},
         "propagation": {"n_neighbors": 8}, "audit": {"per_stratum": 3, "datasets": ["b7"]}},
        root=tmp_path,
    )


def _ids(synth):
    X, y, _ = synth
    ids = [f"{'b7' if i % 4 == 0 else 'tr'}__bee_{i:05d}_conf0.90.png" for i in range(len(X))]
    return X, dict(zip(ids, y)), ids


def make_engine(cfg, synth):
    X, _, ids = _ids(synth)
    return ALEngine(cfg, Embeddings(ids, [f"/d/{i}" for i in ids], X))


def label_rounds(engine, truth, rounds=3):
    for _ in range(rounds):
        batch_id = engine.batch_id
        while engine.batch_id == batch_id:
            item = engine.next_image("a@x.it")
            if item is None:
                break
            engine.submit("a@x.it", item.image_id, truth[item.image_id], "High")


def test_config_checks_the_held_out_datasets(tmp_path):
    with pytest.raises(ValueError, match="holdout_datasets"):
        config_from_dict({"paths": {"dataset_dirs": {"tr": "tr"}}, "validation": {"holdout_datasets": ["b7"]}})


def test_held_out_bees_are_never_proposed_and_never_learned(held_cfg, synth):
    engine = make_engine(held_cfg, synth)
    _, truth, ids = _ids(synth)
    held = {i for i in ids if i.startswith("b7__")}
    label_rounds(engine, truth)
    proposed = {e.image_id for e in engine.store.events()}
    assert proposed and not proposed & held
    assert engine.status()["holdout"] == {"datasets": ["b7"], "n_bees": len(held), "n_labels": 0}

    # Una correzione dalla pagina dei cluster su un'ape tenuta da parte: evento "holdout", la bank non cambia.
    bee = sorted(held)[0]
    before = engine.final_pred.probs.copy()
    assert engine.assign_many("a@x.it", [bee], "Zz") == 1
    assert engine.bank.label_of(bee) is None and engine.holdout_labels == {bee: "Zz"}
    assert "Zz" not in engine.bank.classes and np.array_equal(engine.final_pred.probs, before)
    event = engine.store.events()[-1]
    assert event.query_type == "holdout" and not is_training_event(event)
    assert engine.assign_many("a@x.it", [bee], "Zz") == 0                    # nessun evento inutile
    with pytest.raises(ALError):
        engine.submit("a@x.it", bee, "Zz")                                   # solo dalla pagina dei cluster

    engine.refresh()                                                         # al riavvio si ricostruisce dallo store
    assert engine.holdout_labels == {bee: "Zz"} and engine.bank.label_of(bee) is None
    assert not {b.image_id for b in engine.batch} & held


def test_held_out_bees_do_not_move_the_unassigned_threshold(held_cfg, synth, tmp_path):
    engine = make_engine(held_cfg, synth)
    _, truth, ids = _ids(synth)
    label_rounds(engine, truth)
    free = engine._not_validation() & ~engine._holdout
    free[engine.bank.labeled_indices()] = False
    free[engine.bank.unknown_indices()] = False
    assert (engine.unassigned & ~engine._holdout).sum() == round(0.05 * free.sum())
    threshold = engine.nearest_sim[engine.unassigned & ~engine._holdout].max()
    held_far = engine.unassigned & engine._holdout
    assert np.all(engine.nearest_sim[held_far] <= threshold)
    assert np.all(engine.nearest_sim[engine._holdout & engine._not_validation() & ~held_far] > threshold)


def test_cluster_view_filters_by_batch(held_cfg, synth):
    engine = make_engine(held_cfg, synth)
    _, truth, ids = _ids(synth)
    label_rounds(engine, truth)
    assert engine.batch_counts() == {"b7": 75, "tr": 225}                   # senza manifest: batch = cartella
    assert engine.held_out_batches() == {"b7"}
    name = max(engine.cluster_samples(0), key=lambda c: c["n_predicted"])["label"]
    every = engine.cluster_view("predicted", name, 96, 0, "similar")
    only = engine.cluster_view("predicted", name, 96, 0, "similar", batch="b7")
    assert 0 < only["n_members"] < every["n_members"]
    assert all(b["image_id"].startswith("b7__") and b["holdout"] for b in only["samples"])
    by_batch = sum(c["n_predicted"] for c in engine.cluster_samples(0, batch="b7") if c["label"] == name)
    assert by_batch == only["n_members"]
    assert engine.cluster_view("predicted", name, 12, 0, batch="nessuno") is None


def test_audit_can_be_limited_to_the_held_out_batch(held_cfg, synth):
    engine = make_engine(held_cfg, synth)
    _, truth, _ = _ids(synth)
    label_rounds(engine, truth)
    folder = engine.export_snapshot("s1", "a@x.it")
    create_audit_set(folder, held_cfg)
    with open(folder / "audit_strata.csv", newline="", encoding="utf-8") as f:
        population = [r["image_id"] for r in csv.DictReader(f)]
    assert population and all(i.startswith("b7__") for i in population)


def test_retrain_leaves_out_the_held_out_bees(held_cfg, synth):
    from bee_al.retrain import RetrainManager

    engine = make_engine(held_cfg, synth)
    _, truth, _ = _ids(synth)
    label_rounds(engine, truth)
    seen = {}

    class Spy:
        def retrain(self, labeled, unlabeled):
            seen["ids"] = labeled.ids + unlabeled.ids
            raise RuntimeError("stop")

    manager = RetrainManager.attach(engine, Spy())
    manager.start()
    manager.join(10)
    assert seen["ids"] and not any(i.startswith("b7__") for i in seen["ids"])


def test_clusters_page_has_a_batch_menu(hub_app, held_cfg, synth, monkeypatch, tmp_path):
    engine = make_engine(held_cfg, synth)
    _, truth, _ = _ids(synth)
    label_rounds(engine, truth)
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    hub_app.app.config["TESTING"] = True
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})

    html = c.get("/al-clusters").get_data(as_text=True)
    assert 'id="batch_select"' in html and "All batches" in html and "b7 (75 bees) - held out" in html
    held = c.get("/al-clusters?batch=b7").get_data(as_text=True)
    assert "is held out to validate the model" in held
    tiles = re.findall(r'name="image_(\d+)" value="([^"]+)"', held)
    assert tiles and all(i.startswith("b7__") for _, i in tiles) and "held out" in held
    again = c.get("/al-clusters").get_data(as_text=True)                     # la scelta resta nella sessione
    assert all(i.startswith("b7__") for _, i in re.findall(r'name="image_(\d+)" value="([^"]+)"', again))
    every = c.get("/al-clusters?batch=").get_data(as_text=True)
    assert "is held out" not in every

    k, bee = tiles[0]
    r = c.post("/al-clusters?group=" + re.search(r'<option value="/al-clusters\?group=([^"&]+)[^"]*" selected', held).group(1),
               data={f"image_{k}": bee, f"group_{k}": "0", "label_0": "Zz"})
    assert r.status_code == 302 and engine.holdout_labels == {bee: "Zz"} and engine.bank.label_of(bee) is None
