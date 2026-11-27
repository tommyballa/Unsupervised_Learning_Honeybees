"""Stopping criteria: prediction stability, class discovery, macro-F1 plateau."""
import json

import numpy as np
import pytest

from bee_al import hub as al_hub
from bee_al.config import config_from_dict
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine

from test_engine import label_batch, make_engine


@pytest.fixture
def truth(synth):
    _, y, ids = synth
    return dict(zip(ids, y))


def test_config_defaults_and_checks():
    cfg = config_from_dict({})
    assert (cfg.stopping.window, cfg.stopping.max_pred_change, cfg.stopping.f1_min_delta) == (3, 0.01, 0.01)
    for bad in ({"window": 0}, {"max_pred_change": 1.5}, {"f1_min_delta": -0.1}):
        with pytest.raises(ValueError):
            config_from_dict({"stopping": bad})


def test_pred_change_on_a_hand_built_case(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    assert engine.history[0]["pred_change"] is None                  # first round: nothing to compare with
    label_batch(engine, truth)
    label_batch(engine, truth)
    assert 0 <= engine.history[-1]["pred_change"] <= 1

    # Tamper with the record of the previous round: 7 bees "had" a different class.
    prev = engine._round_final.copy()
    comparable = np.flatnonzero(np.not_equal(prev, None))
    prev[comparable[:7]] = "Altra Classe"
    engine._round_final = prev
    entry = engine.refresh()                                         # no new labels: the predictions do not change
    assert entry["pred_change"] == pytest.approx(7 / len(comparable))
    assert entry["effective"] is False and entry["model_changed"] is False

    excluded = len(synth[2]) - len(comparable)                       # validation + unassigned
    assert excluded >= len(engine.validation_ids)
    assert engine.refresh()["pred_change"] == 0.0


def test_pred_change_ignores_validation_unknown_and_unassigned(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, "non so")
    label_batch(engine, truth)
    final = engine._final_labels()
    ids = engine.bank.ids
    assert final[ids.index(item.image_id)] is None
    assert all(final[ids.index(i)] is None for i in engine.validation_ids)
    assert all(final[i] is None for i in np.flatnonzero(engine.unassigned))
    labeled = engine.bank.labeled_indices()
    assert list(final[labeled]) == engine.bank.labels_at(labeled)    # for labelled bees the human label counts


def test_new_classes_and_rounds_since_new_class(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    assert engine.history[0]["classes"] == [] and engine.history[0]["rounds_since_new_class"] == 0

    def label_all(label):
        batch_id = engine.batch_id
        while engine.batch_id == batch_id:
            engine.submit("a@x.it", engine.next_image("a@x.it").image_id, label)
        return engine.history[-1]

    entry = label_all("Sana")
    assert entry["new_classes"] == ["Sana"] and entry["rounds_since_new_class"] == 0 and entry["effective"]
    assert label_all("Sana")["rounds_since_new_class"] == 1
    entry = label_all("Sana")
    assert entry["new_classes"] == [] and entry["rounds_since_new_class"] == 2
    assert engine.refresh()["rounds_since_new_class"] == 2           # an idle recomputation does not count as a round
    entry = label_all("Malata")
    assert entry["new_classes"] == ["Malata"] and entry["rounds_since_new_class"] == 0
    assert entry["classes"] == ["Malata", "Sana"]


def test_novelty_queries_that_bring_a_new_class(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    while not any(b.query_type == "novelty" for b in engine.batch):   # until steady state
        label_batch(engine, truth)
    novelty = [b.image_id for b in engine.batch if b.query_type == "novelty"]
    assert len(novelty) == 2
    batch_id = engine.batch_id
    while engine.batch_id == batch_id:
        item = engine.next_image("a@x.it")
        label = truth[item.image_id]
        if item.image_id == novelty[0]:
            label = "Mai Vista"
        elif item.image_id == novelty[1]:
            label = "non so"                                          # a "Scarta ape" does not count as labelled
        engine.submit("a@x.it", item.image_id, label)
    entry = engine.history[-1]
    assert entry["novelty_queries"] == 1 and entry["novelty_new_class"] == 1
    assert "Mai Vista" in entry["new_classes"]


def _round(i, f1=None, change=0.0, since=0, model_changed=False, effective=True):
    return {"round": i, "macro_f1": f1, "pred_change": change, "rounds_since_new_class": since,
            "model_changed": model_changed, "effective": effective, "model_version": "v000", "n_labels": 10 * i}


def test_stopping_status_needs_enough_rounds_then_decides(cfg, synth):
    engine = make_engine(cfg, synth)
    st = engine.stopping_status()
    assert st["window"] == 3 and st["all_ok"] is False
    assert [st[k]["ok"] for k in ("f1_plateau", "predictions_stable", "no_new_classes")] == [None, None, None]
    assert engine.status()["stopping"] == st

    engine.history = [_round(1, 0.50, None, 0), _round(2, 0.70, 0.20, 0), _round(3, 0.80, 0.05, 1),
                      _round(4, 0.85, 0.02, 2)]
    st = engine.stopping_status()
    assert st["f1_plateau"]["ok"] is False                           # still rising
    assert st["predictions_stable"]["ok"] is False                   # 20%, 5%, 2%: above 1%
    assert st["no_new_classes"]["ok"] is False and not st["all_ok"]

    engine.history += [_round(5, 0.852, 0.008, 3), _round(6, 0.85, 0.004, 4), _round(7, 0.855, 0.002, 5)]
    st = engine.stopping_status()
    assert st["f1_plateau"]["ok"] is True                            # +0.005 over the last 3 rounds: below 0.01
    assert st["predictions_stable"]["ok"] is True and st["no_new_classes"]["ok"] is True
    assert st["all_ok"] is True
    assert "0.855" in st["f1_plateau"]["detail"] and "0.20%" in st["predictions_stable"]["detail"]


def test_stopping_ignores_retrain_rounds_empty_rounds_and_rounds_without_metrics(cfg, synth):
    engine = make_engine(cfg, synth)
    base = [_round(1, 0.8, None), _round(2, 0.8, 0.005, 1), _round(3, 0.8, 0.005, 2), _round(4, 0.8, 0.005, 3)]
    engine.history = base + [_round(5, 0.8, 0.60, 4, model_changed=True)]   # after a re-train everything changes
    assert engine.stopping_status()["predictions_stable"]["ok"] is True      # that round is not comparable

    engine.history = base[:3] + [_round(4, None, 0.0, 2, effective=False), _round(5, None, 0.0, 2, effective=False)]
    st = engine.stopping_status()
    assert st["predictions_stable"]["ok"] is None                    # two idle recomputations do not make stability
    assert st["f1_plateau"]["ok"] is None

    engine.history = [_round(1, None, None), _round(2, 0.8, 0.0, 1), _round(3, None, 0.0, 2), _round(4, 0.8, 0.0, 3),
                      _round(5, 0.8, 0.0, 4), _round(6, 0.8, 0.0, 5)]
    assert engine.stopping_status()["f1_plateau"]["ok"] is True      # rounds without a metric are skipped


def test_old_round_rows_without_the_new_fields_do_not_break_anything(cfg, synth, truth, hub_app, monkeypatch, tmp_path):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    old = [{"timestamp": "2026-10-01T10:00:00", "model_version": "v000", "config_id": "x", "n_labels": 0,
            "n_classes": 0, "n_validation": 0, "accuracy": None, "macro_f1": None, "knn_accuracy": None,
            "knn_macro_f1": None, "round": i} for i in (1, 2)]
    (cfg.results_dir / "al_rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))

    engine = make_engine(cfg, synth)
    assert engine.round == 3
    entry = engine.history[-1]
    assert entry["pred_change"] is None and entry["new_classes"] == [] and entry["effective"] is False
    assert engine.stopping_status()["no_new_classes"]["ok"] is None
    label_batch(engine, truth)
    assert engine.history[-1]["new_classes"] and engine.history[-1]["effective"] is True

    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    r = c.get("/al-status")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Stopping criteria" in html and "not yet known" in html
    assert "Novelty" in html and "only an indication" in html


def test_status_page_marks_rounds_after_a_retrain(cfg, synth, hub_app, monkeypatch, tmp_path):
    engine = make_engine(cfg, synth)
    engine.set_embeddings(Embeddings(engine.bank.ids, engine.bank.paths, np.roll(engine.bank.X, 1, axis=1)), "v001")
    assert engine.history[-1]["model_changed"] is True
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    assert "new model" in c.get("/al-status").get_data(as_text=True)
