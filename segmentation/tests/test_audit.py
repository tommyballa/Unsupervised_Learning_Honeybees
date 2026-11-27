"""Audit mode: stratified sampling, blind labelling, isolation from the memory bank."""
import csv
import io
import json
import re
from collections import Counter

import numpy as np
import pytest

from bee_al import REMOVED_LABEL, UNKNOWN_LABEL
from bee_al import hub as al_hub
from bee_al.audit import (
    UNASSIGNED_STRATUM,
    allocation,
    assign_strata,
    confidence_bin,
    create_audit_set,
    load_audit_set,
    parse_scores,
)
from bee_al.config import AuditConfig, config_from_dict
from bee_al.engine import ALEngine, ALError
from bee_al.label_store import LabelStore, is_training_event
from bee_al.snapshot import PREDICTION_FIELDS, read_csv

from test_engine import Clock, label_batch
from test_snapshot import labeled_engine


def fake_snapshot(tmp_path, spec, name="snap"):
    """Fake snapshot: ``spec`` is a list of (how many bees, status, class, source, confidence)."""
    folder = tmp_path / "results" / "snapshots" / name
    folder.mkdir(parents=True)
    rows, i = [], 0
    for n, status, label, source, conf in spec:
        for _ in range(n):
            row = dict.fromkeys(PREDICTION_FIELDS, "")
            row.update(image_id=f"bee_{i:06d}.png", status=status, final_label=label, source=source,
                       confidence="" if conf is None else f"{conf:.6f}", dataset="d", session="d")
            rows.append(row)
            i += 1
    with open(folder / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return folder, rows


SPEC = [
    (6000, "predicted", "Sana", "predicted", 0.97),     # large stratum: large_stratum
    (300, "predicted", "Sana", "predicted", 0.60),      # same class, low confidence band
    (25, "predicted", "Malata", "predicted", 0.95),     # small stratum: taken in full
    (80, "predicted", "Malata", "predicted", 0.50),
    (120, "rejected", "Sfocata", "predicted", 0.80),    # predicted reject class: one stratum, no bands
    (70, "unassigned", "", "", None),
    (50, "labeled", "Sana", "human", 1.0),              # everything else is outside the population
    (10, "rejected", "Sfocata", "human", 1.0),
    (20, "validation", "", "", None),
    (5, "unknown", "", "", None),
    (3, "no_prediction", "", "", None),
]


def test_config_defaults_and_checks():
    a = config_from_dict({}).audit
    assert (a.per_stratum, a.large_stratum, a.large_threshold, a.confidence_bins, a.double_fraction, a.seed) == (40, 250, 5000, [0.9], 0.1, 7)
    for bad in ({"per_stratum": 0}, {"large_stratum": 10}, {"confidence_bins": [1.2]}, {"confidence_bins": [0.5, 0.5]},
                {"double_fraction": 2}, {"anomaly_quantile": 1.0}):
        with pytest.raises(ValueError):
            config_from_dict({"audit": bad})


def test_confidence_bins_and_allocation():
    assert [confidence_bin(c, [0.9]) for c in (0.1, 0.8999, 0.9, 1.0)] == ["<0.9", "<0.9", ">=0.9", ">=0.9"]
    assert [confidence_bin(c, [0.9, 0.5]) for c in (0.2, 0.5, 0.95)] == ["<0.5", "0.5-0.9", ">=0.9"]
    assert confidence_bin(0.3, []) == ""
    audit = AuditConfig()
    assert [allocation(n, audit) for n in (0, 7, 40, 41, 4999, 5000, 80000)] == [0, 7, 40, 40, 40, 250, 250]
    assert allocation(100, AuditConfig(per_stratum=40, large_stratum=250, large_threshold=50)) == 100


def test_strata_and_allocations(tmp_path):
    cfg = config_from_dict({}, root=tmp_path)
    folder, rows = fake_snapshot(tmp_path, SPEC)
    path = create_audit_set(folder, cfg)
    audit = load_audit_set(folder)
    assert path == folder / "audit_set.json" and audit["snapshot"] == "snap" and audit["seed"] == 7
    strata = {s["id"]: s for s in audit["strata"]}
    assert {sid: (s["N_h"], s["n_h"]) for sid, s in strata.items()} == {
        "Sana|>=0.9": (6000, 250), "Sana|<0.9": (300, 40), "Malata|>=0.9": (25, 25), "Malata|<0.9": (80, 40),
        "Sfocata|reject": (120, 40), UNASSIGNED_STRATUM: (70, 40),
    }
    assert strata["Sfocata|reject"]["kind"] == "reject" and strata["Sfocata|reject"]["conf_bin"] == ""
    assert strata[UNASSIGNED_STRATUM]["kind"] == "unassigned" and strata["Sana|<0.9"]["label"] == "Sana"
    assert audit["population"] == 6595 and len(audit["bees"]) == 435

    by_id = {r["image_id"]: r for r in rows}
    sampled = [b["image_id"] for b in audit["bees"]]
    assert len(set(sampled)) == 435 and sorted(audit["order"]) == sorted(sampled) and audit["order"] != sampled
    # exclusions: no hand-labelled, validation, "Scarta ape" or unpredicted bees
    assert {by_id[i]["status"] for i in sampled} == {"predicted", "rejected", "unassigned"}
    assert not any(by_id[i]["source"] == "human" for i in sampled)
    for b in audit["bees"]:                                           # every bee is in the right stratum
        r = by_id[b["image_id"]]
        expected = UNASSIGNED_STRATUM if r["status"] == "unassigned" else (
            "Sfocata|reject" if r["status"] == "rejected" else f"{r['final_label']}|{'>=0.9' if float(r['confidence']) >= 0.9 else '<0.9'}")
        assert b["stratum"] == expected

    # double labelling: about 10%, at least one per stratum
    double = Counter(b["stratum"] for b in audit["bees"] if b["double"])
    assert double == {"Sana|>=0.9": 25, "Sana|<0.9": 4, "Malata|>=0.9": 2, "Malata|<0.9": 4, "Sfocata|reject": 4, UNASSIGNED_STRATUM: 4}

    # the stratum of every bee in the population is saved for the estimates
    strata_csv = read_csv(folder / "audit_strata.csv")
    assert len(strata_csv) == 6595 and Counter(r["stratum"] for r in strata_csv)["Sana|>=0.9"] == 6000


def test_same_seed_same_sample_and_created_only_once(tmp_path):
    cfg = config_from_dict({}, root=tmp_path)
    a, _ = fake_snapshot(tmp_path, SPEC, "a")
    b, _ = fake_snapshot(tmp_path, SPEC, "b")
    create_audit_set(a, cfg)
    create_audit_set(b, cfg)
    first, second = load_audit_set(a), load_audit_set(b)
    assert first["bees"] == second["bees"] and first["order"] == second["order"]
    with pytest.raises(FileExistsError, match="only once"):
        create_audit_set(a, cfg)

    cfg.audit.seed = 8
    c, _ = fake_snapshot(tmp_path, SPEC, "c")
    create_audit_set(c, cfg)
    assert load_audit_set(c)["order"] != first["order"]
    empty, _ = fake_snapshot(tmp_path, [(5, "labeled", "Sana", "human", 1.0)], "vuoto")
    with pytest.raises(ValueError, match="no predicted bees"):
        create_audit_set(empty, cfg)


def test_anomaly_scores_split_only_the_chosen_class(tmp_path):
    cfg = config_from_dict({"audit": {"anomaly_split_class": "  sana ", "anomaly_quantile": 0.9}}, root=tmp_path)
    folder, rows = fake_snapshot(tmp_path, SPEC)
    sana = [r["image_id"] for r in rows if r["status"] == "predicted" and r["final_label"] == "Sana"]
    scores = {image_id: float(i) for i, image_id in enumerate(sana[:-10])}     # 10 bees without a score: they go below
    create_audit_set(folder, cfg, scores)
    audit = load_audit_set(folder)
    strata = {s["id"]: s for s in audit["strata"]}
    assert {"Sana|>=0.9|anomaly_high", "Sana|>=0.9|anomaly_low", "Sana|<0.9|anomaly_high", "Sana|<0.9|anomaly_low"} <= set(strata)
    assert "Malata|>=0.9" in strata and not any("anomaly" in s for s in strata if s.startswith("Malata"))
    high = sum(s["N_h"] for sid, s in strata.items() if sid.endswith("anomaly_high"))
    assert high == pytest.approx(0.1 * (len(sana) - 10), abs=2)
    assert strata["Sana|>=0.9|anomaly_high"]["anomaly_bin"] == "high" and audit["anomaly_scores"] is True
    assert parse_scores((folder / "anomaly_scores.csv").read_text()) == scores

    of, plain = assign_strata(rows, cfg.audit, "", scores)                      # no class given: no split
    assert not any("anomaly" in s for s in plain)
    for text in ("a,b\n1,2\n", "image_id,score\nx,abc\n"):
        with pytest.raises(ValueError):
            parse_scores(text)


# ------------------------------- engine ------------------------------------ #

def audit_engine(colony, per_stratum=6, double_fraction=0.34, **kw):
    colony.cfg.audit.per_stratum = per_stratum
    colony.cfg.audit.double_fraction = double_fraction
    engine = labeled_engine(colony, **kw)
    engine.export_snapshot("s1", "a@x.it")
    engine.create_audit("s1")
    return engine


def label_audit(engine, truth, annotator, limit=10_000):
    done = []
    while len(done) < limit:
        item = engine.next_image(annotator, mode="audit")
        if item is None:
            break
        engine.submit(annotator, item.image_id, truth[item.image_id], mode="audit")
        done.append(item.image_id)
    return done


def test_audit_labels_never_enter_the_bank_or_the_validation_metrics(colony):
    engine = audit_engine(colony)
    progress = engine.audit_progress()
    assert progress["snapshot"] == "s1" and progress["done"] == 0 and progress["total"] == len(engine.audit["bees"])
    before = (engine.bank.n_labeled, engine.bank.classes, engine.bank.class_counts())
    probs_before = engine.final_pred.probs.copy()
    events_before = len(engine.store.events())

    wrong = {i: "Classe Solo Audit" for i in colony.truth}             # labels the model must never see
    done = label_audit(engine, wrong, "a@x.it")
    assert len(done) == progress["total"]
    events = engine.store.events()[events_before:]
    assert {e.query_type for e in events} == {"audit"} and {e.batch_id for e in events} == {"audit:s1"}
    assert not any(is_training_event(e) for e in events)

    def unchanged():
        assert (engine.bank.n_labeled, engine.bank.classes, engine.bank.class_counts()) == before
        assert all(engine.bank.label_of(i) is None for i in done)
        np.testing.assert_array_equal(engine.final_pred.probs, probs_before)
        assert "Classe Solo Audit" not in engine.quick_labels() and "Classe Solo Audit" not in engine.final_clusters()
        assert engine.val_labels == {} and engine.history[-1]["n_validation"] == 0

    unchanged()
    entry = engine.refresh()                                           # not even after a full recomputation
    unchanged()
    assert entry["n_labels"] == before[0] and entry["effective"] is False

    restarted = ALEngine(colony.cfg, colony.emb)                       # nor after a restart
    assert restarted.bank.class_counts() == before[2] and "Classe Solo Audit" not in restarted.bank.classes
    assert restarted.audit_progress()["done"] == sum(1 for b in restarted.audit["bees"] if not b["double"])


def test_audit_bees_leave_normal_labeling(colony):
    engine = audit_engine(colony)
    audit_ids = {b["image_id"] for b in engine.audit["bees"]}
    assert not audit_ids & {b.image_id for b in engine.batch}
    seen = []
    for _ in range(4):
        seen += label_batch(engine, colony.truth)
    assert seen and not audit_ids & set(seen)                          # never in later batches

    one = sorted(audit_ids)[0]
    n_events = len(engine.store.events())
    with pytest.raises(ALError, match="audit"):
        engine.assign_many("a@x.it", [one], "Sana")
    with pytest.raises(ALError, match="audit"):
        engine.submit("a@x.it", one, "Sana", manual=True)
    with pytest.raises(ALError):
        engine.submit("a@x.it", seen[0], "Sana", mode="audit")         # and a normal bee does not enter the audit
    assert len(engine.store.events()) == n_events
    shown = {b["image_id"] for c in engine.cluster_samples(n=500) for b in c["samples"]}
    shown |= {b["image_id"] for b in engine.unassigned_samples(n=500)["samples"]}
    assert not audit_ids & shown                                       # and they do not appear on the clusters page


def test_double_bees_need_two_different_annotators(colony):
    clock = Clock()
    colony.cfg.audit.per_stratum, colony.cfg.audit.double_fraction = 6, 0.34
    engine = ALEngine(colony.cfg, colony.emb, clock=clock)
    for _ in range(3):
        label_batch(engine, colony.truth)
    engine.export_snapshot("s1", "a@x.it")
    engine.create_audit("s1")
    bees = {b["image_id"]: b for b in engine.audit["bees"]}
    doubles = {i for i, b in bees.items() if b["double"]}
    assert 0 < len(doubles) < len(bees)

    first = label_audit(engine, colony.truth, "a@x.it")
    assert sorted(first) == sorted(bees) and len(set(first)) == len(first)   # each one only once per annotator
    assert first == engine.audit["order"]                                    # in the saved order, strata mixed
    assert engine.next_image("a@x.it", mode="audit") is None                 # double-labelled bees do not go back to the same annotator
    progress = engine.audit_progress()
    assert progress["done"] == len(bees) - len(doubles) and progress["annotators"] == {"a@x.it": len(bees)}

    b_item = engine.next_image("b@x.it", mode="audit")
    assert b_item.image_id in doubles and b_item.query_type == "audit"
    assert engine.next_image("b@x.it", mode="audit").image_id == b_item.image_id   # reloading does not change the bee
    c_item = engine.next_image("c@x.it", mode="audit")
    assert c_item.image_id != b_item.image_id                                # only one slot left: goes to another bee
    clock.t += colony.cfg.hub.assignment_timeout_s + 1
    assert engine.next_image("c@x.it", mode="audit").image_id == b_item.image_id   # assignment expired

    engine.submit("c@x.it", b_item.image_id, colony.truth[b_item.image_id], mode="audit")

    second = label_audit(engine, colony.truth, "b@x.it")                    # b gets the remaining double-labelled bees
    assert sorted(second) == sorted(doubles - {b_item.image_id})
    assert engine.audit_progress()["done"] == len(bees) and engine.next_image("c@x.it", mode="audit") is None
    assert all(list(engine.audit_labels[i]) == ["a@x.it", "c@x.it" if i == b_item.image_id else "b@x.it"] for i in doubles)
    assert all(s["done"] == s["n_h"] for s in engine.audit_progress()["strata"])


def test_unknown_removal_and_per_annotator_answers(colony):
    engine = audit_engine(colony)
    a = engine.next_image("a@x.it", mode="audit").image_id
    engine.submit("a@x.it", a, "non so", mode="audit")
    assert engine.audit_labels[a] == {"a@x.it": UNKNOWN_LABEL} and engine.bank.unknown_indices().size == 0
    engine.submit("a@x.it", a, "Sana", mode="audit")                    # correction: that annotator's last answer counts
    engine.submit("b@x.it", a, "Malata", mode="audit")
    assert engine.audit_labels[a] == {"a@x.it": "Sana", "b@x.it": "Malata"}
    engine.submit("a@x.it", a, REMOVED_LABEL, mode="audit")
    assert engine.audit_labels[a] == {"b@x.it": "Malata"} and engine.audit_label_of(a, "a@x.it") is None

    store = LabelStore(colony.cfg.al_labels_path)
    answers = store.per_annotator(lambda e: e.query_type == "audit")
    assert {k: {n: e.label for n, e in v.items()} for k, v in answers.items()} == {a: {"b@x.it": "Malata"}}
    assert a not in store.current(is_training_event)


def test_restart_rebuilds_the_audit_state_from_the_store(colony):
    engine = audit_engine(colony)
    label_audit(engine, colony.truth, "a@x.it", limit=7)
    before = engine.audit_progress()
    assert json.loads((colony.cfg.results_dir / "active_audit.json").read_text()) == {"snapshot": "s1"}

    restarted = ALEngine(colony.cfg, colony.emb)
    assert restarted.audit_progress() == before and restarted.audit_labels == engine.audit_labels
    audit_ids = {b["image_id"] for b in restarted.audit["bees"]}
    assert not audit_ids & {b.image_id for b in restarted.batch}
    nxt = restarted.next_image("a@x.it", mode="audit")
    assert nxt.image_id == engine.audit["order"][7]

    # without the active-audit file, the latest snapshot with an audit set is resumed
    (colony.cfg.results_dir / "active_audit.json").unlink()
    assert ALEngine(colony.cfg, colony.emb).audit_progress()["snapshot"] == "s1"


def test_switching_audits_and_errors(colony):
    engine = audit_engine(colony)
    with pytest.raises(ALError, match="only once"):
        engine.create_audit("s1")
    with pytest.raises(ALError, match="does not exist"):
        engine.create_audit("mai_esistito")
    with pytest.raises(ALError):
        engine.load_audit("../x")
    label_batch(engine, colony.truth)
    engine.export_snapshot("s2", "a@x.it")
    with pytest.raises(ALError, match="no audit set"):
        engine.load_audit("s2")
    assert engine.audit_progress()["snapshot"] == "s1"
    label_audit(engine, colony.truth, "a@x.it", limit=3)
    engine.create_audit("s2")
    assert engine.audit_progress()["snapshot"] == "s2" and engine.audit_progress()["done"] == 0
    assert engine.load_audit("s1")["done"] >= 1                         # the answers of the first audit are still there
    assert {s["name"]: s["has_audit"] for s in engine.list_snapshots()} == {"s1": True, "s2": True}


def test_no_audit_means_no_audit_mode(colony):
    engine = labeled_engine(colony, rounds=1)
    assert engine.audit_progress() is None and engine.status()["audit"] is None
    assert engine.next_image("a@x.it", mode="audit") is None
    with pytest.raises(ALError):
        engine.submit("a@x.it", engine.bank.ids[0], "Sana", mode="audit")


# --------------------------------- hub -------------------------------------- #

@pytest.fixture
def hub(colony, hub_app, monkeypatch, tmp_path):
    colony.cfg.audit.per_stratum, colony.cfg.audit.double_fraction = 6, 0.34
    colony.cfg.hub.show_suggestion = True                              # must not appear in audit mode anyway
    engine = labeled_engine(colony)
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    c.engine = engine
    return c


def test_hub_creates_activates_and_labels_an_audit(hub, colony):
    engine = hub.engine
    assert "No audit yet" in hub.get("/al-labeling?mode=audit").get_data(as_text=True)
    hub.post("/al-snapshot", data={"name": "s1"})
    assert "Create audit set" in hub.get("/al-status").get_data(as_text=True)

    r = hub.post("/al-audit-create", data={"snapshot": "s1"})
    html = hub.get(r.headers["Location"]).get_data(as_text=True)
    assert "is now the active audit" in html and "Audit of snapshot s1" in html and ">active</b>" in html
    total = engine.audit_progress()["total"]

    page = hub.get("/al-labeling?mode=audit").get_data(as_text=True)
    image_id = re.search(r'name="image_id" value="([^"]+)"', page).group(1)
    assert image_id == engine.audit["order"][0] and "Audit Labeling" in page and f"Audit: 0 / {total} labeled" in page
    assert "Scarta ape" in page and 'href="/al-labeling?mode=audit"' in page
    # blind: no suggestion, no predicted class, no stratum
    stratum = next(b["stratum"] for b in engine.audit["bees"] if b["image_id"] == image_id)
    assert "kNN suggestion" not in page and stratum not in page and "stratum" not in page.lower()
    assert "kNN suggestion" in hub.get("/al-labeling").get_data(as_text=True)      # in normal mode it is there

    hub.post("/al-labeling?mode=audit", data={"image_id": image_id, "label": "sana", "action": "label"})
    assert engine.audit_labels[image_id] == {"t@x.it": "Sana"} and engine.bank.label_of(image_id) is None
    assert engine.store.events()[-1].query_type == "audit"
    nxt = hub.get("/al-labeling?mode=audit").get_data(as_text=True)
    assert re.search(r'name="image_id" value="([^"]+)"', nxt).group(1) == engine.audit["order"][1]
    hub.post("/al-labeling?mode=audit", data={"image_id": engine.audit["order"][1], "action": "unknown"})
    assert engine.audit_labels[engine.audit["order"][1]] == {"t@x.it": UNKNOWN_LABEL}
    redo = hub.get("/al-labeling?mode=audit&redo=1").get_data(as_text=True)
    assert "Fixing your previous label" in redo and "Scarta ape" in redo

    bad = hub.post("/al-labeling", data={"image_id": image_id, "label": "Sana", "action": "label"})
    assert bad.status_code == 200 and "audit" in bad.get_data(as_text=True)

    for data, text in (({"snapshot": "s1"}, "only once"), ({"snapshot": "boh"}, "does not exist")):
        r = hub.post("/al-audit-create", data=data)
        assert r.status_code == 302 and "Audit set not created" in hub.get(r.headers["Location"]).get_data(as_text=True)
    r = hub.post("/al-audit-activate", data={"snapshot": "boh"})
    assert "Audit not activated" in hub.get(r.headers["Location"]).get_data(as_text=True)


def test_hub_audit_with_uploaded_scores_and_activation(hub, colony):
    engine = hub.engine
    hub.post("/al-snapshot", data={"name": "s1"})
    hub.post("/al-snapshot", data={"name": "s2"})
    scores = "image_id,score\n" + "".join(f"{i},{k}\n" for k, i in enumerate(engine.bank.ids))
    r = hub.post("/al-audit-create", data={"snapshot": "s1", "scores": (io.BytesIO(scores.encode()), "scores.csv")},
                 content_type="multipart/form-data")
    assert "is now the active audit" in hub.get(r.headers["Location"]).get_data(as_text=True)
    assert (colony.cfg.results_dir / "snapshots" / "s1" / "anomaly_scores.csv").exists()
    r = hub.post("/al-audit-create", data={"snapshot": "s2", "scores": (io.BytesIO(b"nope"), "scores.csv")},
                 content_type="multipart/form-data")
    assert "Audit set not created" in hub.get(r.headers["Location"]).get_data(as_text=True)

    hub.post("/al-audit-create", data={"snapshot": "s2"})
    assert engine.audit_progress()["snapshot"] == "s2"
    assert "Make this audit active" in hub.get("/al-status").get_data(as_text=True)
    r = hub.post("/al-audit-activate", data={"snapshot": "s1"})
    assert "now active" in hub.get(r.headers["Location"]).get_data(as_text=True) and engine.audit_progress()["snapshot"] == "s1"
    assert {"/al-audit-create", "/al-audit-activate"} <= {r.rule for r in hub.application.url_map.iter_rules()}
