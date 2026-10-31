"""Coda di revisione delle etichette sospette: api etichettate a mano che i vicini smentiscono."""
import csv
import json
import re

import numpy as np
import pytest

from bee_al import UNKNOWN_LABEL
from bee_al import hub as al_hub
from bee_al.config import config_from_dict
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine, ALError
from bee_al.knn import knn_predict
from bee_al.review import confirmations, find_suspects, read_log
from bee_al.simulation import make_synthetic

N_LABELED = 120


@pytest.fixture
def data():
    """300 api in 4 cluster ben separati."""
    X, y = make_synthetic(n=300, dim=32, n_classes=4, imbalance=0.8, noise=0.3, seed=5)
    return X, np.asarray(y), [f"bee_{i:05d}_conf0.90.png" for i in range(len(X))]


def build(cfg, data, wrong=None, confidence=None, n=N_LABELED):
    """Engine con ``n`` api etichettate a mano; ``wrong`` = {image_id: etichetta sbagliata}."""
    X, y, ids = data
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    wrong, confidence = wrong or {}, confidence or {}
    chosen = [i for i in ids if i not in engine.validation_ids][:n]
    for image_id in chosen:    # scritte nello store come etichette manuali, poi un solo ricalcolo
        label = wrong.get(image_id, y[ids.index(image_id)])
        engine.store.append(image_id, label, "a@x.it", confidence.get(image_id, "High"), "manual", cfg.config_id, "v000", "manual")
    if chosen:
        engine.refresh()
    engine.truth = dict(zip(ids, y))
    engine.labeled = chosen
    return engine


def other_class(engine, image_id):
    return next(c for c in sorted(set(engine.truth.values())) if c != engine.truth[image_id])


def flipped(cfg, data, k=8, **kw):
    """Engine con ``k`` etichette invertite apposta (rumore noto)."""
    probe = build(cfg, data, n=0)
    pool = [i for i in data[2] if i not in probe.validation_ids][:N_LABELED]
    noisy = pool[5::14][:k]
    wrong = {i: other_class(probe, i) for i in noisy}
    engine = build(cfg, data, wrong=wrong, **kw)
    return engine, noisy


def test_config_defaults_and_checks():
    r = config_from_dict({}).review
    assert (r.enabled, r.k, r.min_disagreement, r.min_class_size, r.show_neighbors) == (True, 0, 0.8, 5, 6)
    for bad in ({"min_disagreement": 0}, {"min_disagreement": 1.1}, {"min_class_size": 0}, {"k": -1}, {"show_neighbors": 0}):
        with pytest.raises(ValueError):
            config_from_dict({"review": bad})


def test_correct_labels_of_separated_clusters_raise_no_suspects(cfg, data):
    engine = build(cfg, data)
    queue = engine.suspects()
    assert queue == {"main": [], "rare": [], "n_checked": N_LABELED, "n_confirmed_hidden": 0}
    assert engine.status()["n_suspects"] == 0 and engine.history[-1]["n_suspects"] == 0


def test_injected_label_noise_is_found_with_the_true_class_suggested(cfg, data):
    engine, noisy = flipped(cfg, data, k=8)
    main = engine.suspects()["main"]
    found = {s["image_id"]: s for s in main}
    assert set(noisy) <= set(found)                                   # recall piena sul rumore iniettato
    assert len(found) - len(noisy) <= 1                               # e quasi nessun falso allarme
    for image_id in noisy:
        s = found[image_id]
        assert s["suggested"] == engine.truth[image_id] and s["label"] == other_class(engine, image_id)
        assert s["p_suggested"] >= 0.8 and s["p_label"] <= 0.2 and not s["rare_class"]
        assert s["annotator"] == "a@x.it" and s["confidence"] == "High" and s["query_type"] == "manual" and s["timestamp"]
        assert len(s["neighbors"]) == 6 and all(n["image_id"] != image_id for n in s["neighbors"])
        sims = [n["similarity"] for n in s["neighbors"]]
        assert sims == sorted(sims, reverse=True) and sum(n["label"] == s["suggested"] for n in s["neighbors"]) >= 5
    # stessa probabilità del k-NN dell'hub (che già esclude l'ape stessa)
    one = found[noisy[0]]
    pred = knn_predict(engine.bank, [one["index"]], cfg.knn.k, cfg.knn.temperature)
    assert one["p_suggested"] == pytest.approx(pred.probs[0, pred.classes.index(one["suggested"])])
    assert find_suspects(engine.bank, engine.store, cfg, engine.validation_ids) == engine._suspects   # stessi input, stessi output


def test_a_bee_does_not_vote_for_itself(cfg, data):
    cfg.review.k, cfg.review.min_class_size = 1, 1
    probe = build(cfg, data, n=0)
    lonely = [i for i in data[2] if i not in probe.validation_ids][3]
    engine = build(cfg, data, wrong={lonely: "Sola"})                 # unica ape della classe "Sola", dentro un altro cluster
    s = next(x for x in engine.suspects()["main"] if x["image_id"] == lonely)
    # con k = 1, se votasse per sé il vicino più vicino sarebbe lei stessa e non risulterebbe sospetta
    assert s["image_id"] == lonely and s["label"] == "Sola" and s["suggested"] == engine.truth[lonely]
    assert s["p_suggested"] == 1.0 and s["p_label"] == 0.0


def test_low_confidence_labels_come_first(cfg, data):
    probe = build(cfg, data, n=0)
    pool = [i for i in data[2] if i not in probe.validation_ids][:N_LABELED]
    a, b, c = pool[10], pool[40], pool[70]
    wrong = {i: other_class(probe, i) for i in (a, b, c)}
    engine = build(cfg, data, wrong=wrong, confidence={a: "High", b: "Low", c: "Medium"})
    main = engine.suspects()["main"]
    order = [s["image_id"] for s in main]
    assert order.index(b) < order.index(c) < order.index(a)           # Low, poi Medium, poi High
    assert {s["p_suggested"] for s in main if s["image_id"] in (a, b, c)} == {1.0}   # a parità di disaccordo


def test_rare_classes_are_kept_apart(cfg, data):
    probe = build(cfg, data, n=0)
    pool = [i for i in data[2] if i not in probe.validation_ids][:N_LABELED]
    rare = pool[20:23]                                                 # 3 etichette di una classe nuova: meno di min_class_size
    engine = build(cfg, data, wrong={i: "Fenotipo Raro" for i in rare})
    queue = engine.suspects()
    assert queue["main"] == [] and {s["image_id"] for s in queue["rare"]} == set(rare)
    assert all(s["rare_class"] and s["label"] == "Fenotipo Raro" for s in queue["rare"])
    assert engine.status()["n_suspects"] == 0 and engine.review_stats()["n_rare"] == 3

    cfg.review.min_class_size = 2                                      # con una soglia più bassa tornano nella coda principale
    engine.recompute()
    assert len(engine.suspects()["main"]) == 3 and engine.suspects()["rare"] == []


def test_validation_audit_and_set_aside_bees_are_never_reviewed(cfg, data):
    engine, noisy = flipped(cfg, data)
    val = engine.validation_ids[0]
    engine.submit("a@x.it", val, other_class(engine, val), mode="validation")     # etichetta di validazione sbagliata apposta
    aside = engine.labeled[0]
    engine.assign_many("a@x.it", [aside], UNKNOWN_LABEL)
    engine.export_snapshot("s", "a@x.it")
    engine.create_audit("s")
    audit_bee = engine.audit["order"][0]
    engine.submit("b@x.it", audit_bee, "Classe Sbagliata", mode="audit")
    engine.refresh()

    queue = engine.suspects()
    shown = {s["image_id"] for s in queue["main"] + queue["rare"]}
    assert set(noisy) <= shown and not shown & ({val, aside, audit_bee} | set(engine.validation_ids))
    assert queue["n_checked"] == engine.bank.n_labeled == N_LABELED - 1
    for image_id, text in ((val, "validation"), (audit_bee, "audit"), (aside, "sospette"), ("non_esiste.png", "sconosciuta")):
        with pytest.raises(ALError, match=text):
            engine.review("a@x.it", image_id, "confirmed")
    with pytest.raises(ALError, match="decisione"):
        engine.review("a@x.it", noisy[0], "boh")
    assert read_log(engine.review_log_path) == []


def test_confirm_hides_the_bee_without_touching_the_labels(cfg, data):
    engine, noisy = flipped(cfg, data)
    target = noisy[0]
    events_before = len(engine.store.events())
    before = engine.suspects()
    suggested = next(s["suggested"] for s in before["main"] if s["image_id"] == target)

    engine.review("r@x.it", target, "confirmed")
    after = engine.suspects()
    assert target not in {s["image_id"] for s in after["main"]} and after["n_confirmed_hidden"] == 1
    assert len(after["main"]) == len(before["main"]) - 1
    assert len(engine.store.events()) == events_before                # nessun evento nuovo tra le etichette
    assert engine.bank.label_of(target) == other_class(engine, target)
    (row,) = read_log(engine.review_log_path)
    assert row["decision"] == "confirmed" and row["annotator"] == "r@x.it" and row["image_id"] == target
    assert row["old_label"] == row["new_label"] == other_class(engine, target) and row["suggested"] == suggested
    assert row["annotator_confidence"] == "High" and row["model_version"] == "v000" and row["rare_class"] is False
    assert set(row) == {"timestamp", "annotator", "image_id", "decision", "old_label", "new_label", "suggested",
                        "p_suggested", "annotator_confidence", "model_version", "round", "rare_class"}

    engine.refresh()                                                   # un ricalcolo non la fa tornare
    assert target not in {s["image_id"] for s in engine.suspects()["main"]}

    # se cambia la classe suggerita (i vicini vengono rietichettati) può ricomparire
    neighbours = [i for i in engine.labeled if engine.truth[i] == suggested and i != target and engine.bank.label_of(i) == suggested]
    engine.assign_many("a@x.it", neighbours, "Classe Rinominata")
    again = {s["image_id"]: s for s in engine.suspects()["main"]}
    assert target in again and again[target]["suggested"] == "Classe Rinominata"
    assert engine.suspects()["n_confirmed_hidden"] == 0


def test_correct_and_set_aside_go_through_the_label_store(cfg, data):
    engine, noisy = flipped(cfg, data)
    a, b, c = noisy[:3]
    events_before = len(engine.store.events())
    n_before = len(engine.suspects()["main"])

    engine.review("r@x.it", a, "corrected", engine.truth[a])           # la classe suggerita
    assert engine.bank.label_of(a) == engine.truth[a]
    last = engine.store.events()[-1]
    assert (last.image_id, last.label, last.annotator, last.query_type) == (a, engine.truth[a], "r@x.it", "manual")
    assert a not in {s["image_id"] for s in engine.suspects()["main"]} and len(engine.suspects()["main"]) == n_before - 1

    engine.review("r@x.it", b, "corrected", "  tutt'altra classe ")    # o qualsiasi altra classe
    assert engine.bank.label_of(b) == "Tutt'Altra Classe" and "Tutt'Altra Classe" in engine.final_pred.classes

    engine.review("r@x.it", c, "set_aside")
    assert engine.bank.label_of(c) == UNKNOWN_LABEL and engine.store.events()[-1].label == UNKNOWN_LABEL
    assert len(engine.store.events()) == events_before + 3

    log = read_log(engine.review_log_path)
    assert [(r["image_id"], r["decision"], r["new_label"]) for r in log] == [
        (a, "corrected", engine.truth[a]), (b, "corrected", "Tutt'Altra Classe"), (c, "set_aside", UNKNOWN_LABEL)]
    assert all(r["old_label"] == other_class(engine, r["image_id"]) for r in log)

    d = noisy[3]
    for label, text in (("", "vuota"), (other_class(engine, d), "già")):   # correzioni senza senso: niente di scritto
        with pytest.raises(ALError, match=text):
            engine.review("r@x.it", d, "corrected", label)
    assert len(read_log(engine.review_log_path)) == 3 and len(engine.store.events()) == events_before + 3
    with pytest.raises(ALError, match="sospette"):
        engine.review("r@x.it", a, "confirmed")                        # corretta: non è più in coda


def test_restart_rebuilds_confirmations_from_the_log(cfg, data):
    engine, noisy = flipped(cfg, data)
    engine.review("r@x.it", noisy[0], "confirmed")
    engine.review("r@x.it", noisy[1], "confirmed")
    engine.review("r@x.it", noisy[1], "corrected", engine.truth[noisy[1]])   # ripensamento: la conferma non vale più
    X, y, ids = data
    restarted = ALEngine(cfg, Embeddings(ids, list(ids), X))
    assert restarted._confirmed == engine._confirmed == {noisy[0]: (other_class(engine, noisy[0]), engine.truth[noisy[0]])}
    assert restarted.suspects() == engine.suspects() and restarted.suspects()["n_confirmed_hidden"] == 1
    assert confirmations(read_log(restarted.review_log_path)) == restarted._confirmed


def test_review_stats_match_the_decisions(cfg, data):
    probe = build(cfg, data, n=0)
    pool = [i for i in data[2] if i not in probe.validation_ids][:N_LABELED]
    noisy = pool[5::14][:8]
    engine = build(cfg, data, wrong={i: other_class(probe, i) for i in noisy},
                   confidence={**{i: "Low" for i in noisy[:4]}, **{i: "Medium" for i in pool[:10] if i not in noisy}})
    stats = engine.review_stats()
    assert stats["n_checked"] == N_LABELED and stats["n_suspects"] >= 8 and stats["n_reviewed"] == 0
    assert stats["decisions"] == {"confirmed": 0, "corrected": 0, "set_aside": 0} and stats["corrections"] == {}
    low = stats["suspects_by_confidence"]["Low"]
    assert low == {"labels": 4, "suspects": 4, "share": 1.0}            # le quattro "Low" erano tutte sbagliate
    assert stats["suspects_by_confidence"]["Medium"]["suspects"] == 0
    assert stats["suspects_by_confidence"]["High"]["labels"] == N_LABELED - 4 - stats["suspects_by_confidence"]["Medium"]["labels"]

    engine.review("r@x.it", noisy[0], "confirmed")
    engine.review("r@x.it", noisy[1], "corrected", engine.truth[noisy[1]])
    engine.review("r@x.it", noisy[4], "corrected", engine.truth[noisy[4]])
    engine.review("r@x.it", noisy[5], "set_aside")
    stats = engine.review_stats()
    assert stats["decisions"] == {"confirmed": 1, "corrected": 2, "set_aside": 1} and stats["n_reviewed"] == 4
    assert stats["n_confirmed_hidden"] == 1 and stats["n_checked"] == N_LABELED - 1
    assert sum(stats["corrections"].values()) == 2
    assert all(" -> " in pair for pair in stats["corrections"])
    assert stats["decisions_by_confidence"]["Low"] == {"confirmed": 1, "corrected": 1, "set_aside": 0}
    assert stats["decisions_by_confidence"]["High"] == {"confirmed": 0, "corrected": 1, "set_aside": 1}
    assert engine.status()["review"] == stats and engine.status()["n_suspects"] == stats["n_suspects"]


def test_rounds_record_the_number_of_suspects_and_old_rows_still_load(cfg, data):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "al_rounds.jsonl").write_text(json.dumps({"round": 1, "n_labels": 0, "model_version": "v000"}) + "\n")
    engine, noisy = flipped(cfg, data)
    assert engine.history[0].get("n_suspects") is None                 # riga vecchia, senza il campo
    entry = engine.refresh()
    assert entry["n_suspects"] == len(engine.suspects()["main"]) >= 8
    assert json.loads((cfg.results_dir / "al_rounds.jsonl").read_text().splitlines()[-1])["n_suspects"] == entry["n_suspects"]


def test_review_can_be_switched_off(cfg, data):
    cfg.review.enabled = False
    engine, noisy = flipped(cfg, data)
    assert engine.suspects() == {"main": [], "rare": [], "n_checked": 0, "n_confirmed_hidden": 0}
    assert engine.review_stats()["enabled"] is False
    with pytest.raises(ALError):
        engine.review("r@x.it", noisy[0], "confirmed")


# --------------------------------- hub -------------------------------------- #

@pytest.fixture
def client(cfg, data, hub_app, monkeypatch, tmp_path):
    engine, noisy = flipped(cfg, data, k=5)
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    c.engine, c.noisy = engine, noisy
    return c


def _shown(html):
    m = re.search(r'name="image_id" value="([^"]+)"', html)
    return m.group(1) if m else None


def test_review_page_shows_the_evidence(client):
    engine = client.engine
    n = len(engine.suspects()["main"])
    html = client.get("/al-review").get_data(as_text=True)
    first = engine.suspects()["main"][0]
    assert _shown(html) == first["image_id"] and f"{n} suspects left, 0 reviewed" in html
    assert f'Keep "{first["label"]}"' in html.replace("&#34;", '"') and f'Change to "{first["suggested"]}"' in html.replace("&#34;", '"')
    assert "Scarta ape" in html and "Neighbours suggest" in html and "Declared confidence" in html and "a@x.it" in html
    assert html.count("/al-image/") == 1 + len(first["neighbors"]) and "sim 0." in html
    for url in ("/al-status", "/al-clusters", "/al-labeling"):          # la voce di menu mostra il conteggio
        assert f"Review ({n})</a>" in client.get(url).get_data(as_text=True)
    assert f"Review ({n})</a>" in client.get("/al-grid").get_data(as_text=True)


def test_review_actions_from_the_hub(client):
    engine = client.engine
    n = len(engine.suspects()["main"])

    first = _shown(client.get("/al-review").get_data(as_text=True))     # 1. conferma
    r = client.post("/al-review", data={"image_id": first, "decision": "confirmed"})
    assert r.status_code == 302
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Label kept." in page and f"{n - 1} suspects left, 1 reviewed" in page and _shown(page) != first

    second = _shown(page)                                              # 2. cambia nella classe suggerita
    suggested = next(s["suggested"] for s in engine.suspects()["main"] if s["image_id"] == second)
    r = client.post("/al-review", data={"image_id": second, "decision": "suggested"})
    assert engine.bank.label_of(second) == suggested and "Label changed." in client.get(r.headers["Location"]).get_data(as_text=True)

    third = _shown(client.get("/al-review").get_data(as_text=True))     # 3. un'altra classe, scritta a mano
    client.post("/al-review", data={"image_id": third, "decision": "corrected", "new_label": "altra classe"})
    assert engine.bank.label_of(third) == "Altra Classe"

    fourth = _shown(client.get("/al-review").get_data(as_text=True))    # 4. scarta
    r = client.post("/al-review", data={"image_id": fourth, "decision": "set_aside"})
    assert engine.bank.label_of(fourth) == UNKNOWN_LABEL and "Bee set aside." in client.get(r.headers["Location"]).get_data(as_text=True)

    log = read_log(engine.review_log_path)
    assert [r["decision"] for r in log] == ["confirmed", "corrected", "corrected", "set_aside"]
    assert {r["annotator"] for r in log} == {"t@x.it"}

    # errori come messaggio, mai un 500
    val = engine.validation_ids[0]
    for form in ({"image_id": val, "decision": "confirmed"}, {"image_id": first, "decision": "corrected", "new_label": ""},
                 {"image_id": "boh.png", "decision": "suggested"}, {"image_id": second, "decision": "confirmed"}):
        r = client.post("/al-review", data=form)
        assert r.status_code == 200 and "color: var(--btn-red)" in r.get_data(as_text=True)
    assert len(read_log(engine.review_log_path)) == 4


def test_review_page_without_suspects_and_with_rare_classes(client):
    engine = client.engine
    for s in engine.suspects()["main"]:
        engine.review("t@x.it", s["image_id"], "corrected", s["suggested"])
    html = client.get("/al-review").get_data(as_text=True)
    assert client.get("/al-review").status_code == 200 and "No suspicious labels to review" in html and _shown(html) is None
    assert ">Review</a>" in html and "Review (" not in html              # nessun conteggio quando la coda è vuota

    rare = engine.labeled[30:32]
    engine.assign_many("t@x.it", rare, "Fenotipo Raro")
    html = client.get("/al-review").get_data(as_text=True)
    assert "Rare classes" in html and "Fenotipo Raro" in html and "expect most to be correct" in html
    assert _shown(html) is None                                         # non si mescolano con la coda principale
    one = client.get(f"/al-review?image={rare[0]}").get_data(as_text=True)
    assert _shown(one) == rare[0] and "Weak signal" in one
    client.post("/al-review", data={"image_id": rare[0], "decision": "confirmed"})
    assert rare[0] not in {s["image_id"] for s in engine.suspects()["rare"]}


def test_status_box_and_downloads(client):
    engine = client.engine
    first = engine.suspects()["main"][0]
    client.post("/al-review", data={"image_id": first["image_id"], "decision": "suggested"})
    html = client.get("/al-status").get_data(as_text=True)
    assert "Label review" in html and "Hand labels checked" in html and "1 corrected" in html
    assert f"{first['label']} -&gt; {first['suggested']}: 1" in html and "Suspicious by declared confidence" in html

    log = client.get("/al-review-file/review_log.jsonl")
    assert log.status_code == 200 and "attachment" in log.headers["Content-Disposition"]
    assert json.loads(log.get_data(as_text=True).splitlines()[0])["decision"] == "corrected"
    report = client.get("/al-review-file/review_report.csv")
    rows = list(csv.DictReader(report.get_data(as_text=True).splitlines()))
    assert len(rows) == 1 and rows[0]["image_id"] == first["image_id"] and rows[0]["new_label"] == first["suggested"]
    assert client.get("/al-review-file/al_labels.csv").status_code == 404
    assert {"/al-review", "/al-review-file/<filename>"} <= {r.rule for r in client.application.url_map.iter_rules()}
