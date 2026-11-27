import csv

import numpy as np
import pytest
import yaml

from bee_al import REMOVED_LABEL, UNKNOWN_LABEL
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine, ALError
from bee_al.label_store import LabelStore


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_engine(cfg, synth, **kw):
    X, y, ids = synth
    return ALEngine(cfg, Embeddings(ids, [f"/d/{i}" for i in ids], X), **kw)


def label_batch(engine, truth, annotator="a@x.it"):
    """Etichetta tutto il batch corrente con l'oracolo. Restituisce le immagini etichettate."""
    batch_id, done = engine.batch_id, []
    while engine.batch_id == batch_id:
        item = engine.next_image(annotator)
        if item is None:
            break
        engine.submit(annotator, item.image_id, truth[item.image_id], "High")
        done.append(item.image_id)
    return done


@pytest.fixture
def truth(synth):
    _, y, ids = synth
    return dict(zip(ids, y))


def test_starts_in_cold_start_with_a_batch_outside_validation(cfg, synth):
    engine = make_engine(cfg, synth)
    st = engine.status()
    assert st["cold_start"] and st["n_labels"] == 0 and st["round"] == 1
    assert len(engine.batch) == 10
    assert {b.query_type for b in engine.batch} <= {"diversity", "random"}
    assert len(engine.validation_ids) == 20
    assert not {b.image_id for b in engine.batch} & set(engine.validation_ids)


def test_submit_updates_bank_immediately_and_writes_event(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, "  nuova   classe ", "Medium")
    assert engine.bank.label_of(item.image_id) == "Nuova Classe"   # add_label subito, senza refresh
    assert engine.round == 1
    assert "Nuova Classe" in engine.quick_labels()
    with open(cfg.al_labels_path) as f:
        (row,) = list(csv.DictReader(f))
    assert row["image_id"] == item.image_id and row["label"] == "Nuova Classe"
    assert row["annotator"] == "a@x.it" and row["confidence"] == "Medium"
    assert row["query_type"] == item.query_type and row["batch_id"] == "r0001"
    assert row["config_id"] == cfg.config_id and row["model_version"] == "v000"


def test_finished_batch_triggers_refresh_and_regime_change(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    seen = []
    for _ in range(4):
        seen += label_batch(engine, truth)
    assert len(seen) == len(set(seen)) == 40           # mai riproposte api già etichettate
    assert engine.round == 5 and engine.status()["n_labels"] == 40
    assert not engine.status()["cold_start"]            # soglia 10 superata, più di una classe
    assert "uncertainty" in {b.query_type for b in engine.batch}
    assert not set(seen) & {b.image_id for b in engine.batch}
    assert not set(seen) & set(engine.validation_ids)   # il validation set non viene mai interrogato


def test_unknown_is_set_aside_and_never_proposed_again(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, UNKNOWN_LABEL)
    assert engine.bank.classes == [] and engine.status()["n_unknown"] == 1
    seen = []
    for _ in range(5):
        seen += label_batch(engine, truth)
    assert item.image_id not in seen
    assert UNKNOWN_LABEL not in engine.quick_labels()
    assert all(item.image_id not in [m["image_id"] for m in members] for members in engine.final_clusters().values())


def test_shared_batch_one_annotator_per_image(cfg, synth):
    clock = Clock()
    engine = make_engine(cfg, synth, clock=clock)
    a = engine.next_image("a@x.it")
    b = engine.next_image("b@x.it")
    assert a.image_id != b.image_id
    assert engine.next_image("a@x.it").image_id == a.image_id  # ricaricare la pagina non cambia immagine
    assigned = {engine.next_image(f"u{i}@x.it").image_id for i in range(8)}
    assert len(assigned | {a.image_id, b.image_id}) == 10
    # Tutto il batch è già assegnato: chi arriva non aspetta, parte il round successivo
    round_before = engine.round
    late = engine.next_image("late@x.it")
    assert late is not None and late.image_id not in assigned | {a.image_id, b.image_id}
    assert engine.round == round_before + 1
    assert engine.next_image("a@x.it").image_id == a.image_id  # chi aveva un'ape la tiene
    many = engine.next_images("late2@x.it", 5)                  # lo stesso per la griglia
    assert many and not {m.image_id for m in many} & (assigned | {a.image_id, b.image_id, late.image_id})


def test_refresh_keeps_images_someone_is_looking_at(cfg, synth):
    engine = make_engine(cfg, synth)
    a = engine.next_image("a@x.it")
    engine.refresh()
    assert engine.next_image("a@x.it").image_id == a.image_id
    assert len({b.image_id for b in engine.batch}) == len(engine.batch) == 11
    engine.submit("a@x.it", a.image_id, "X")
    assert engine.store.events()[-1].batch_id == "r0001"       # il batch che l'aveva proposta


def test_validation_mode_is_separate_and_used_only_for_metrics(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    for _ in range(3):
        label_batch(engine, truth)
    assert engine.status()["metrics"]["accuracy"] is None       # nessuna etichetta di validazione ancora

    labeled_val = []
    for _ in range(20):
        item = engine.next_image("a@x.it", mode="validation")
        assert item.image_id in engine.validation_ids and item.query_type == "validation"
        engine.submit("a@x.it", item.image_id, truth[item.image_id], "High", mode="validation")
        labeled_val.append(item.image_id)
    assert engine.next_image("a@x.it", mode="validation") is None
    assert sorted(labeled_val) == sorted(engine.validation_ids)
    assert engine.status()["n_labels"] == 30                    # le etichette di validazione non entrano nella bank
    assert all(engine.bank.label_of(i) is None for i in labeled_val)

    entry = engine.refresh()
    assert entry["n_validation"] == 20 and entry["accuracy"] > 0.8 and entry["macro_f1"] > 0.5
    assert entry["knn_accuracy"] is not None

    with pytest.raises(ALError):   # un'ape di validazione non si etichetta in modalità normale
        engine.submit("a@x.it", labeled_val[0], "X")
    with pytest.raises(ALError):   # e viceversa
        engine.submit("a@x.it", engine.batch[0].image_id, "X", mode="validation")


def test_validation_set_is_fixed_on_disk(cfg, synth):
    first = make_engine(cfg, synth).validation_ids
    assert cfg.validation_path.exists()
    cfg.validation.seed = 999  # il file salvato vince sulla config
    assert make_engine(cfg, synth).validation_ids == first


def test_legacy_labels_feed_knn_but_never_evaluation(cfg, synth, truth):
    X, y, ids = synth
    probe = make_engine(cfg, synth)
    val_id = probe.validation_ids[0]
    train_id = next(i for i in ids if i not in probe.validation_ids)

    store = LabelStore(cfg.al_labels_path)
    store.append(val_id, truth[val_id], query_type="legacy")
    store.append(train_id, truth[train_id], query_type="legacy")
    engine = make_engine(cfg, synth, store=store)
    assert engine.bank.label_of(train_id) == truth[train_id]   # usata per kNN e propagazione
    assert engine.bank.label_of(val_id) is None                # mai nella bank se è di validazione
    assert engine.val_labels == {}                             # e mai per la valutazione
    assert engine.history[-1]["n_validation"] == 0
    with pytest.raises(ALError):                               # le legacy si correggono dalle pagine esistenti
        engine.submit("a@x.it", train_id, "Altro")


def test_alias_change_only_needs_config_and_refresh(cfg, synth, truth, tmp_path):
    config_path = tmp_path / "al_config.yaml"
    config_path.write_text(yaml.safe_dump({"labels": {"classes": ["Sana", "Varroa"]}}))
    engine = make_engine(cfg, synth, config_path=config_path)
    for _ in range(3):
        label_batch(engine, truth)
    assert len(engine.bank.classes) > 2
    before = cfg.al_labels_path.read_bytes()

    config_path.write_text(yaml.safe_dump({"labels": {
        "label_aliases": {"Class_1": "Class_0", "Class_2": "Class_0"},
        "reject_classes": ["Class_3"],
        "classes": ["Sana"],
    }}))
    engine.refresh()
    assert set(engine.bank.classes) <= {"Class_0", "Class_3"}
    assert engine.knn_pred.classes == engine.bank.classes       # il kNN segue subito le classi nuove
    assert cfg.al_labels_path.read_bytes() == before            # nessuna modifica al CSV
    assert "Sana" in engine.quick_labels() and "Class_1" not in engine.quick_labels()
    assert "Class_3" not in engine.final_clusters()             # classe di scarto fuori dai cluster finali


def test_correction_and_removal_are_new_events(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    first = label_batch(engine, truth)[0]
    n_events = len(engine.store.events())

    engine.submit("a@x.it", first, "Corretta")                  # correzione di un batch precedente
    assert engine.bank.label_of(first) == "Corretta"
    engine.submit("a@x.it", first, REMOVED_LABEL)
    assert engine.bank.label_of(first) is None
    events = engine.store.events()
    assert len(events) == n_events + 2
    assert events[-1].query_type == events[0].query_type and events[-1].batch_id == "r0001"
    with pytest.raises(ALError):
        engine.submit("a@x.it", "mai_vista.png", "X")
    with pytest.raises(ALError):
        engine.submit("a@x.it", engine.batch[0].image_id, "   ")


def test_restart_rebuilds_state_from_store(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    label_batch(engine, truth)
    label_batch(engine, truth)
    restarted = make_engine(cfg, synth)
    assert restarted.bank.class_counts() == engine.bank.class_counts()
    assert restarted.round == engine.round + 1 and len(restarted.history) == restarted.round
    labeled = {i for i in restarted.bank.ids if restarted.bank.label_of(i)}
    assert not labeled & {b.image_id for b in restarted.batch}


def test_suggestion_hidden_by_default(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    label_batch(engine, truth)
    image_id = engine.batch[0].image_id
    assert engine.suggestion(image_id) is None
    cfg.hub.show_suggestion = True
    label, confidence = engine.suggestion(image_id)
    assert label in engine.bank.classes and 0 < confidence <= 1


def test_single_class_stays_in_cold_start(cfg, synth):
    engine = make_engine(cfg, synth)
    for _ in range(3):
        batch_id = engine.batch_id
        while engine.batch_id == batch_id:
            item = engine.next_image("a@x.it")
            engine.submit("a@x.it", item.image_id, "Solo")
    st = engine.status()
    assert st["n_labels"] == 30 and st["class_counts"] == {"Solo": 30}
    assert st["cold_start"]                                      # meno di 2 classi
    assert {b.query_type for b in engine.batch} <= {"diversity", "random"}


def test_cluster_samples_show_predicted_bees_per_class(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    assert engine.cluster_samples() == []                       # senza etichette non ci sono cluster
    for _ in range(3):
        label_batch(engine, truth)
    human = {i for i in engine.bank.ids if engine.bank.label_of(i)}

    clusters = engine.cluster_samples(n=5)
    assert [c["label"] for c in clusters] == engine.final_pred.classes
    n_unassigned = engine.status()["n_unassigned"]              # tenute fuori dai cluster
    assert sum(c["n_predicted"] for c in clusters) == 300 - len(human) - n_unassigned - len(engine.validation_ids)
    shown = {b["image_id"] for c in engine.cluster_samples(n=500) for b in c["samples"]}
    assert not shown & set(engine.validation_ids)               # il validation set non compare tra i cluster
    assert sum(c["n_human"] for c in clusters) == len(human)
    for c in clusters:
        ids = [b["image_id"] for b in c["samples"]]
        assert len(ids) == min(5, c["n_predicted"]) == len(set(ids))
        assert not set(ids) & human                             # solo api assegnate dal modello
        assert all(0 < b["confidence"] <= 1 for b in c["samples"])
    assert clusters == engine.cluster_samples(n=5)              # campione stabile fino al prossimo round

    top = engine.cluster_samples(n=5, order="confident")
    low = engine.cluster_samples(n=5, order="uncertain")
    for a, b in zip(top, low):
        if a["n_predicted"] > 5:
            assert min(x["confidence"] for x in a["samples"]) >= max(x["confidence"] for x in b["samples"])
    big = next(c for c in top if c["n_predicted"] > 5)
    right = sum(truth[b["image_id"]] == big["label"] for b in big["samples"])
    assert right >= 4                                           # le più sicure sono quasi sempre giuste


def test_assign_many_edits_single_bees_and_groups(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    done = label_batch(engine, truth) + label_batch(engine, truth)
    round_before, batch_before = engine.round, [b.image_id for b in engine.batch]
    n_events = len(engine.store.events())

    # cambio idea su un'ape etichettata
    assert engine.assign_many("a@x.it", [done[0]], "ripensata") == 1
    assert engine.bank.label_of(done[0]) == "Ripensata"
    last = engine.store.events()[-1]
    assert last.query_type == engine.store.events()[0].query_type   # correzione: resta il tipo di query originale
    assert "Ripensata" in engine.final_pred.classes                  # predizioni ricalcolate subito...
    assert engine.round == round_before and [b.image_id for b in engine.batch] == batch_before  # ...senza nuovo round o batch

    # un gruppo di api mai proposte, prese dai cluster predetti
    cluster = next(c for c in engine.cluster_samples(n=4) if len(c["samples"]) == 4)
    group = [b["image_id"] for b in cluster["samples"]]
    assert engine.assign_many("b@x.it", group + group[:1], "Gruppo Nuovo") == 4
    assert all(engine.bank.label_of(i) == "Gruppo Nuovo" for i in group)
    events = engine.store.events()[-4:]
    assert {e.query_type for e in events} == {"manual"} and {e.annotator for e in events} == {"b@x.it"}
    assert not set(group) & {b["image_id"] for c in engine.cluster_samples(n=500) for b in c["samples"]}

    # ripetere la stessa scelta non scrive nulla; rimozione e "Non so" sono eventi nuovi
    count = len(engine.store.events())
    assert engine.assign_many("b@x.it", group, "Gruppo Nuovo") == 0 and len(engine.store.events()) == count
    assert engine.assign_many("b@x.it", group[:2], REMOVED_LABEL) == 2
    assert engine.assign_many("b@x.it", group[:2], REMOVED_LABEL) == 0  # non c'è più nulla da togliere
    assert engine.assign_many("b@x.it", group[2:3], UNKNOWN_LABEL) == 1
    assert engine.bank.label_of(group[0]) is None and engine.bank.label_of(group[2]) == UNKNOWN_LABEL
    assert len(engine.store.events()) == count + 3 and len(engine.store.events()) > n_events

    # o tutto o niente: un'ape di validazione nel gruppo blocca l'intera modifica
    count = len(engine.store.events())
    with pytest.raises(ALError):
        engine.assign_many("b@x.it", [group[3], engine.validation_ids[0]], "X")
    with pytest.raises(ALError):
        engine.assign_many("b@x.it", [group[3]], "  ")
    assert len(engine.store.events()) == count and engine.bank.label_of(group[3]) == "Gruppo Nuovo"

    # dopo un riavvio le modifiche ci sono ancora
    restarted = make_engine(cfg, synth)
    assert restarted.bank.label_of(done[0]) == "Ripensata" and restarted.bank.label_of(group[3]) == "Gruppo Nuovo"


def test_labeled_samples_lists_hand_labels_most_recent_first(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    done = label_batch(engine, truth)
    engine.assign_many("a@x.it", [done[0]], UNKNOWN_LABEL)
    groups = engine.labeled_samples(n=100)
    assert groups[-1]["unknown"] and [b["image_id"] for b in groups[-1]["samples"]] == [done[0]]
    listed = {b["image_id"] for g in groups for b in g["samples"]}
    assert listed == set(done)
    assert sum(g["n_human"] for g in groups) == 10
    assert all(len(g["samples"]) <= 2 for g in engine.labeled_samples(n=2))


def test_manual_label_on_pending_batch_image_closes_it(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    target = engine.batch[3]
    engine.assign_many("a@x.it", [target.image_id], "Vista Nei Cluster")
    assert target.done and engine.store.events()[-1].query_type == target.query_type
    seen = label_batch(engine, truth)
    assert target.image_id not in seen and len(seen) == 9          # non viene riproposta


def test_similarity_bands_take_top_median_and_bottom_thirds():
    from bee_al.engine import similarity_bands

    sims = np.linspace(1.0, 0.0, 120)                            # indice = posizione, dalla più simile
    picks, n_pages = similarity_bands(sims, 12, 0)
    assert n_pages == 9 and [b for _, b in picks] == ["high"] * 4 + ["median"] * 4 + ["low"] * 4
    assert [i for i, _ in picks] == [0, 1, 2, 3, 58, 59, 60, 61, 116, 117, 118, 119]
    seen = [i for page in range(n_pages) for i, _ in similarity_bands(sims, 12, page)[0]]
    assert len(seen) == len(set(seen)) == 108                    # ogni pagina api nuove, mai ripetute
    assert similarity_bands(sims, 12, n_pages) == (picks, n_pages)   # dopo l'ultima pagina si ricomincia
    assert similarity_bands(sims, 12, -1)[0] == similarity_bands(sims, 12, n_pages - 1)[0]

    shuffled = np.random.default_rng(0).permutation(sims)
    top = [i for i, b in similarity_bands(shuffled, 12, 0)[0] if b == "high"]
    assert sorted(shuffled[top], reverse=True) == sorted(shuffled, reverse=True)[:4]
    few, pages = similarity_bands(np.array([0.2, 0.9, 0.5, 0.7, 0.1]), 12, 3)
    assert pages == 1 and [i for i, _ in few] == [1, 3, 2, 0, 4] and [b for _, b in few] == ["high", "high", "median", "median", "low"]
    assert similarity_bands(np.array([]), 12, 0) == ([], 1)
    for m in range(1, 80):                                       # nessuna ripetizione, qualunque sia la dimensione
        pages = similarity_bands(np.arange(m, dtype=float), 12, 0)[1]
        for page in range(pages):
            shown = [i for i, _ in similarity_bands(np.arange(m, dtype=float), 12, page)[0]]
            assert len(shown) == len(set(shown)) == min(m, 12)


def test_cluster_view_orders_one_group_by_similarity_to_its_centre(cfg, synth, truth):
    engine = make_engine(cfg, synth)
    assert engine.cluster_view("predicted", "boh") is None and engine.cluster_view("unassigned") is None
    label_batch(engine, truth)
    name = max(engine.cluster_samples(0), key=lambda c: c["n_predicted"])["label"]
    view = engine.cluster_view("predicted", name, 12, 0)
    members = {b["image_id"] for c in engine.cluster_samples(5000) if c["label"] == name for b in c["samples"]}
    assert view["n_members"] == len(members) and {b["image_id"] for b in view["samples"]} <= members
    X = engine.bank.X[[engine.bank.index_of(i) for i in sorted(members, key=engine.bank.index_of)]]
    centre = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    sims = sorted((X @ centre).tolist(), reverse=True)
    assert view["samples"][0]["similarity"] == pytest.approx(sims[0], abs=1e-5)
    assert view["samples"][-1]["similarity"] == pytest.approx(sims[-1], abs=1e-5)
    assert view == engine.cluster_view("predicted", name, 12, 0)             # deterministico

    hand = engine.cluster_view("labeled", name, 12, 0)
    assert all(engine.bank.label_of(b["image_id"]) == name and b["annotator"] for b in hand["samples"])
    far = engine.cluster_view("unassigned", "", 12, 0)
    assert far["n_members"] == engine.status()["n_unassigned"]
    assert not {b["image_id"] for b in far["samples"]} & members
    assert engine.cluster_view("altro") is None
