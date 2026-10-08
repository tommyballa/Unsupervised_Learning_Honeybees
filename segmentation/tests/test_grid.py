"""Etichettatura a griglia (più classi in un colpo, un colore per classe) e fasi successive su un sottoinsieme."""
import json
import re
import subprocess
import sys

import pytest
import yaml

from bee_al import UNKNOWN_LABEL
from bee_al import hub as al_hub
from bee_al.config import REPO_ROOT, config_from_dict
from bee_al.embeddings import Embeddings, restrict_embeddings, save_embeddings
from bee_al.engine import ALEngine

from test_engine import Clock, label_batch, make_engine


def test_next_images_hands_out_a_grid_per_annotator(cfg, synth):
    clock = Clock()
    engine = make_engine(cfg, synth, clock=clock)                    # batch di 10
    a = engine.next_images("a@x.it", 6)
    b = engine.next_images("b@x.it", 6)
    assert len(a) == 6 and len(b) == 4                               # il batch è finito: a b restano 4
    assert not {i.image_id for i in a} & {i.image_id for i in b}
    assert [i.image_id for i in engine.next_images("a@x.it", 6)] == [i.image_id for i in a]   # ricaricare non cambia griglia
    assert engine.next_images("c@x.it", 6) == []

    engine.submit("a@x.it", a[0].image_id, "Bene")
    again = engine.next_images("a@x.it", 6)
    assert [i.image_id for i in again] == [i.image_id for i in a[1:]]   # le non etichettate tornano, niente di nuovo da prendere
    clock.t += cfg.hub.assignment_timeout_s + 1
    assert len(engine.next_images("c@x.it", 12)) == 9                 # assegnazioni scadute: tutto libero

    val = engine.next_images("a@x.it", 12, mode="validation")
    assert len(val) == 12 and {i.image_id for i in val} <= set(engine.validation_ids)
    assert len(engine.next_images("b@x.it", 12, mode="validation")) == 8
    assert not {i.image_id for i in val} & {i.image_id for i in engine.next_images("b@x.it", 12, mode="validation")}


@pytest.fixture
def client(hub_app, cfg, synth, monkeypatch, tmp_path):
    X, y, ids = synth
    cfg.query.batch_size = 30
    cfg.labels.classes = ["segmentata bene", "Segmentata Male"]
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    c.engine = engine
    return c


def _tiles(html):
    return re.findall(r'name="image_(\d+)" value="([^"]+)"', html)


def _form(tiles, groups, labels):
    data = {f"image_{k}": image_id for k, image_id in tiles}
    data.update({f"group_{k}": groups.get(int(k), "") for k, _ in tiles})
    data.update({f"label_{g}": label for g, label in enumerate(labels)})
    return data


def test_grid_page_layout_and_suggested_classes(client):
    assert 'href="/al-grid"' in client.get("/").get_data(as_text=True)
    html = client.get("/al-grid").get_data(as_text=True)
    tiles = _tiles(html)
    assert len(tiles) == 12 and len({i for _, i in tiles}) == 12      # 4 colonne x 3 righe
    assert "grid-template-columns: repeat(4, 1fr)" in html and html.count('class="grid-tile"') == 12
    assert html.index('class="grid-left"') < html.index('class="grid-right"')   # immagini a sinistra, comandi a destra
    assert html.count('data-group="') == 7 and 'data-group="discard"' in html   # 6 colori + "Scarta ape"
    # i primi due colori partono con le classi suggerite in config
    assert 'name="label_0" value="Segmentata Bene"' in html and 'name="label_1" value="Segmentata Male"' in html
    assert 'name="label_2" value=""' in html and "Scarta ape" in html
    assert all(f"/al-image/{i}" in html for _, i in tiles)


def test_grid_saves_several_classes_at_once(client):
    engine = client.engine
    tiles = _tiles(client.get("/al-grid").get_data(as_text=True))
    ids = [i for _, i in tiles]
    groups = {0: "0", 1: "0", 2: "0", 3: "1", 4: "1", 5: "2", 6: "discard"}     # 7 api colorate, 5 lasciate
    r = client.post("/al-grid", data=_form(tiles, groups, ["Segmentata Bene", "segmentata male", "  classe nuova ", "", "", ""]))
    assert r.status_code == 302 and "saved=7" in r.headers["Location"]
    assert [engine.bank.label_of(i) for i in ids[:7]] == ["Segmentata Bene"] * 3 + ["Segmentata Male"] * 2 + ["Classe Nuova", UNKNOWN_LABEL]
    assert all(engine.bank.label_of(i) is None for i in ids[7:])
    events = engine.store.events()
    assert len(events) == 7 and {e.annotator for e in events} == {"t@x.it"} and {e.batch_id for e in events} == {"r0001"}
    assert {e.query_type for e in events} <= {"diversity", "random"}

    html = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "7 bees saved." in html and "7 / 30 done" in html
    nxt = [i for _, i in _tiles(html)]
    assert nxt[:5] == ids[7:] and len(nxt) == 12 and not set(nxt) & set(ids[:7])   # le 5 lasciate tornano, più 7 nuove
    # i colori ricordano le classi date, anche quella nuova
    assert 'name="label_2" value="classe nuova"' in html or 'name="label_2" value="Classe Nuova"' in html


def test_grid_errors_save_nothing(client):
    engine = client.engine
    tiles = _tiles(client.get("/al-grid").get_data(as_text=True))
    r = client.post("/al-grid", data=_form(tiles, {0: "0", 1: "3"}, ["Segmentata Bene", "", "", "", "", ""]))
    assert r.status_code == 200 and "has no class" in r.get_data(as_text=True)     # colore usato ma senza classe
    r = client.post("/al-grid", data=_form(tiles, {}, ["Segmentata Bene", "", "", "", "", ""]))
    assert "No bee has a colour yet" in r.get_data(as_text=True)
    bad = _form(tiles, {0: "0"}, ["Segmentata Bene", "", "", "", "", ""])
    bad["image_0"] = "non_esiste.png"
    assert "cannot label" in client.post("/al-grid", data=bad).get_data(as_text=True)
    assert engine.store.events() == [] and engine.bank.n_labeled == 0


def test_grid_finishes_the_batch_and_validation_grid(client):
    engine = client.engine
    for _ in range(3):                                               # 12 + 12 + 6 = batch da 30
        tiles = _tiles(client.get("/al-grid").get_data(as_text=True))
        client.post("/al-grid", data=_form(tiles, {int(k): str(int(k) % 2) for k, _ in tiles}, ["Bene", "Male", "", "", "", ""]))
    assert engine.round == 2 and engine.bank.n_labeled == 30 and engine.bank.classes == ["Bene", "Male"]
    assert len(_tiles(client.get("/al-grid").get_data(as_text=True))) == 12       # batch successivo

    html = client.get("/al-grid?mode=validation").get_data(as_text=True)
    tiles = _tiles(html)
    assert "Validation grid" in html and {i for _, i in tiles} <= set(engine.validation_ids)
    client.post("/al-grid?mode=validation", data=_form(tiles, {0: "0", 1: "1"}, ["Bene", "Male", "", "", "", ""]))
    assert engine.val_labels == {tiles[0][1]: "Bene", tiles[1][1]: "Male"} and engine.bank.n_labeled == 30
    assert {e.query_type for e in engine.store.events()[-2:]} == {"validation"}


# ------------------------------ fasi successive ----------------------------- #

def test_subset_file_restricts_the_engine(colony, tmp_path):
    keep = colony.emb.ids[::3]
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"image_ids": keep + ["non_esiste.png"]}))
    emb = restrict_embeddings(colony.emb, subset)
    assert emb.ids == keep and emb.X.shape[0] == len(keep) and emb.meta["subset_of"] == len(colony.emb.ids)
    with pytest.raises(FileNotFoundError):
        restrict_embeddings(colony.emb, tmp_path / "boh.json")
    subset.write_text(json.dumps({"image_ids": ["x"]}))
    with pytest.raises(ValueError):
        restrict_embeddings(colony.emb, subset)
    assert config_from_dict({}).subset_path is None
    assert config_from_dict({"paths": {"subset_file": "s.json"}}, root=tmp_path).subset_path == tmp_path / "s.json"


def test_new_stage_script_builds_a_second_stage_on_the_good_bees(colony, tmp_path):
    cfg = colony.cfg
    engine = ALEngine(cfg, colony.emb)
    good = lambda i: "Segmentata Bene" if colony.truth[i] in ("Class_0", "Class_1") else "Segmentata Male"
    for _ in range(4):
        label_batch(engine, {i: good(i) for i in colony.truth})
    engine.export_snapshot("fase1", "a@x.it")
    save_embeddings(cfg.embeddings_dir, colony.emb.ids, colony.emb.paths, colony.emb.X)
    config_path = tmp_path / "al_fase1.yaml"
    data = cfg.to_dict()
    data["paths"].update(results_dir=str(cfg.results_dir), models_dir=str(cfg.models_dir), embeddings_dir=str(cfg.embeddings_dir),
                         dataset_dirs={k: str(cfg.resolve(v)) for k, v in cfg.paths.dataset_dirs.items()})
    config_path.write_text(yaml.safe_dump(data))

    script = REPO_ROOT / "scripts" / "new_stage.py"
    out_config = tmp_path / "al_fase2.yaml"
    args = [sys.executable, str(script), "--snapshot", "fase1", "--config", str(config_path), "--out-config", str(out_config)]
    run = subprocess.run(args + ["--keep", "segmentata bene", "--name", "fase2"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "BEE_AL_CONFIG=" in run.stdout

    from bee_al.snapshot import read_csv
    rows = read_csv(cfg.results_dir / "snapshots" / "fase1" / "predictions.csv")
    expected = [r["image_id"] for r in rows if r["final_label"] == "Segmentata Bene" and r["status"] in ("labeled", "predicted")]
    subset = json.loads((cfg.results_dir.parent / "results_fase2" / "subset.json").read_text())
    assert subset["image_ids"] == expected and subset["keep"] == ["Segmentata Bene"] and 0 < len(expected) < len(rows)

    stage2 = ALEngine.from_config(out_config)                         # la fase 2 parte pulita, solo sulle api buone
    assert stage2.bank.ids == expected and stage2.bank.n_labeled == 0 and stage2.cfg.labels.classes == []
    assert stage2.cfg.results_dir == cfg.results_dir.parent / "results_fase2" and stage2.status()["subset_file"].endswith("subset.json")
    assert set(stage2.validation_ids) <= set(expected)
    label_batch(stage2, colony.truth)
    assert stage2.bank.n_labeled == 10 and engine.bank.class_counts() == ALEngine(cfg, colony.emb).bank.class_counts()   # la fase 1 non cambia

    for extra, text in ((["--keep", "Mai Vista", "--name", "fase3"], "non presenti"), (["--keep", "Segmentata Bene", "--name", "fase2"], "esiste già"),
                        (["--keep", "Segmentata Bene", "--name", "con spazi"], "non valido")):
        bad = subprocess.run(args[:4] + ["--config", str(config_path)] + extra + (["--out-config", str(out_config)] if "fase2" in extra else []),
                             capture_output=True, text=True)
        assert bad.returncode != 0 and text in bad.stderr, bad.stderr


def test_hub_config_can_be_chosen_with_an_environment_variable(monkeypatch):
    monkeypatch.setenv("BEE_AL_CONFIG", "config/al_fase2.yaml")
    monkeypatch.setattr(al_hub.STATE, "config_path", None)
    al_hub.create_blueprint("")
    assert al_hub.STATE.config_path == "config/al_fase2.yaml"
    monkeypatch.delenv("BEE_AL_CONFIG")
    al_hub.create_blueprint("")
    assert al_hub.STATE.config_path is None


def test_colour_classes_are_remembered_per_phase(client):
    tiles = _tiles(client.get("/al-grid").get_data(as_text=True))
    client.post("/al-grid", data=_form(tiles, {0: "2"}, ["Segmentata Bene", "Segmentata Male", "Solo Fase Uno", "", "", ""]))
    assert 'name="label_2" value="Solo Fase Uno"' in client.get("/al-grid").get_data(as_text=True)
    client.engine.cfg.stage.name = "fase2"                              # stessa sessione del browser, fase diversa
    client.engine.cfg.labels.classes = []
    html = client.get("/al-grid").get_data(as_text=True)
    assert 'value="Solo Fase Uno"' not in html.split("<datalist")[0].split('name="label_2"')[1][:40]
    assert 'name="label_2" value=""' in html or 'name="label_2" value="Solo Fase Uno"' not in html


def test_reshuffle_replaces_only_the_unlabeled_bees_of_the_batch(cfg, synth):
    clock = Clock()
    engine = make_engine(cfg, synth, clock=clock)                    # batch di 10
    mine = engine.next_images("a@x.it", 4)
    theirs = engine.next_images("b@x.it", 3)                         # b ne sta guardando 3
    engine.submit("a@x.it", mine[0].image_id, "Bene")
    before = {b.image_id for b in engine.batch}
    round_before, labels_before = engine.round, engine.bank.n_labeled

    assert engine.reshuffle_batch("a@x.it") == 6                     # 10 - 1 etichettata - 3 di b
    after = {b.image_id for b in engine.batch}
    assert len(engine.batch) == 10 and engine.round == round_before and engine.bank.n_labeled == labels_before
    assert mine[0].image_id in after and {i.image_id for i in theirs} <= after
    replaced = before - after
    assert len(replaced) == 6 and not replaced & {i.image_id for i in theirs}
    forbidden = set(engine.validation_ids) | {mine[0].image_id}
    assert not (after - before) & forbidden
    assert [i.image_id for i in engine.next_images("b@x.it", 3)] == [i.image_id for i in theirs]

    engine.reshuffle_batch("a@x.it")                                 # le api tolte non tornano nello stesso round
    assert not {b.image_id for b in engine.batch} & replaced

    engine.refresh()                                                 # round nuovo: tornano candidabili
    assert engine._shuffled_out == set()


def test_reshuffle_is_deterministic_and_refuses_an_empty_batch(cfg, synth):
    def run():
        (cfg.results_dir / "al_rounds.jsonl").unlink(missing_ok=True)   # stesso round a ogni avvio
        engine = make_engine(cfg, synth)
        engine.reshuffle_batch("a@x.it")
        return engine
    a, b = run(), run()
    assert [i.image_id for i in a.batch] == [i.image_id for i in b.batch]
    for item in list(a.batch)[:-1]:
        a.submit("a@x.it", item.image_id, "Bene")
    last = a.next_images("b@x.it", 1)                                # l'ultima è in mano a un altro
    assert len(last) == 1
    with pytest.raises(Exception, match="no unlabeled bees"):
        a.reshuffle_batch("a@x.it")


def test_shuffle_button_on_grid_and_single_page(client):
    html = client.get("/al-grid").get_data(as_text=True)
    assert "Shuffle: propose other bees" in html and "/al-reshuffle?next=grid" in html
    assert "Shuffle: propose other bees" not in client.get("/al-grid?mode=validation").get_data(as_text=True)
    first = [i for _, i in _tiles(html)]
    r = client.post("/al-reshuffle?next=grid")
    assert r.status_code == 302 and "/al-grid" in r.headers["Location"] and "shuffled=30" in r.headers["Location"]
    html = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "30 bees of the batch replaced" in html and not set(first) & {i for _, i in _tiles(html)}
    assert client.engine.bank.n_labeled == 0

    single = client.get("/al-labeling").get_data(as_text=True)
    assert "/al-reshuffle?next=labeling" in single
    assert "/al-reshuffle" not in client.get("/al-labeling?mode=validation").get_data(as_text=True)
    r = client.post("/al-reshuffle?next=labeling")
    assert r.status_code == 302 and "/al-labeling" in r.headers["Location"]
