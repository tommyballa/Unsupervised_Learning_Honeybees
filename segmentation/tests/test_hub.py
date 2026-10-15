"""L'hub reale (``active_learning_tool/app.py``) con le pagine di active learning."""
import csv
import importlib.util
import re

import pytest

from bee_al import hub as al_hub
from bee_al.config import REPO_ROOT
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine

ORIGINAL_ROUTES = {
    "/", "/logout", "/get_image/<filename>", "/labeling", "/ab-test",
    "/grid-labeling", "/inspect", "/clean", "/results",
}


@pytest.fixture
def client(hub_app, cfg, synth, monkeypatch, tmp_path):
    X, y, ids = synth
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)  # le route storiche leggono "results" relativo
    hub_app.app.config["TESTING"] = True
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    c.engine = engine
    return c


def _image_id(html):
    return re.search(r'name="image_id" value="([^"]+)"', html).group(1)


def test_existing_routes_are_untouched_and_new_ones_added(hub_app):
    rules = {r.rule for r in hub_app.app.url_map.iter_rules()}
    assert ORIGINAL_ROUTES <= rules
    assert {"/al-labeling", "/al-status", "/al-refresh"} <= rules


def test_dashboard_links_and_old_pages_still_work(client):
    home = client.get("/").get_data(as_text=True)
    assert 'href="/al-labeling"' in home and 'href="/al-status"' in home
    assert 'href="/labeling"' in home and 'href="/clean"' in home
    assert client.get("/results").status_code == 200
    assert client.get("/inspect").status_code == 200


def test_pages_require_login(hub_app):
    c = hub_app.app.test_client()
    for url in ("/al-labeling", "/al-status"):
        r = c.get(url)
        assert r.status_code == 302 and r.headers["Location"].endswith("/")
    assert c.post("/al-refresh").status_code == 302


def test_labeling_saves_label_and_moves_on(client, cfg):
    html = client.get("/al-labeling").get_data(as_text=True)
    first = _image_id(html)
    assert f"/al-image/{first}" in html
    assert "kNN suggestion" not in html                      # suggerimento nascosto di default
    assert _image_id(client.get("/al-labeling").get_data(as_text=True)) == first  # ricaricare non cambia ape

    r = client.post("/al-labeling", data={"image_id": first, "label": "  nuova classe", "confidence": "Low", "action": "label"})
    assert r.status_code == 302
    assert client.engine.bank.label_of(first) == "Nuova Classe"  # add_label chiamato subito
    with open(cfg.al_labels_path) as f:
        (row,) = list(csv.DictReader(f))
    assert row["annotator"] == "t@x.it" and row["confidence"] == "Low" and row["label"] == "Nuova Classe"

    html = client.get("/al-labeling").get_data(as_text=True)
    assert _image_id(html) != first
    assert 'data-label="Nuova Classe"' in html                # la classe nuova è subito una quick label


def test_non_so_button_and_free_text_are_escaped(client):
    first = _image_id(client.get("/al-labeling").get_data(as_text=True))
    client.post("/al-labeling", data={"image_id": first, "label": "", "action": "unknown"})
    assert client.engine.status()["n_unknown"] == 1 and client.engine.bank.classes == []

    second = _image_id(client.get("/al-labeling").get_data(as_text=True))
    client.post("/al-labeling", data={"image_id": second, "label": "<script>x</script>", "action": "label"})
    html = client.get("/al-labeling").get_data(as_text=True)
    assert "<script>x</script>" not in html.lower() and "&lt;script&gt;" in html.lower()


def test_fix_previous_label(client):
    first = _image_id(client.get("/al-labeling").get_data(as_text=True))
    client.post("/al-labeling", data={"image_id": first, "label": "Sbagliata", "action": "label"})
    html = client.get("/al-labeling?redo=1").get_data(as_text=True)
    assert _image_id(html) == first and "Sbagliata" in html
    client.post("/al-labeling", data={"image_id": first, "label": "Giusta", "action": "label"})
    assert client.engine.bank.label_of(first) == "Giusta"
    assert [e.label for e in client.engine.store.events()] == ["Sbagliata", "Giusta"]


def test_validation_submode(client):
    html = client.get("/al-labeling?mode=validation").get_data(as_text=True)
    image_id = _image_id(html)
    assert image_id in client.engine.validation_ids
    client.post("/al-labeling?mode=validation", data={"image_id": image_id, "label": "Sana", "action": "label"})
    assert client.engine.val_labels == {image_id: "Sana"}
    assert client.engine.status()["n_labels"] == 0             # non entra nel kNN
    assert client.engine.store.events()[-1].query_type == "validation"

    bad = client.post("/al-labeling", data={"image_id": image_id, "label": "Sana", "action": "label"})
    assert bad.status_code == 200 and "validation set" in bad.get_data(as_text=True)


def test_status_page_and_refresh_button(client):
    first = _image_id(client.get("/al-labeling").get_data(as_text=True))
    client.post("/al-labeling", data={"image_id": first, "label": "Sana", "action": "label"})
    html = client.get("/al-status").get_data(as_text=True)
    assert "1 of 300 images" in html and "Cold start" in html and "Sana" in html
    assert "r0001: 1 / 10 done" in html and "0 / 20 labeled" in html

    round_before = client.engine.round
    r = client.post("/al-refresh")
    assert r.status_code == 302 and client.engine.round == round_before + 1
    assert "r0002" in client.get("/al-status").get_data(as_text=True)


def test_missing_embeddings_shows_instructions(hub_app, monkeypatch, tmp_path):
    monkeypatch.setattr(al_hub.STATE, "engine", None)

    def boom(_):
        raise FileNotFoundError("Nessun embedding: esegui prima scripts/extract_embeddings.py")

    monkeypatch.setattr(ALEngine, "from_config", classmethod(lambda cls, path=None: boom(path)))
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    r = c.get("/al-labeling")
    assert r.status_code == 503 and "extract_embeddings.py" in r.get_data(as_text=True)
    assert c.get("/results").status_code == 200  # il resto dell'hub continua a funzionare


def test_slow_start_shows_a_loading_page_instead_of_hanging(hub_app, cfg, synth, monkeypatch, tmp_path):
    import threading

    X, y, ids = synth
    gate = threading.Event()

    def slow(cls, path=None):
        gate.wait(10)
        return ALEngine(cfg, Embeddings(ids, list(ids), X))

    monkeypatch.setattr(al_hub.STATE, "engine", None)
    monkeypatch.setattr(al_hub.STATE, "_thread", None)
    monkeypatch.setattr(ALEngine, "from_config", classmethod(slow))
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})

    for url in ("/al-labeling", "/al-status", "/al-clusters"):      # tutte rispondono subito
        r = c.get(url)
        assert r.status_code == 503 and "Active learning is starting" in r.get_data(as_text=True)
        assert 'http-equiv="refresh"' in r.get_data(as_text=True)
    assert c.get("/").status_code == 200                              # la dashboard non aspetta
    gate.set()
    al_hub.STATE.wait(20)
    assert c.get("/al-status").status_code == 200 and "0 of 300 images" in c.get("/al-status").get_data(as_text=True)


def _cluster_tiles(html):
    return re.findall(r'name="image_(\d+)" value="([^"]+)"', html)


def _cluster_form(tiles, groups, labels=()):
    data = {f"image_{k}": image_id for k, image_id in tiles}
    data.update({f"group_{k}": groups.get(int(k), "") for k, _ in tiles})
    data.update({f"label_{g}": label for g, label in enumerate(labels)})
    return data


def hub_app_rules(client):
    return {r.rule for r in client.application.url_map.iter_rules()}


def _label_first_batch(client, truth):
    done = []
    for _ in range(10):
        image_id = _image_id(client.get("/al-labeling").get_data(as_text=True))
        client.post("/al-labeling", data={"image_id": image_id, "label": truth[image_id], "action": "label"})
        done.append(image_id)
    return done


def test_home_keeps_only_active_learning_and_groups_the_old_tools(client):
    html = client.get("/").get_data(as_text=True)
    before, old = html.split("<details>")
    for url in ("/al-grid", "/al-labeling", "/al-labeling?mode=validation", "/al-labeling?mode=audit",
                "/al-clusters", "/al-review", "/al-status"):
        assert f'href="{url}"' in before
    for url in ("/labeling", "/ab-test", "/grid-labeling", "/inspect", "/clean", "/results"):
        assert f'href="{url}"' in old and f'href="{url}"' not in before
    assert html.count("<summary") == 1                           # un solo tasto per tutte le sezioni vecchie


def test_pages_do_not_wait_for_a_busy_engine(client):
    """Il banner di fase gira a ogni pagina: con il motore occupato (re-train, ricalcoli lunghi) non deve bloccare."""
    import threading

    held, release = threading.Event(), threading.Event()

    def busy():
        with client.engine.lock:
            held.set()
            release.wait(10)

    worker = threading.Thread(target=busy)
    worker.start()
    held.wait(5)
    try:
        result = {}
        page = threading.Thread(target=lambda: result.setdefault("r", client.get("/")))
        page.start()
        page.join(5)
        assert not page.is_alive() and result["r"].status_code == 200
    finally:
        release.set()
        worker.join()


def test_clusters_page_shows_one_group_chosen_from_a_dropdown(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    html = client.get("/al-clusters").get_data(as_text=True)
    assert "No clusters yet" in html and "<select" not in html

    _label_first_batch(client, truth)                            # a fine batch si ricalcola
    html = client.get("/al-clusters").get_data(as_text=True)
    classes = client.engine.bank.classes
    assert html.count('id="cluster_select"') == 1 and 'label="Model predictions"' in html and 'label="Labels given by hand"' in html
    assert all(f"group=p:{c}" in html and f"group=h:{c}" in html for c in classes)
    n = client.engine.status()["n_unassigned"]
    assert n == round(0.05 * (290 - 20)) and f"Unassigned - possible new class ({n})" in html
    assert 'class="grid-tiles"' in html and 0 < len(_cluster_tiles(html)) <= 12      # un gruppo solo, api grandi
    assert "very similar" in html and "least similar" in html and "Second third: around the median" in html
    assert "Remove label" not in html

    big = max(client.engine.cluster_samples(0), key=lambda c: c["n_predicted"])
    first = client.get(f"/al-clusters?group=p:{big['label']}").get_data(as_text=True)
    second = client.get(f"/al-clusters?group=p:{big['label']}&page=1").get_data(as_text=True)
    assert len(_cluster_tiles(first)) == 12 and "set 1 of" in first and "set 2 of" in second
    assert not {i for _, i in _cluster_tiles(first)} & {i for _, i in _cluster_tiles(second)}
    sims = [float(v) for v in re.findall(r"sim (-?\d\.\d\d)", first)]
    assert sims == sorted(sims, reverse=True)                    # dalla più simile alla meno simile

    unassigned = client.get("/al-clusters?group=u").get_data(as_text=True)
    assert "far from every labeled example" in unassigned and "their closest labeled bee" in unassigned
    assert client.get("/al-clusters?group=boh&page=abc").status_code == 200
    assert "Unassigned (far from every label)" in client.get("/al-status").get_data(as_text=True)
    assert hub_app_rules(client) >= {"/al-clusters"}


def test_clusters_page_lets_you_choose_how_many_bees_and_their_order(client, synth):
    X, y, ids = synth
    _label_first_batch(client, dict(zip(ids, y)))
    big = max(client.engine.cluster_samples(0), key=lambda c: c["n_predicted"])
    url = f"/al-clusters?group=p:{big['label']}"
    members = big["n_predicted"]

    def page(extra=""):
        html = client.get(url + extra).get_data(as_text=True)
        caps = re.findall(r'class="cap">([^<]*)', html)
        return html, caps, [float(re.search(r"sim (-?\d\.\d\d)", c).group(1)) for c in caps]

    html, caps, _ = page()
    assert len(caps) == 12 and all(f"n={k}" in html for k in (12, 24, 48, 96))
    assert ".grid-left { max-height: 100%; overflow-y: auto;" in html          # scorrono solo le api
    assert "overflow-y: auto" not in client.get("/al-grid").get_data(as_text=True)
    assert all(f"order={o}" in html for o in ("bands", "similar", "dissimilar", "confident", "uncertain", "random"))

    html, caps, sims = page("&n=24&order=uncertain")
    conf = [int(re.search(r"conf (\d+)%", c).group(1)) for c in caps]
    assert len(caps) == min(24, members) and conf == sorted(conf) and "very similar" not in html
    html, caps, _ = page()                                         # la scelta resta, anche senza parametri
    assert len(caps) == min(24, members) and 'order=uncertain" selected' in html
    conf_top = [int(re.search(r"conf (\d+)%", c).group(1)) for c in page("&order=confident")[1]]
    assert conf_top == sorted(conf_top, reverse=True)
    assert page("&order=similar")[2] == sorted(page("&order=similar")[2], reverse=True)
    assert page("&order=dissimilar")[2] == sorted(page("&order=dissimilar")[2])
    first, second = page("&n=12&order=random")[0], page("&page=1")[0]
    assert first == page()[0] and not {i for _, i in _cluster_tiles(first)} & {i for _, i in _cluster_tiles(second)}

    # un gruppo senza confidenza ricade sulle fasce; valori strani ignorati
    hand = client.get(f"/al-clusters?group=h:{big['label']}&order=uncertain").get_data(as_text=True)
    assert "order=confident" not in hand and "order=bands\" selected" in hand
    assert len(page("&n=7&order=boh")[1]) == 12

    # con 24 api mostrate si salvano anche quelle oltre la dodicesima
    tiles = _cluster_tiles(page("&n=24&order=similar")[0])
    if len(tiles) > 12:
        client.post(url, data=_cluster_form(tiles, {len(tiles) - 1: "0"}, ["Ultima"]))
        assert client.engine.bank.label_of(tiles[-1][1]) == "Ultima"


def test_clusters_page_edits_labels(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    done = _label_first_batch(client, truth)
    label = truth[done[0]]
    same = [i for i in done if truth[i] == label]

    html = client.get(f"/al-clusters?group=h:{label}").get_data(as_text=True)
    tiles = _cluster_tiles(html)
    assert {i for _, i in tiles} == set(same) and "Remove label" in html
    predicted = client.get("/al-clusters").get_data(as_text=True)
    assert not {i for _, i in _cluster_tiles(predicted)} & set(done)
    assert client.get("/al-clusters?view=labeled").get_data(as_text=True).count("Remove label") > 0

    # cambio classe a un'ape etichettata, con un colore
    r = client.post(f"/al-clusters?group=h:{label}", data=_cluster_form(tiles, {0: "1"}, ["", "cambiata"]))
    assert r.status_code == 302 and "changed=1" in r.headers["Location"]
    assert client.engine.bank.label_of(tiles[0][1]) == "Cambiata"
    assert "1 bee(s) updated" in client.get(r.headers["Location"]).get_data(as_text=True)

    # più classi in un colpo su api predette, più una scartata
    tiles = _cluster_tiles(client.get("/al-clusters").get_data(as_text=True))
    client.post("/al-clusters",
                data=_cluster_form(tiles, {0: "0", 1: "0", 2: "3", 3: "discard"}, ["Nuovo Gruppo", "", "", "Altro"]))
    assert [client.engine.bank.label_of(tiles[k][1]) for k in range(3)] == ["Nuovo Gruppo", "Nuovo Gruppo", "Altro"]
    assert tiles[3][1] in {client.engine.bank.ids[i] for i in client.engine.bank.unknown_indices()}
    assert {e.query_type for e in client.engine.store.events()[-4:]} == {"manual"}

    # rimozione: solo tra le etichette date a mano
    tiles = _cluster_tiles(client.get("/al-clusters?group=h:Nuovo Gruppo").get_data(as_text=True))
    client.post("/al-clusters?group=h:Nuovo Gruppo", data=_cluster_form(tiles, {0: "remove"}))
    assert client.engine.bank.label_of(tiles[0][1]) is None
    assert "Scarta ape (set aside)" in client.get("/al-clusters").get_data(as_text=True)

    # errori: nessuna selezione, colore senza classe, "remove" su api predette: non si scrive nulla
    n_events = len(client.engine.store.events())
    tiles = _cluster_tiles(client.get("/al-clusters").get_data(as_text=True))
    for groups in ({}, {0: "2"}, {0: "remove"}):
        r = client.post("/al-clusters", data=_cluster_form(tiles, groups, ["A", "B", ""]))
        assert r.status_code == 200 and "color: var(--btn-red)" in r.get_data(as_text=True)
    assert len(client.engine.store.events()) == n_events
    assert hub_app_rules(client) >= {"/al-clusters"} and client.get("/inspect").status_code == 200


def test_retrain_button(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    client.engine.cfg.retrain.trainer = "stub"

    r = client.post("/al-retrain")                            # troppo poche etichette: messaggio, nessun job
    assert r.status_code == 302 and "retrain_error" in r.headers["Location"]
    html = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Re-train not started" in html and "at least 20 labels" in html and "disabled" in html

    for _ in range(3):
        _label_first_batch(client, truth)
    html = client.get("/al-status").get_data(as_text=True)
    assert "Start re-train now" in html and "only when you press the button" in html and "Not available" not in html

    assert client.post("/al-retrain").status_code == 302
    client.engine.retrain.join(20)
    assert client.engine.model_version == "v001"
    html = client.get("/al-status").get_data(as_text=True)
    assert "v001" in html and "In use." in html and "button, t@x.it" not in html
    assert "/al-retrain" in hub_app_rules(client)


def test_review_page_and_menu_entry(client):
    """La pagina di revisione risponde anche senza sospette; i dettagli sono in test_review.py."""
    r = client.get("/al-review")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Label review" in html and "No suspicious labels to review" in html
    for url in ("/al-status", "/al-clusters", "/al-labeling", "/al-grid"):
        page = client.get(url).get_data(as_text=True)
        assert 'href="/al-review"' in page and "Review (" not in page          # voce di menu, senza conteggio a coda vuota
    assert page.index('href="/al-clusters"') < page.index('href="/al-review"') < page.index('href="/al-status"')
    assert "Label review" in client.get("/al-status").get_data(as_text=True)
    assert client.get("/al-review-file/review_log.jsonl").get_data(as_text=True) == ""
    assert client.post("/al-review", data={"image_id": "boh.png", "decision": "confirmed"}).status_code == 200
    c2 = client.application.test_client()
    assert c2.get("/al-review").status_code == 302 and c2.get("/al-review-file/review_log.jsonl").status_code == 403
