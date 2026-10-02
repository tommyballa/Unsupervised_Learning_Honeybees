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


def test_clusters_page(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    html = client.get("/al-clusters").get_data(as_text=True)
    assert "No clusters yet" in html and 'href="/al-clusters"' in client.get("/").get_data(as_text=True)

    for _ in range(10):                                         # un batch intero: a fine batch si ricalcola
        image_id = _image_id(client.get("/al-labeling").get_data(as_text=True))
        client.post("/al-labeling", data={"image_id": image_id, "label": truth[image_id], "action": "label"})
    html = client.get("/al-clusters?n=6&order=confident").get_data(as_text=True)
    classes = client.engine.bank.classes
    assert all(c in html for c in classes) and "predicted" in html
    assert 0 < html.count("/al-image/") <= 6 * (len(classes) + 1)   # +1: gruppo "non assegnate"
    assert html.count("/al-image/") > client.get("/al-clusters?n=1").get_data(as_text=True).count("/al-image/")
    assert client.get("/al-clusters?n=abc&order=boh").status_code == 200
    assert hub_app_rules(client) >= {"/al-clusters"}


def hub_app_rules(client):
    return {r.rule for r in client.application.url_map.iter_rules()}


def test_clusters_page_shows_unassigned_group(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    for _ in range(10):
        image_id = _image_id(client.get("/al-labeling").get_data(as_text=True))
        client.post("/al-labeling", data={"image_id": image_id, "label": truth[image_id], "action": "label"})
    n = client.engine.status()["n_unassigned"]
    assert n == round(0.05 * (290 - 20))                       # etichettate e validation set esclusi
    html = client.get("/al-clusters?order=uncertain").get_data(as_text=True)
    assert "possible new class" in html and f"{n} bees far from every labeled example" in html
    assert "Unassigned (far from every label)" in client.get("/al-status").get_data(as_text=True)


def _label_first_batch(client, truth):
    done = []
    for _ in range(10):
        image_id = _image_id(client.get("/al-labeling").get_data(as_text=True))
        client.post("/al-labeling", data={"image_id": image_id, "label": truth[image_id], "action": "label"})
        done.append(image_id)
    return done


def test_clusters_page_edits_labels(client, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    done = _label_first_batch(client, truth)

    html = client.get("/al-clusters?view=labeled&n=2000").get_data(as_text=True)
    assert all(f'value="{i}"' in html for i in done) and "Remove label" in html
    predicted = client.get("/al-clusters").get_data(as_text=True)
    assert not any(f'value="{i}"' in predicted for i in done) and "Remove label" not in predicted

    # cambio classe a due api etichettate
    r = client.post("/al-clusters?view=labeled", data={"selected_images": done[:2], "label": "cambiata", "action": "label"})
    assert r.status_code == 302 and "changed=2" in r.headers["Location"]
    assert [client.engine.bank.label_of(i) for i in done[:2]] == ["Cambiata", "Cambiata"]
    assert "2 bee(s) updated" in client.get(r.headers["Location"]).get_data(as_text=True)

    # un gruppo di api predette, scelte con le checkbox
    group = re.findall(r'name="selected_images" value="([^"]+)"', client.get("/al-clusters?n=6").get_data(as_text=True))[:3]
    client.post("/al-clusters", data={"selected_images": group, "label": "Nuovo Gruppo", "action": "label"})
    assert all(client.engine.bank.label_of(i) == "Nuovo Gruppo" for i in group)
    assert {e.query_type for e in client.engine.store.events()[-3:]} == {"manual"}

    client.post("/al-clusters?view=labeled", data={"selected_images": done[2:3], "action": "remove"})
    client.post("/al-clusters?view=labeled", data={"selected_images": done[3:4], "action": "unknown"})
    assert client.engine.bank.label_of(done[2]) is None
    assert "Non so (set aside)" in client.get("/al-clusters?view=labeled").get_data(as_text=True)

    # errori: nessuna selezione, etichetta vuota
    for data in ({"label": "X", "action": "label"}, {"selected_images": done[4:5], "label": " ", "action": "label"}):
        r = client.post("/al-clusters?view=labeled", data=data)
        assert r.status_code == 200 and "color: var(--btn-red)" in r.get_data(as_text=True)
    assert client.engine.bank.label_of(done[4]) == truth[done[4]]
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
