import csv

import pytest

from bee_al import REMOVED_LABEL, UNKNOWN_LABEL
from bee_al.hub_compat import HUB_APP_PATH, load_hub_helpers
from bee_al.label_store import FIELDS, LabelStore, read_legacy_labels


def _store(tmp_path, aliases=None):
    return LabelStore(tmp_path / "results" / "al_labels.csv", aliases)


def _write_legacy(results_dir, single_rows=(), grid_rows=()):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "single_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "user_email", "image_name", "label", "confidence"])
        w.writerows(single_rows)
    with open(results_dir / "grid_batch_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "user_email", "selected_images", "assigned_label"])
        w.writerows(grid_rows)


def test_hub_helpers_come_from_app_py():
    helpers = load_hub_helpers()
    assert helpers.clean_label("  bad   seg ") == "Bad Seg"
    assert helpers.parse_shifted_row_single.__code__.co_filename == str(HUB_APP_PATH)


def test_append_writes_all_columns_and_normalizes(tmp_path):
    store = _store(tmp_path)
    ev = store.append("a.png", "  healthy   bee ", "t@x.it", "High", "uncertainty", "cfg1", "v000", "r0001")
    assert ev.label == "Healthy Bee"
    with open(store.path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == FIELDS
    assert rows[0] == {
        "image_id": "a.png", "label": "Healthy Bee", "timestamp": ev.timestamp, "annotator": "t@x.it",
        "confidence": "High", "query_type": "uncertainty", "config_id": "cfg1",
        "model_version": "v000", "batch_id": "r0001",
    }
    with pytest.raises(ValueError):
        store.append("a.png", "X", query_type="boh")
    with pytest.raises(ValueError):
        store.append("a.png", "   ", query_type="random")


def test_last_event_wins_and_removed_is_a_removal(tmp_path):
    store = _store(tmp_path)
    store.append("a.png", "Sana", query_type="random")
    store.append("b.png", "Sana", query_type="random")
    store.append("a.png", "Varroa", query_type="random")       # correzione
    store.append("b.png", REMOVED_LABEL, query_type="random")  # rimozione
    current = store.current()
    assert {k: v.label for k, v in current.items()} == {"a.png": "Varroa"}
    assert len(store.events()) == 4  # lo storico resta tutto

    store.append("b.png", "non so", query_type="random")
    assert store.current()["b.png"].label == UNKNOWN_LABEL


def test_aliases_apply_on_read_without_rewriting_history(tmp_path):
    store = _store(tmp_path)
    store.append("a.png", "Bad Seg", query_type="random")
    store.append("b.png", "Bad_Seg", query_type="random")
    store.append("c.png", "Blurry", query_type="random")
    before = store.path.read_bytes()

    store.set_aliases({"bad seg": "Bad_Seg", "Blurry": "Sfocata", "Sfocata": "Scarto"})
    labels = {k: v.label for k, v in store.current().items()}
    assert labels == {"a.png": "Bad_Seg", "b.png": "Bad_Seg", "c.png": "Scarto"}  # unione + catena
    assert store.path.read_bytes() == before
    assert [e.label for e in store.events()] == ["Bad Seg", "Bad_Seg", "Blurry"]

    store.set_aliases({"A": "B", "B": "A"})  # un ciclo non blocca la lettura
    assert store.resolve("A") in {"A", "B"}
    store.set_aliases({"Blurry": REMOVED_LABEL})  # un alias può anche scartare una classe
    assert "c.png" not in store.current()


def test_legacy_import_with_shifted_rows_and_aliases(tmp_path):
    results = tmp_path / "results"
    _write_legacy(
        results,
        single_rows=[
            ["2026-09-22T11:20:04", "t@x.it", "bee_1.png", "bad_seg", "High"],
        ],
        grid_rows=[
            ["2026-09-22T11:24:01", "t@x.it", "bee_2.png;bee_3.png", "good low reflection"],
        ],
    )
    # righe scritte col vecchio disallineamento: l'email finisce nella colonna dell'immagine
    with open(results / "single_labels.csv", "a", newline="") as f:
        csv.writer(f).writerow(["2026-09-21T10:00:00", "old@x.it", "bee_4.png", "blurry"])
    with open(results / "grid_batch_labels.csv", "a", newline="") as f:
        csv.writer(f).writerow(["2026-09-21T10:05:00", "x", "old@x.it", "bee_5.png;bee_6.png", "blurry"])

    legacy = read_legacy_labels(results)
    assert set(legacy) == {f"bee_{i}.png" for i in range(1, 7)}
    assert legacy["bee_4.png"]["annotator"] == "old@x.it"

    store = _store(tmp_path, aliases={"Good Low Reflection": "Good"})
    assert store.import_legacy(results, config_id="cfg") == 6
    current = store.current()
    assert {k: v.label for k, v in current.items()} == {
        "bee_1.png": "Bad_Seg", "bee_2.png": "Good", "bee_3.png": "Good",
        "bee_4.png": "Blurry", "bee_5.png": "Blurry", "bee_6.png": "Blurry",
    }
    assert {e.query_type for e in store.events()} == {"legacy"}
    assert current["bee_1.png"].confidence == "High"
    assert current["bee_1.png"].timestamp == "2026-09-22T11:20:04"
    assert store.import_legacy(results) == 0  # idempotente


def test_legacy_import_never_overrides_al_labels_and_follows_clean(tmp_path):
    results = tmp_path / "results"
    _write_legacy(results, grid_rows=[["2026-09-22T11:24:01", "t@x.it", "bee_1.png;bee_2.png", "Sana"]])
    store = _store(tmp_path)
    store.append("bee_1.png", "Varroa", query_type="uncertainty")
    assert store.import_legacy(results) == 1
    assert store.current()["bee_1.png"].label == "Varroa"
    assert store.current()["bee_2.png"].query_type == "legacy"

    # /clean toglie bee_2 dal CSV legacy: nello store diventa un evento di rimozione
    _write_legacy(results, grid_rows=[["2026-09-22T11:24:01", "t@x.it", "bee_1.png", "Sana"]])
    assert store.import_legacy(results) == 1
    assert "bee_2.png" not in store.current()
    assert store.events()[-1].label == REMOVED_LABEL
    assert store.import_legacy(results) == 0


def test_no_legacy_files_is_a_noop(tmp_path):
    store = _store(tmp_path)
    assert read_legacy_labels(tmp_path / "results") is None
    assert store.import_legacy(tmp_path / "results") == 0
    assert store.events() == [] and store.current() == {}
