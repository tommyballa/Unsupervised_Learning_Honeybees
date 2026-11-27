import time

import numpy as np
import pytest

from bee_al import REMOVED_LABEL, UNKNOWN_LABEL
from bee_al import memory_bank as mb
from bee_al.knn import knn_predict
from bee_al.memory_bank import MemoryBank, build_index, faiss_available, resolve_backend

BACKENDS = [
    pytest.param("faiss", marks=pytest.mark.skipif(not faiss_available(), reason="faiss not installed")),
    "sklearn",
    "numpy",
]


def _brute(X, Q, k):
    sims = Q @ X.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return np.take_along_axis(sims, idx, axis=1), idx


@pytest.mark.parametrize("backend", BACKENDS)
def test_neighbor_search_matches_brute_force(synth, backend):
    X, _, _ = synth
    sims, idx = build_index(X, backend).search(X[:20], 5)
    ref_sims, ref_idx = _brute(X, X[:20], 5)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(sims, ref_sims, atol=1e-5)
    assert (idx[:, 0] == np.arange(20)).all()  # the first neighbour is the point itself


@pytest.mark.parametrize("backend", BACKENDS)
def test_k_larger_than_index_is_clipped(synth, backend):
    X, _, _ = synth
    sims, idx = build_index(X[:3], backend).search(X[:2], 10)
    assert sims.shape == idx.shape == (2, 3)


def test_auto_backend_with_and_without_faiss(synth, monkeypatch):
    X, _, ids = synth
    monkeypatch.setattr(mb, "faiss_available", lambda: False)
    assert resolve_backend("auto") == "sklearn"
    with pytest.raises(ImportError):
        resolve_backend("faiss")
    bank = MemoryBank(ids, X)  # the bank works even without FAISS
    assert bank.backend == "sklearn"
    assert bank.search(X[:1], 3)[1][0, 0] == 0
    monkeypatch.undo()
    if faiss_available():
        assert MemoryBank(ids, X).backend == "faiss"


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_label_updates_knn_immediately(synth, backend):
    X, y, ids = synth
    bank = MemoryBank(ids, X, backend=backend)
    assert bank.classes == [] and bank.n_labeled == 0
    assert knn_predict(bank).probs.shape == (len(X), 0)  # no labels, no classes

    for i in range(60):
        bank.add_label(ids[i], y[i])
    pred = knn_predict(bank, np.arange(60, 300), k=5)
    assert pred.classes == sorted(set(y[:60]))
    np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-9)
    acc = np.mean(np.array(pred.labels) == y[60:])
    assert acc > 0.9

    # correction: the new label replaces the old one, without training
    bank.add_label(ids[0], "Altro")
    assert bank.label_of(ids[0]) == "Altro" and "Altro" in bank.classes
    assert bank.n_labeled == 60


def test_add_label_is_fast(synth):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    t0 = time.perf_counter()
    for i in range(200):
        bank.add_label(ids[i], y[i])
    assert (time.perf_counter() - t0) / 200 < 0.005  # a few milliseconds per label


def test_unknown_and_removed_are_not_classes(synth):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    bank.add_label(ids[0], y[0])
    bank.add_label(ids[1], UNKNOWN_LABEL)
    assert bank.classes == [y[0]]
    assert bank.label_of(ids[1]) == UNKNOWN_LABEL
    assert bank.unknown_indices().tolist() == [1]
    _, nb = bank.search_labeled(X[:5], 3)
    assert set(nb.ravel()) == {0}  # "Don't know" does not enter the kNN

    bank.add_label(ids[0], REMOVED_LABEL)
    assert bank.label_of(ids[0]) is None and bank.classes == []
    bank.add_label(ids[1], y[1])  # a "Don't know" can be corrected into a real class
    assert bank.unknown_indices().size == 0 and bank.classes == [y[1]]
    with pytest.raises(KeyError):
        bank.add_label("non_esiste.png", "X")


def test_labeled_bee_does_not_vote_for_itself(synth):
    X, _, ids = synth
    bank = MemoryBank(ids, X)
    bank.add_label(ids[0], "A")
    bank.add_label(ids[1], "B")
    pred = knn_predict(bank, [0], k=1)
    assert pred.labels == ["B"]


def test_save_load_roundtrip(synth, tmp_path):
    X, y, ids = synth
    bank = MemoryBank(ids, X, paths=[f"/d/{i}" for i in ids])
    for i in range(30):
        bank.add_label(ids[i], y[i])
    bank.add_label(ids[30], UNKNOWN_LABEL)
    bank.save(tmp_path / "bank")

    loaded = MemoryBank.load(tmp_path / "bank")
    assert loaded.ids == bank.ids and loaded.paths == bank.paths
    np.testing.assert_array_equal(loaded.X, bank.X)
    assert loaded.class_counts() == bank.class_counts()
    assert loaded.label_of(ids[30]) == UNKNOWN_LABEL
    np.testing.assert_allclose(knn_predict(loaded).probs, knn_predict(bank).probs)
