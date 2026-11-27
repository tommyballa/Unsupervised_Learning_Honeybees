import numpy as np
import pytest
import scipy.sparse as sp

from bee_al.memory_bank import MemoryBank
from bee_al.propagation import LabelSpreader, build_knn_graph


def _bank(synth, n_labeled=0):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    for i in range(n_labeled):
        bank.add_label(ids[i], y[i])
    return bank


def test_graph_is_sparse_symmetric_without_self_loops(synth):
    bank = _bank(synth)
    S = build_knn_graph(bank, n_neighbors=8)
    assert sp.issparse(S) and S.shape == (300, 300)
    assert abs(S - S.T).max() < 1e-12
    assert S.diagonal().sum() == 0
    assert S.nnz < 300 * 2 * 8 + 1  # al più 2k archi per nodo


def test_propagation_spreads_labels_from_few_examples(synth):
    X, y, ids = synth
    bank = _bank(synth)
    # un solo esempio per classe: la propagazione li estende lungo il grafo
    for cls in sorted(set(y)):
        i = int(np.flatnonzero(y == cls)[0])
        bank.add_label(ids[i], cls)
    spreader = LabelSpreader.from_bank(bank, n_neighbors=8)
    pred = spreader.propagate_bank(bank)
    assert pred.classes == sorted(set(y))
    reached = pred.probs.sum(axis=1) > 0
    np.testing.assert_allclose(pred.probs[reached].sum(axis=1), 1.0, atol=1e-9)
    assert np.mean(np.array(pred.labels)[reached] == y[reached]) > 0.85


def test_graph_is_reused_and_labeled_nodes_are_clamped(synth):
    X, y, ids = synth
    bank = _bank(synth, n_labeled=20)
    spreader = LabelSpreader.from_bank(bank, n_neighbors=8)
    graph = spreader.graph
    first = spreader.propagate_bank(bank)
    bank.add_label(ids[0], "Corretta")  # nuovo batch di etichette: si ricalcola, senza ricostruire il grafo
    second = spreader.propagate_bank(bank)
    assert spreader.graph is graph
    assert "Corretta" in second.classes and "Corretta" not in first.classes
    assert second.probs[0, second.classes.index("Corretta")] == 1.0


def test_zero_and_single_class(synth):
    X, y, ids = synth
    bank = _bank(synth)
    spreader = LabelSpreader.from_bank(bank, n_neighbors=8)
    assert spreader.propagate_bank(bank).probs.shape == (300, 0)
    bank.add_label(ids[0], "Solo")
    pred = spreader.propagate_bank(bank)
    assert pred.classes == ["Solo"]
    assert set(np.unique(pred.probs)) <= {0.0, 1.0}


def test_unreached_nodes_have_zero_rows():
    # due componenti scollegate: l'etichetta non può attraversarle
    X = np.zeros((6, 4), dtype=np.float32)
    X[:3, 0] = 1.0
    X[3:, 1] = 1.0
    X[:3, 2] = [0.0, 0.01, 0.02]
    X[3:, 3] = [0.0, 0.01, 0.02]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    bank = MemoryBank([str(i) for i in range(6)], X, backend="numpy")
    bank.add_label("0", "A")
    pred = LabelSpreader.from_bank(bank, n_neighbors=2).propagate_bank(bank)
    assert (pred.probs[:3, 0] == 1.0).all()
    assert (pred.probs[3:].sum(axis=1) == 0).all()


def test_graph_cache_is_reused_and_tied_to_the_embeddings(synth, tmp_path, monkeypatch):
    from bee_al import propagation

    bank = _bank(synth)
    first = LabelSpreader.from_bank(bank, n_neighbors=8, cache_dir=tmp_path).graph
    assert len(list(tmp_path.glob("knn_graph_*_k8.npz"))) == 1

    monkeypatch.setattr(propagation, "build_knn_graph", lambda *a, **k: pytest.fail("il grafo andava letto dalla cache"))
    again = LabelSpreader.from_bank(bank, n_neighbors=8, cache_dir=tmp_path).graph
    assert abs(first - again).max() < 1e-12
    monkeypatch.undo()

    X, _, ids = synth
    other = MemoryBank(ids, np.roll(X, 1, axis=1))               # embedding diversi (es. dopo un re-train): file nuovo
    LabelSpreader.from_bank(other, n_neighbors=8, cache_dir=tmp_path)
    LabelSpreader.from_bank(bank, n_neighbors=5, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("knn_graph_*.npz"))) == 3

    broken = next(tmp_path.glob("knn_graph_*_k8.npz"))
    for f in tmp_path.glob("knn_graph_*_k8.npz"):
        f.write_bytes(b"rotto")                                   # cache rovinata: si ricostruisce senza errori
    assert LabelSpreader.from_bank(bank, n_neighbors=8, cache_dir=tmp_path).graph.shape == (300, 300)
    assert broken.stat().st_size > 100


def test_graph_built_in_subprocess_is_reused_from_cache(tmp_path, monkeypatch):
    """L'hub non costruisce il grafo: lo trova in cache, preparato da un processo separato."""
    from bee_al import propagation
    from bee_al.embeddings import l2_normalize
    from bee_al.memory_bank import MemoryBank

    X = l2_normalize(np.random.default_rng(0).normal(size=(60, 8)))
    path = propagation.build_graph_cache_in_subprocess(X, 5, tmp_path, backend="numpy")
    assert path.exists() and path == propagation.graph_cache_path(X, 5, tmp_path)

    def no_build(*args, **kwargs):
        raise AssertionError("il grafo doveva arrivare dalla cache")

    monkeypatch.setattr(propagation, "build_knn_graph", no_build)
    bank = MemoryBank([f"b{i}" for i in range(len(X))], X, backend="numpy")
    graph = propagation.load_or_build_graph(bank, 5, tmp_path)
    assert graph.shape == (60, 60) and graph.nnz > 0


def test_saved_retrain_embeddings_keep_the_graph_cache_key(tmp_path):
    """Gli embedding di un re-train ricaricati dopo un riavvio devono ritrovare lo stesso grafo in cache."""
    from bee_al.embeddings import l2_normalize, load_embeddings, save_embeddings
    from bee_al.propagation import graph_cache_path

    X = l2_normalize(np.random.default_rng(1).normal(size=(500, 32)))
    ids = [f"b{i}" for i in range(len(X))]
    save_embeddings(tmp_path / "emb", ids, ids, X, normalize=False)
    assert graph_cache_path(load_embeddings(tmp_path / "emb").X, 15, tmp_path) == graph_cache_path(X, 15, tmp_path)
