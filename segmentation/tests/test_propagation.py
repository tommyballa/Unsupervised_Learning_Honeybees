import numpy as np
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
