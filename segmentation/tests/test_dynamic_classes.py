"""The class set is dynamic: it comes from the labels and can even have a single class."""
import numpy as np

from bee_al.knn import knn_predict
from bee_al.memory_bank import MemoryBank
from bee_al.uncertainty import uncertainty_scores


def test_single_class_gives_probability_one(synth):
    X, _, ids = synth
    bank = MemoryBank(ids, X)
    bank.add_label(ids[0], "Solo")
    pred = knn_predict(bank)
    assert pred.classes == ["Solo"] and pred.probs.shape == (len(X), 1)
    assert set(pred.labels) == {"Solo"}
    # the only labelled bee has no other labelled neighbours: it gets no vote, but nothing breaks
    assert np.isfinite(pred.probs).all()
    for name in ("margin", "entropy", "least_confidence"):
        assert np.isfinite(uncertainty_scores(pred.probs, name)).all()


def test_new_class_at_runtime_becomes_a_knn_class(synth):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    for i in range(40):
        bank.add_label(ids[i], y[i])
    before = knn_predict(bank)
    assert "Nuova Classe" not in before.classes

    # the annotator types a never-seen class on a bee: it immediately becomes a kNN class
    target = 200
    bank.add_label(ids[target], "Nuova Classe")
    after = knn_predict(bank)
    assert "Nuova Classe" in after.classes
    assert after.probs.shape[1] == before.probs.shape[1] + 1
    nearest = bank.search(X[target], 2)[1][0, 1]  # the nearest neighbour of the new bee
    col = after.classes.index("Nuova Classe")
    assert after.probs[nearest, col] > 0


def test_class_with_one_example_and_renamed_class(synth):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    for i in range(40):
        bank.add_label(ids[i], y[i])
    bank.add_label(ids[40], "Rara")
    assert bank.class_counts()["Rara"] == 1
    assert np.isfinite(knn_predict(bank).probs).all()

    # removing the only example makes the class disappear
    bank.remove_label(ids[40])
    assert "Rara" not in knn_predict(bank).classes
