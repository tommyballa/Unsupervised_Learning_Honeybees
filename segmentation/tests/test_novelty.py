"""Rare, never-labelled bees: novelty share in the batch and the "unassigned" group."""
from collections import Counter

import numpy as np
import pytest

from bee_al.config import QueryConfig, config_from_dict
from bee_al.embeddings import Embeddings
from bee_al.engine import ALEngine
from bee_al.knn import knn_predict, nearest_labeled_similarity
from bee_al.memory_bank import MemoryBank
from bee_al.query import NOVELTY, select_batch
from bee_al.simulation import make_synthetic


@pytest.fixture
def rare():
    """300 bees from 3 common classes plus 12 "sick" bees in an isolated region of the space."""
    X, y = make_synthetic(n=300, dim=32, n_classes=3, imbalance=0.7, noise=0.6, seed=3)
    rng = np.random.default_rng(0)
    center = rng.normal(size=32)
    sick = center / np.linalg.norm(center) + 0.6 * rng.normal(size=(12, 32)) / np.sqrt(32)
    X = np.concatenate([X, sick / np.linalg.norm(sick, axis=1, keepdims=True)]).astype(np.float32)
    y = list(y) + ["Malata"] * 12
    ids = [f"bee_{i:05d}.png" for i in range(len(X))]
    return X, y, ids


def _bank_without_rare(rare, n_labeled=80):
    X, y, ids = rare
    bank = MemoryBank(ids, X)
    for i in range(n_labeled):           # only common bees are labelled
        bank.add_label(ids[i], y[i])
    return bank


def test_nearest_labeled_similarity(rare):
    X, y, ids = rare
    bank = MemoryBank(ids, X)
    assert (nearest_labeled_similarity(bank) == 0).all()          # no labels
    bank.add_label(ids[0], y[0])
    sims = nearest_labeled_similarity(bank)
    assert sims[0] == 0                                           # a bee is not its own neighbour
    np.testing.assert_allclose(sims[1:], X[1:] @ X[0], atol=1e-5)


def test_unlabeled_rare_bees_look_confident_but_are_novel(rare):
    X, y, ids = rare
    bank = _bank_without_rare(rare)
    sick = np.arange(300, 312)
    assert "Malata" not in knn_predict(bank).classes               # without labels the class does not exist
    novelty = 1 - nearest_labeled_similarity(bank)
    assert novelty[sick].min() > np.percentile(novelty[:300], 95)  # but they are the farthest from everything


def test_novelty_quota_surfaces_rare_bees(rare):
    X, y, ids = rare
    bank = _bank_without_rare(rare)
    probs = knn_predict(bank).probs
    novelty = 1 - nearest_labeled_similarity(bank)
    cfg = QueryConfig(batch_size=20, cold_start_threshold=50, random_fraction=0.1, novelty_fraction=0.2,
                      candidate_multiplier=3)
    args = (X, bank.labeled_indices(), probs, bank.n_labeled, len(bank.classes), cfg)

    batch = select_batch(*args, np.random.default_rng(0), novelty=novelty)
    assert Counter(q.query_type for q in batch) == {"uncertainty": 14, "novelty": 4, "random": 2}
    assert len({q.index for q in batch}) == 20
    picked = [q.index for q in batch if q.query_type == NOVELTY]
    assert sum(i >= 300 for i in picked) >= 3                      # the sick bees reach the annotator

    without = select_batch(*args, np.random.default_rng(0))        # without scores the share goes to uncertainty
    assert Counter(q.query_type for q in without) == {"uncertainty": 18, "random": 2}

    cold = QueryConfig(batch_size=20, cold_start_threshold=500, novelty_fraction=0.2)
    batch = select_batch(X, bank.labeled_indices(), probs, bank.n_labeled, len(bank.classes), cold,
                         np.random.default_rng(0), novelty=novelty)
    assert NOVELTY not in {q.query_type for q in batch}            # cold start stays diversity + random


def test_engine_keeps_far_bees_out_of_clusters(rare, tmp_path):
    X, y, ids = rare
    truth = dict(zip(ids, y))
    cfg = config_from_dict({
        "query": {"batch_size": 10, "cold_start_threshold": 20},
        "validation": {"size": 20},
        "propagation": {"n_neighbors": 8},
        "knn": {"unassigned_fraction": 0.05},
    }, root=tmp_path)
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    assert engine.status()["n_unassigned"] == 0 and engine.unassigned_samples()["n"] == 0  # no labels

    labeled = 0
    while labeled < 40:                                            # the annotator does not recognise the sick bees: "Don't know"
        item = engine.next_image("a@x.it")
        engine.submit("a@x.it", item.image_id, "non so" if truth[item.image_id] == "Malata" else truth[item.image_id])
        labeled += 1
    engine.refresh()

    st = engine.status()
    free = len(ids) - st["n_labels"] - st["n_unknown"] - len(engine.validation_ids)
    assert st["n_unassigned"] == round(0.05 * free)
    group = engine.unassigned_samples(n=100, order="uncertain")
    shown = [b["image_id"] for b in group["samples"]]
    sick_left = [i for i in ids[300:] if engine.bank.label_of(i) is None and i not in engine.validation_ids]
    assert set(sick_left) <= set(shown)                            # the remaining sick bees are in the separate group
    sims = [b["similarity"] for b in group["samples"]]
    assert sims == sorted(sims) and all(b["nearest_class"] for b in group["samples"])

    in_clusters = {m["image_id"] for members in engine.final_clusters().values() for m in members}
    in_samples = {b["image_id"] for c in engine.cluster_samples(n=500) for b in c["samples"]}
    assert not set(shown) & in_clusters and not set(shown) & in_samples
    assert "novelty" in {b.query_type for b in engine.batch}

    cfg.knn.unassigned_fraction = 0.0                              # can be disabled from the config
    engine.refresh()
    assert engine.status()["n_unassigned"] == 0
