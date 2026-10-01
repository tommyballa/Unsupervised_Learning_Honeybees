from collections import Counter

import numpy as np
import pytest

from bee_al.config import QueryConfig
from bee_al.knn import knn_predict
from bee_al.memory_bank import MemoryBank
from bee_al.query import DIVERSITY, RANDOM, UNCERTAINTY, is_cold_start, select_batch
from bee_al.uncertainty import uncertainty_scores


def _labeled_bank(synth, n_labeled):
    X, y, ids = synth
    bank = MemoryBank(ids, X)
    for i in range(n_labeled):
        bank.add_label(ids[i], y[i])
    return bank


def _select(bank, cfg, excluded, probs=None, seed=0):
    return select_batch(bank.X, excluded, probs, bank.n_labeled, len(bank.classes), cfg, np.random.default_rng(seed))


def test_cold_start_uses_diversity_plus_random_quota(synth):
    bank = _labeled_bank(synth, 0)
    cfg = QueryConfig(batch_size=30, cold_start_threshold=50, random_fraction=0.1)
    batch = _select(bank, cfg, excluded=[])
    types = Counter(q.query_type for q in batch)
    assert types == {DIVERSITY: 27, RANDOM: 3}
    assert len({q.index for q in batch}) == 30  # nessun duplicato


def test_cold_start_rules():
    cfg = QueryConfig(cold_start_threshold=50)
    assert is_cold_start(49, 5, cfg)        # poche etichette
    assert is_cold_start(500, 1, cfg)       # una sola classe: si resta in cold start
    assert is_cold_start(0, 0, cfg)
    assert not is_cold_start(50, 2, cfg)


def test_cold_start_respected_even_with_probs(synth):
    bank = _labeled_bank(synth, 20)
    cfg = QueryConfig(batch_size=10, cold_start_threshold=50)
    batch = _select(bank, cfg, bank.labeled_indices(), probs=knn_predict(bank).probs)
    assert UNCERTAINTY not in {q.query_type for q in batch}

    one_class = MemoryBank(bank.ids, bank.X)
    for i in range(60):
        one_class.add_label(bank.ids[i], "Solo")
    batch = _select(one_class, cfg, one_class.labeled_indices(), probs=knn_predict(one_class).probs)
    assert UNCERTAINTY not in {q.query_type for q in batch}


def test_normal_regime_uncertainty_with_diversity_not_top_n(synth):
    bank = _labeled_bank(synth, 60)
    cfg = QueryConfig(batch_size=20, cold_start_threshold=50, random_fraction=0.1, candidate_multiplier=5)
    probs = knn_predict(bank).probs
    excluded = bank.labeled_indices()
    batch = _select(bank, cfg, excluded, probs)
    types = Counter(q.query_type for q in batch)
    assert types == {UNCERTAINTY: 18, RANDOM: 2}

    picked = {q.index for q in batch if q.query_type == UNCERTAINTY}
    unc = uncertainty_scores(probs, cfg.uncertainty)
    unc[excluded] = -1
    ranking = np.argsort(-unc)
    assert picked != set(ranking[:18].tolist())                 # mai le top-N in blocco
    random_pick = {q.index for q in batch if q.query_type == RANDOM}
    pool = [i for i in ranking if i not in random_pick][:5 * 18]
    assert unc[sorted(picked)].min() >= unc[pool].min() - 1e-12  # ma sempre dentro le top-M


@pytest.mark.parametrize("n_labeled", [0, 80])
def test_never_proposes_excluded_bees(synth, n_labeled):
    bank = _labeled_bank(synth, n_labeled)
    cfg = QueryConfig(batch_size=30, cold_start_threshold=50)
    labeled = set(bank.labeled_indices().tolist())
    unknown = set(range(100, 130))       # "Non so"
    validation = set(range(130, 180))
    current_batch = set(range(180, 200))
    excluded = labeled | unknown | validation | current_batch
    for seed in range(5):
        batch = _select(bank, cfg, excluded, knn_predict(bank).probs, seed=seed)
        idx = [q.index for q in batch]
        assert len(idx) == len(set(idx)) == 30
        assert not set(idx) & excluded


@pytest.mark.parametrize("fraction,expected", [(0.0, 0), (0.1, 3), (0.5, 15), (1.0, 30)])
def test_random_quota(synth, fraction, expected):
    bank = _labeled_bank(synth, 80)
    cfg = QueryConfig(batch_size=30, cold_start_threshold=50, random_fraction=fraction)
    batch = _select(bank, cfg, bank.labeled_indices(), knn_predict(bank).probs)
    assert sum(q.query_type == RANDOM for q in batch) == expected
    assert len(batch) == 30


def test_small_pool_and_determinism(synth):
    bank = _labeled_bank(synth, 80)
    cfg = QueryConfig(batch_size=30, cold_start_threshold=50)
    probs = knn_predict(bank).probs
    excluded = list(range(293))  # restano 7 api
    batch = _select(bank, cfg, excluded, probs)
    assert sorted(q.index for q in batch) == list(range(293, 300))
    assert _select(bank, cfg, range(300), probs) == []

    a = _select(bank, cfg, bank.labeled_indices(), probs, seed=7)
    b = _select(bank, cfg, bank.labeled_indices(), probs, seed=7)
    assert a == b  # stesso seed, stesso batch


@pytest.mark.parametrize("kind", ["margin", "entropy", "least_confidence"])
def test_uncertainty_measures_are_interchangeable(synth, kind):
    bank = _labeled_bank(synth, 80)
    cfg = QueryConfig(batch_size=10, cold_start_threshold=50, uncertainty=kind)
    batch = _select(bank, cfg, bank.labeled_indices(), knn_predict(bank).probs)
    assert len(batch) == 10
    probs = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.4, 0.3, 0.3]])
    scores = uncertainty_scores(probs, kind)
    assert scores[1] == pytest.approx(0.0, abs=1e-9) and scores.max() <= 1.0
    assert scores[0] > scores[1] and scores[2] > scores[1]
