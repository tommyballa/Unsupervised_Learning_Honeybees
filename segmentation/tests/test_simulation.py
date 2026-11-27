import csv
import json

import numpy as np
import pytest

from bee_al.config import config_from_dict
from bee_al.simulation import (
    labels_to_threshold,
    make_synthetic,
    oracle_subset,
    run_simulation,
    run_single,
    save_results,
    variant_config,
)


@pytest.fixture
def sim_cfg(tmp_path):
    return config_from_dict(
        {
            "query": {"batch_size": 10, "cold_start_threshold": 20},
            "validation": {"size": 60},
            "propagation": {"n_neighbors": 8},
            "simulation": {"seeds": [0, 1], "max_labels": 60, "threshold": 0.8},
        },
        root=tmp_path,
    )


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n=400, dim=32, n_classes=4, imbalance=0.5, noise=0.6, seed=1)


def test_synthetic_is_normalized_and_imbalanced():
    X, y = make_synthetic(n=500, dim=16, n_classes=5, imbalance=0.5, seed=0)
    assert X.shape == (500, 16) and X.dtype == np.float32 and len(y) == 500
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    counts = sorted((y.count(c) for c in set(y)), reverse=True)
    assert len(counts) == 5 and counts[0] > 4 * counts[-1]


def test_variants_differ_only_in_query_and_semi_supervision(sim_cfg):
    base, v1, v2 = (variant_config(sim_cfg, v) for v in ("baseline", "v1", "v2"))
    assert base.query.random_fraction == 1.0 and not base.propagation.enabled
    assert v1.query.uncertainty_source == "knn" and not v1.propagation.enabled
    assert v2.propagation.enabled and v2.query.candidate_multiplier == sim_cfg.query.candidate_multiplier
    assert sim_cfg.query.random_fraction == pytest.approx(0.1)  # the starting config is not touched
    with pytest.raises(NotImplementedError):
        variant_config(sim_cfg, "v3")


def test_single_run_produces_a_learning_curve(sim_cfg, data):
    X, y = data
    rows = run_single(X, y, "v2", sim_cfg, seed=0)
    assert [r["n_labels"] for r in rows] == [10, 20, 30, 40, 50, 60]
    for r in rows:
        assert all(0.0 <= r[m] <= 1.0 for m in ("accuracy", "macro_f1", "nmi"))
    assert rows[-1]["accuracy"] > 0.85 and rows[-1]["accuracy"] >= rows[0]["accuracy"]
    assert rows == run_single(X, y, "v2", sim_cfg, seed=0)       # reproducible
    assert rows != run_single(X, y, "v2", sim_cfg, seed=1)


def test_full_simulation_outputs(sim_cfg, data, tmp_path):
    X, y = data
    logs = []
    result = run_simulation(X, y, sim_cfg, variants=("baseline", "v1", "v2", "v3"), log=logs.append)
    summary = result["summary"]
    assert set(summary["variants"]) == {"baseline", "v1", "v2", "v3"}
    assert summary["variants"]["v3"]["implemented"] is False and any("v3" in line for line in logs)
    assert len(result["rows"]) == 3 * 2 * 6                       # variants × seeds × rounds

    for name in ("baseline", "v1", "v2"):
        v = summary["variants"][name]
        assert [p["n_labels"] for p in v["curve"]] == [10, 20, 30, 40, 50, 60]
        assert all(p["accuracy_std"] >= 0 for p in v["curve"])
        assert len(v["labels_to_threshold"]["per_seed"]) == 2

    paths = save_results(result, tmp_path / "out")
    assert paths["png"].stat().st_size > 10_000
    with open(paths["csv"]) as f:
        assert len(list(csv.DictReader(f))) == 36
    assert json.loads(paths["json"].read_text())["threshold"] == 0.8


def test_labels_to_threshold():
    rows = [{"n_labels": 10, "macro_f1": 0.5}, {"n_labels": 20, "macro_f1": 0.85}, {"n_labels": 30, "macro_f1": 0.7}]
    assert labels_to_threshold(rows, "macro_f1", 0.8) == 20
    assert labels_to_threshold(rows, "macro_f1", 0.95) is None


def test_any_label_set_works_as_oracle(sim_cfg, data):
    X, y = data
    # arbitrary oracle: free-form names, a class with very few examples, partial coverage of the bees
    ids = [f"bee_{i}.png" for i in range(len(X))]
    labels = {ids[i]: ("Rare" if i < 3 else f"free label {y[i][-1]}") for i in range(0, len(X), 2)}
    Xo, yo = oracle_subset(ids, X, labels)
    assert len(Xo) == len(yo) == 200 and "Rare" in yo
    rows = run_single(Xo, yo, "v2", sim_cfg, seed=0)
    assert rows and rows[-1]["n_labels"] == 60

    one_class = run_single(Xo, ["Only"] * 200, "v1", sim_cfg, seed=0)  # even with a single class
    assert one_class[-1]["accuracy"] == 1.0 and one_class[-1]["n_classes"] == 1
