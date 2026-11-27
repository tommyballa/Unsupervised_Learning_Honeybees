import json
import threading

import numpy as np
import pytest
import yaml

from bee_al.config import RetrainConfig
from bee_al.embeddings import Embeddings, load_embeddings
from bee_al.engine import ALEngine
from bee_al.retrain import (
    FeatureExtractorTrainer,
    FrozenExtractor,
    RetrainManager,
    RetrainPolicy,
    StubTrainer,
    latest_embeddings,
    list_versions,
    next_version,
)


def test_label_count_schedule_grows():
    policy = RetrainPolicy(RetrainConfig(initial_labels=50, growth_factor=2.0))
    assert [policy.label_threshold(i) for i in range(4)] == [50, 100, 200, 400]
    assert policy.decide(49, 0, []) is None
    assert policy.decide(50, 0, []) is not None
    assert policy.decide(99, 1, []) is None and policy.decide(100, 1, []) is not None


def test_plateau_policy():
    policy = RetrainPolicy(RetrainConfig(policy="plateau", plateau_rounds=3))
    assert policy.decide(10, 0, [0.5, 0.6, 0.7, 0.8]) is None            # still improving
    assert policy.decide(10, 0, [0.5, 0.8, 0.8, 0.79, 0.8]) is not None  # flat for 3 rounds
    assert policy.decide(10, 0, [None, 0.8, None]) is None               # too few values
    assert policy.decide(1000, 0, [0.5, 0.6]) is None                    # the count does not matter in this policy
    assert policy.decide(0, 0, [0.8, 0.8, 0.8, 0.8]) is None             # not needed without new labels
    either = RetrainPolicy(RetrainConfig(policy="either", initial_labels=50))
    assert either.decide(60, 0, []) and either.decide(10, 0, [0.8, 0.8, 0.8, 0.8])


def test_versions(cfg, tmp_path):
    assert list_versions(cfg.models_dir) == [] and next_version(cfg.models_dir) == "v001"
    assert latest_embeddings(cfg) == ("v000", cfg.embeddings_dir)
    (cfg.models_dir / "v001" / "embeddings").mkdir(parents=True)
    (cfg.models_dir / "v002").mkdir()                                    # incomplete: no embeddings
    (cfg.models_dir / "v001" / "embeddings" / "manifest.json").write_text("{}")
    assert list_versions(cfg.models_dir) == ["v001"] and next_version(cfg.models_dir) == "v003"
    assert latest_embeddings(cfg)[0] == "v001"


class RotatingTrainer(FeatureExtractorTrainer):
    """Fake trainer that actually changes the embeddings and waits for a go-ahead."""

    def __init__(self):
        self.gate = threading.Event()
        self.calls = []

    def retrain(self, labeled, unlabeled):
        self.calls.append((len(labeled.ids), len(unlabeled.ids), sorted(set(labeled.labels))))
        self.gate.wait(5)
        paths = labeled.paths + unlabeled.paths
        X = np.concatenate([labeled.X, unlabeled.X])
        return FrozenExtractor(paths, np.roll(X, 1, axis=1) * 3.0)


def _engine(cfg, synth):
    X, y, ids = synth
    cfg.retrain.enabled = True
    cfg.retrain.initial_labels = 20
    cfg.retrain.trainer = "stub"
    engine = ALEngine(cfg, Embeddings(ids, [f"/d/{i}" for i in ids], X))
    return engine, dict(zip(ids, y))


def _label(engine, truth, n):
    for _ in range(n):
        item = engine.next_image("a@x.it")
        engine.submit("a@x.it", item.image_id, truth[item.image_id])


def test_retrain_runs_in_background_and_versions_the_model(cfg, synth):
    engine, truth = _engine(cfg, synth)
    trainer = RotatingTrainer()
    manager = RetrainManager.attach(engine, trainer)
    old_X = engine.bank.X.copy()

    _label(engine, truth, 10)
    assert manager.state == "idle" and manager.status()["new_labels"] == 10
    _label(engine, truth, 10)                      # 20 labels: the job starts at the end of the batch
    assert manager.state == "running" and "20 new labels" in manager.reason

    # the hub is not blocked: labelling continues with the old model
    _label(engine, truth, 5)
    assert engine.model_version == "v000" and engine.status()["n_labels"] == 25
    assert trainer.calls == [(20, 280, sorted(set(engine.bank.classes)))]

    trainer.gate.set()
    manager.join(10)
    assert manager.state == "idle" and manager.error == ""
    assert engine.model_version == "v001"
    assert engine.status()["n_labels"] == 25       # labels given during the job are not lost
    assert not np.allclose(engine.bank.X, old_X)   # bank rebuilt on the new embeddings
    np.testing.assert_allclose(np.linalg.norm(engine.bank.X, axis=1), 1.0, atol=1e-5)
    assert engine.history[-1]["model_version"] == "v001"   # uncertainties and metrics recomputed

    out = cfg.models_dir / "v001"
    assert yaml.safe_load((out / "config.yaml").read_text())["query"]["batch_size"] == 10
    assert (out / "weights.json").exists()
    assert json.loads((out / "meta.json").read_text())["n_labels"] == 20
    np.testing.assert_allclose(load_embeddings(out / "embeddings").X, engine.bank.X, atol=1e-6)

    # next threshold doubled: 40 new labels are needed on top of the 20 of the previous job
    st = manager.status()
    assert st["n_retrains"] == 1 and st["next_threshold"] == 40 and st["new_labels"] == 5
    labeled = {i for i in engine.bank.ids if engine.bank.label_of(i)}
    assert not labeled & {b.image_id for b in engine.batch if not b.done}

    # on restart it resumes from the latest version
    assert latest_embeddings(cfg) == ("v001", out / "embeddings")
    restarted = RetrainManager(engine)
    assert restarted.n_retrains == 1 and restarted.labels_at_last == 20


def test_stub_trainer_and_failure_do_not_break_the_hub(cfg, synth):
    engine, truth = _engine(cfg, synth)
    manager = RetrainManager.attach(engine)        # StubTrainer by default
    assert isinstance(manager.trainer, StubTrainer)
    old_X = engine.bank.X.copy()
    _label(engine, truth, 20)
    manager.join(10)
    assert engine.model_version == "v001"
    np.testing.assert_allclose(engine.bank.X, old_X, atol=1e-6)  # the stub returns the frozen extractor

    class Broken(FeatureExtractorTrainer):
        def retrain(self, labeled, unlabeled):
            raise RuntimeError("GPU out of memory")

    manager.trainer = Broken()
    manager.start("manual")
    manager.join(10)
    assert manager.state == "failed" and "GPU out of memory" in manager.status()["error"]
    assert engine.model_version == "v001"
    _label(engine, truth, 3)                       # the hub keeps working


def test_automatic_jobs_are_off_by_default_but_manual_start_works(cfg, synth):
    X, y, ids = synth
    truth = dict(zip(ids, y))
    cfg.retrain.trainer = "stub"
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    manager = RetrainManager.attach(engine)
    assert engine.status()["retrain"]["auto"] is False

    with pytest.raises(ValueError, match="at least 20 labels"):      # too few labels
        manager.start()
    assert "20 labels" in manager.status()["blocked_by"]
    _label(engine, truth, 60)
    assert manager.state == "idle" and manager.n_retrains == 0        # no job started by itself
    assert manager.status()["blocked_by"] is None

    manager.start("button, a@x.it")
    manager.join(10)
    assert engine.model_version == "v001" and manager.status()["last"]["reason"] == "button, a@x.it"


def test_pseudo_labels_are_confident_predictions_only(cfg, synth):
    engine, truth = _engine(cfg, synth)
    cfg.retrain.enabled = False
    cfg.retrain.pseudo_label_threshold = 0.9
    trainer = RotatingTrainer()
    trainer.gate.set()
    seen = {}

    def capture(labeled, unlabeled):
        seen["labeled"], seen["unlabeled"] = labeled, unlabeled
        return RotatingTrainer.retrain(trainer, labeled, unlabeled)

    manager = RetrainManager.attach(engine, trainer)
    trainer.retrain = capture
    _label(engine, truth, 30)
    item = engine.next_image("a@x.it")
    engine.submit("a@x.it", item.image_id, "non so")
    engine.refresh()
    confidence = dict(zip(engine.bank.ids, engine.final_pred.confidence))
    unassigned = {engine.bank.ids[i] for i in np.flatnonzero(engine.unassigned)}
    manager.start()
    manager.join(10)

    unl = seen["unlabeled"]
    assert len(unl.pseudo_labels) == len(unl.ids) and seen["labeled"].pseudo_labels == []
    pseudo = {i: p for i, p in zip(unl.ids, unl.pseudo_labels) if p is not None}
    assert 0 < len(pseudo) < len(unl.ids)
    assert all(confidence[i] >= 0.9 for i in pseudo)
    assert item.image_id in unl.ids and item.image_id not in pseudo   # a "Non so" (unknown) never becomes a pseudo-label
    assert not set(pseudo) & unassigned                               # nor do bees far from everything
    right = np.mean([truth[i] == p for i, p in pseudo.items()])
    assert right > 0.9 and manager.status()["last"]["n_pseudo"] == len(pseudo)


class NoiseTrainer(FeatureExtractorTrainer):
    """Trainer that ruins the embeddings: the version must be rejected."""

    def retrain(self, labeled, unlabeled):
        paths = labeled.paths + unlabeled.paths
        return FrozenExtractor(paths, np.random.default_rng(0).normal(size=(len(paths), 32)))


def _label_validation(engine, truth):
    while True:
        item = engine.next_image("a@x.it", mode="validation")
        if item is None:
            break
        engine.submit("a@x.it", item.image_id, truth[item.image_id], mode="validation")


def test_worse_model_is_saved_but_not_used(cfg, synth):
    engine, truth = _engine(cfg, synth)
    cfg.retrain.enabled = False
    manager = RetrainManager.attach(engine, NoiseTrainer())
    _label(engine, truth, 30)
    _label_validation(engine, truth)
    old_X = engine.bank.X.copy()

    manager.start()
    manager.join(20)
    last = manager.status()["last"]
    assert manager.state == "idle" and last["accepted"] is False
    assert last["validation_after"]["macro_f1"] < last["validation_before"]["macro_f1"]
    assert engine.model_version == "v000"                              # stays on the previous model
    np.testing.assert_array_equal(engine.bank.X, old_X)
    assert (cfg.models_dir / "v001" / "embeddings" / "manifest.json").exists()   # but stays on disk
    assert list_versions(cfg.models_dir) == [] and list_versions(cfg.models_dir, include_rejected=True) == ["v001"]
    assert latest_embeddings(cfg)[0] == "v000" and next_version(cfg.models_dir) == "v002"
    assert RetrainManager(engine).n_retrains == 1                      # the attempt counts for the next threshold

    cfg.retrain.require_improvement = False                            # the check can be disabled
    manager.start()
    manager.join(20)
    assert engine.model_version == "v002" and manager.status()["last"]["accepted"] is True


def test_better_or_unverifiable_model_is_used(cfg, synth):
    engine, truth = _engine(cfg, synth)
    cfg.retrain.enabled = False
    manager = RetrainManager.attach(engine)                            # stub: same embeddings, same metric
    _label(engine, truth, 30)
    manager.start()
    manager.join(20)
    last = manager.status()["last"]
    assert last["accepted"] and last["validation_before"] is None      # without validation it cannot be verified
    _label_validation(engine, truth)
    manager.start()
    manager.join(20)
    last = manager.status()["last"]
    assert last["accepted"] and last["validation_after"] == last["validation_before"]
    assert engine.model_version == "v002"
