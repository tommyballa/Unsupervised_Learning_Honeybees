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
    assert policy.decide(10, 0, [0.5, 0.6, 0.7, 0.8]) is None            # sta ancora migliorando
    assert policy.decide(10, 0, [0.5, 0.8, 0.8, 0.79, 0.8]) is not None  # fermo da 3 round
    assert policy.decide(10, 0, [None, 0.8, None]) is None               # troppo pochi valori
    assert policy.decide(1000, 0, [0.5, 0.6]) is None                    # il conteggio non conta in questa policy
    assert policy.decide(0, 0, [0.8, 0.8, 0.8, 0.8]) is None             # senza etichette nuove non serve
    either = RetrainPolicy(RetrainConfig(policy="either", initial_labels=50))
    assert either.decide(60, 0, []) and either.decide(10, 0, [0.8, 0.8, 0.8, 0.8])


def test_versions(cfg, tmp_path):
    assert list_versions(cfg.models_dir) == [] and next_version(cfg.models_dir) == "v001"
    assert latest_embeddings(cfg) == ("v000", cfg.embeddings_dir)
    (cfg.models_dir / "v001" / "embeddings").mkdir(parents=True)
    (cfg.models_dir / "v002").mkdir()                                    # incompleta: senza embedding
    (cfg.models_dir / "v001" / "embeddings" / "manifest.json").write_text("{}")
    assert list_versions(cfg.models_dir) == ["v001"] and next_version(cfg.models_dir) == "v003"
    assert latest_embeddings(cfg)[0] == "v001"


class RotatingTrainer(FeatureExtractorTrainer):
    """Trainer finto che cambia davvero gli embedding e aspetta un via libera."""

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
    _label(engine, truth, 10)                      # 20 etichette: a fine batch parte il job
    assert manager.state == "running" and "20 nuove etichette" in manager.reason

    # l'hub non è bloccato: si continua a etichettare con il modello vecchio
    _label(engine, truth, 5)
    assert engine.model_version == "v000" and engine.status()["n_labels"] == 25
    assert trainer.calls == [(20, 280, sorted(set(engine.bank.classes)))]

    trainer.gate.set()
    manager.join(10)
    assert manager.state == "idle" and manager.error == ""
    assert engine.model_version == "v001"
    assert engine.status()["n_labels"] == 25       # le etichette date durante il job non si perdono
    assert not np.allclose(engine.bank.X, old_X)   # bank ricostruita sugli embedding nuovi
    np.testing.assert_allclose(np.linalg.norm(engine.bank.X, axis=1), 1.0, atol=1e-5)
    assert engine.history[-1]["model_version"] == "v001"   # incertezze e metriche ricalcolate

    out = cfg.models_dir / "v001"
    assert yaml.safe_load((out / "config.yaml").read_text())["query"]["batch_size"] == 10
    assert (out / "weights.json").exists()
    assert json.loads((out / "meta.json").read_text())["n_labels"] == 20
    np.testing.assert_allclose(load_embeddings(out / "embeddings").X, engine.bank.X, atol=1e-6)

    # soglia successiva raddoppiata: servono 40 etichette nuove dalle 20 del job precedente
    st = manager.status()
    assert st["n_retrains"] == 1 and st["next_threshold"] == 40 and st["new_labels"] == 5
    labeled = {i for i in engine.bank.ids if engine.bank.label_of(i)}
    assert not labeled & {b.image_id for b in engine.batch if not b.done}

    # al riavvio si riparte dall'ultima versione
    assert latest_embeddings(cfg) == ("v001", out / "embeddings")
    restarted = RetrainManager(engine)
    assert restarted.n_retrains == 1 and restarted.labels_at_last == 20


def test_stub_trainer_and_failure_do_not_break_the_hub(cfg, synth):
    engine, truth = _engine(cfg, synth)
    manager = RetrainManager.attach(engine)        # StubTrainer di default
    assert isinstance(manager.trainer, StubTrainer)
    old_X = engine.bank.X.copy()
    _label(engine, truth, 20)
    manager.join(10)
    assert engine.model_version == "v001"
    np.testing.assert_allclose(engine.bank.X, old_X, atol=1e-6)  # lo stub restituisce l'extractor congelato

    class Broken(FeatureExtractorTrainer):
        def retrain(self, labeled, unlabeled):
            raise RuntimeError("GPU esaurita")

    manager.trainer = Broken()
    manager.start("manuale")
    manager.join(10)
    assert manager.state == "failed" and "GPU esaurita" in manager.status()["reason"]
    assert engine.model_version == "v001"
    _label(engine, truth, 3)                       # l'hub continua a funzionare


def test_disabled_by_default(cfg, synth):
    X, y, ids = synth
    engine = ALEngine(cfg, Embeddings(ids, list(ids), X))
    assert RetrainManager.attach(engine) is None and engine.status()["retrain"] is None
