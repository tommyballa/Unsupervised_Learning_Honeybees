"""Re-train pesante dell'extractor: versioni dei modelli, policy di trigger, job in background."""
from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Union

import numpy as np
import yaml

from .config import ALConfig, RetrainConfig
from .embeddings import Embeddings, l2_normalize, save_embeddings

if TYPE_CHECKING:
    from .engine import ALEngine

BASE_VERSION = "v000"  # extractor congelato di partenza (paths.embeddings_dir)
_VERSION_RE = re.compile(r"^v(\d{3,})$")


def list_versions(models_dir: Union[str, Path]) -> list[str]:
    """Versioni ``vNNN`` complete (con embedding) presenti in ``models_dir``, in ordine."""
    d = Path(models_dir)
    if not d.is_dir():
        return []
    found = [p.name for p in d.iterdir() if _VERSION_RE.match(p.name) and (p / "embeddings" / "manifest.json").exists()]
    return sorted(found, key=lambda v: int(v[1:]))


def next_version(models_dir: Union[str, Path]) -> str:
    d = Path(models_dir)
    numbers = [int(m.group(1)) for p in (d.iterdir() if d.is_dir() else []) if (m := _VERSION_RE.match(p.name))]
    return f"v{max(numbers, default=0) + 1:03d}"


def latest_embeddings(cfg: ALConfig) -> tuple[str, Path]:
    """Versione corrente e cartella dei suoi embedding: l'ultimo re-train, o l'extractor congelato."""
    versions = list_versions(cfg.models_dir)
    if versions:
        return versions[-1], cfg.models_dir / versions[-1] / "embeddings"
    return BASE_VERSION, cfg.embeddings_dir


# --------------------------------------------------------------------------- #
# Interfacce
# --------------------------------------------------------------------------- #

@dataclass
class TrainingData:
    """Api passate al trainer. ``labels`` è vuoto per le non etichettate."""

    ids: list[str]
    paths: list[str]
    X: np.ndarray
    labels: list[str] = field(default_factory=list)


class Extractor(Protocol):
    """Extractor prodotto da un re-train."""

    def embed(self, paths: Sequence[str]) -> np.ndarray:
        """Embedding (n, dim) delle immagini, nell'ordine dato."""
        ...

    def save(self, out_dir: Path) -> None:
        """Salva i pesi in ``out_dir``."""
        ...


class FeatureExtractorTrainer(ABC):
    """Re-train pesante dell'extractor.

    L'implementazione vera farà fine-tuning contrastive (SupCon/SimCLR/DINO) più
    pseudo-label con soglia alta, ricalcolate da zero a ogni round, e consistency
    regularization. Qui c'è solo l'architettura attorno.
    """

    @abstractmethod
    def retrain(self, labeled: TrainingData, unlabeled: TrainingData) -> Extractor: ...


class FrozenExtractor:
    """Restituisce gli embedding già noti: è l'extractor congelato."""

    def __init__(self, paths: Sequence[str], X: np.ndarray) -> None:
        self._row = {p: i for i, p in enumerate(paths)}
        self._X = X

    def embed(self, paths: Sequence[str]) -> np.ndarray:
        return self._X[[self._row[p] for p in paths]]

    def save(self, out_dir: Path) -> None:
        with open(Path(out_dir) / "weights.json", "w", encoding="utf-8") as f:
            json.dump({"stub": True, "note": "extractor congelato, nessun peso nuovo"}, f, indent=2)


class StubTrainer(FeatureExtractorTrainer):
    """Implementazione finta: non addestra nulla e restituisce l'extractor congelato."""

    def retrain(self, labeled: TrainingData, unlabeled: TrainingData) -> Extractor:
        return FrozenExtractor(
            labeled.paths + unlabeled.paths,
            np.concatenate([labeled.X, unlabeled.X]) if len(labeled.ids) else unlabeled.X,
        )


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

class RetrainPolicy:
    """Decide quando lanciare un re-train."""

    def __init__(self, cfg: RetrainConfig) -> None:
        self.cfg = cfg

    def label_threshold(self, n_retrains: int) -> int:
        """Nuove etichette richieste per il prossimo job: 50, 100, 200, 400, ..."""
        return int(round(self.cfg.initial_labels * self.cfg.growth_factor ** n_retrains))

    def plateau(self, metrics: Sequence[Optional[float]]) -> bool:
        """Vero se la metrica di validazione non migliora da ``plateau_rounds`` round."""
        values = [m for m in metrics if m is not None]
        k = self.cfg.plateau_rounds
        if len(values) <= k:
            return False
        return max(values[-k:]) <= max(values[:-k]) + self.cfg.min_delta

    def decide(self, new_labels: int, n_retrains: int, metrics: Sequence[Optional[float]]) -> Optional[str]:
        """Motivo del re-train, o ``None``. ``metrics``: un valore per round dall'ultimo re-train."""
        by_count = self.cfg.policy in ("label_count", "either")
        by_plateau = self.cfg.policy in ("plateau", "either")
        if by_count and new_labels >= self.label_threshold(n_retrains):
            return f"{new_labels} nuove etichette (soglia {self.label_threshold(n_retrains)})"
        if by_plateau and new_labels > 0 and self.plateau(metrics):
            return f"{self.cfg.metric} fermo da {self.cfg.plateau_rounds} round"
        return None


# --------------------------------------------------------------------------- #
# Job in background
# --------------------------------------------------------------------------- #

class RetrainManager:
    """Lancia i re-train in un thread, senza bloccare l'hub, e versiona il risultato.

    A fine job: ricalcola tutti gli embedding, li salva in ``models/vNNN/`` con config
    e pesi, poi fa ricostruire all'engine bank, grafo e incertezze.
    """

    def __init__(self, engine: "ALEngine", trainer: Optional[FeatureExtractorTrainer] = None) -> None:
        self.engine = engine
        self.trainer = trainer or StubTrainer()
        self.policy = RetrainPolicy(engine.cfg.retrain)
        self.state = "idle"  # idle | running | failed
        self.reason = ""
        self.error = ""
        self._thread: Optional[threading.Thread] = None
        # Lo stato dei job precedenti si ricostruisce dalle versioni su disco.
        versions = list_versions(engine.cfg.models_dir)
        self.n_retrains = len(versions)
        self.labels_at_last = 0
        self.round_at_last = 0
        if versions:
            with open(engine.cfg.models_dir / versions[-1] / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.labels_at_last = meta["n_labels"]
            self.round_at_last = meta["round"]

    @classmethod
    def attach(cls, engine: "ALEngine", trainer: Optional[FeatureExtractorTrainer] = None) -> Optional["RetrainManager"]:
        """Collega il manager all'engine se ``retrain.enabled`` è attivo."""
        if not engine.cfg.retrain.enabled:
            return None
        engine.retrain = cls(engine, trainer)
        return engine.retrain

    def _new_labels(self) -> int:
        return max(0, self.engine.bank.n_labeled - self.labels_at_last)

    def maybe_start(self) -> Optional[str]:
        """Chiamato dall'engine a ogni refresh: avvia un job se la policy lo chiede."""
        if self.state == "running":
            return None
        metric = self.engine.cfg.retrain.metric
        metrics = [h.get(metric) for h in self.engine.history if h["round"] > self.round_at_last]
        reason = self.policy.decide(self._new_labels(), self.n_retrains, metrics)
        if reason:
            self.start(reason)
        return reason

    def start(self, reason: str = "manuale") -> None:
        """Avvia il job in un thread. I dati sono una fotografia presa adesso."""
        engine = self.engine
        with engine.lock:
            if self.state == "running":
                return
            bank = engine.bank
            lab = bank.labeled_indices()
            unl = np.setdiff1d(np.arange(len(bank)), lab)
            labeled = TrainingData([bank.ids[i] for i in lab], [bank.paths[i] for i in lab], bank.X[lab], bank.labels_at(lab))
            unlabeled = TrainingData([bank.ids[i] for i in unl], [bank.paths[i] for i in unl], bank.X[unl])
            snapshot = {
                "ids": list(bank.ids), "paths": list(bank.paths), "reason": reason,
                "n_labels": bank.n_labeled, "round": engine.round, "parent_version": engine.model_version,
            }
            self.state, self.reason, self.error = "running", reason, ""
        self._thread = threading.Thread(target=self._run, args=(labeled, unlabeled, snapshot), daemon=True, name="bee-al-retrain")
        self._thread.start()

    def _run(self, labeled: TrainingData, unlabeled: TrainingData, snapshot: dict[str, Any]) -> None:
        cfg = self.engine.cfg
        try:
            extractor = self.trainer.retrain(labeled, unlabeled)
            X = l2_normalize(extractor.embed(snapshot["paths"]))  # tutti gli embedding, da zero

            version = next_version(cfg.models_dir)
            out = cfg.models_dir / version
            out.mkdir(parents=True)
            with open(out / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
            extractor.save(out)
            meta = {k: snapshot[k] for k in ("reason", "n_labels", "round", "parent_version")}
            meta.update({"version": version, "trainer": type(self.trainer).__name__, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
            with open(out / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            # Gli embedding per ultimi: una versione è "completa" solo quando ci sono loro.
            save_embeddings(out / "embeddings", snapshot["ids"], snapshot["paths"], X, meta={"model": version})

            self.n_retrains += 1
            self.labels_at_last = snapshot["n_labels"]
            self.round_at_last = snapshot["round"]
            self.engine.set_embeddings(Embeddings(snapshot["ids"], snapshot["paths"], X, {"model": version}), version)
            self.state = "idle"
        except Exception as e:  # il job non deve mai far cadere l'hub
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.error or self.reason,
            "n_retrains": self.n_retrains,
            "new_labels": self._new_labels(),
            "next_threshold": self.policy.label_threshold(self.n_retrains),
            "policy": self.engine.cfg.retrain.policy,
        }
