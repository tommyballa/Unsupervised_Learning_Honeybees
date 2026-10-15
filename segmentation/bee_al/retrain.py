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
from . import UNKNOWN_LABEL
from .embeddings import Embeddings, l2_normalize, save_embeddings
from .propagation import build_graph_cache_in_subprocess

if TYPE_CHECKING:
    from .engine import ALEngine

BASE_VERSION = "v000"  # extractor congelato di partenza (paths.embeddings_dir)
_VERSION_RE = re.compile(r"^v(\d{3,})$")


def read_meta(version_dir: Union[str, Path]) -> dict[str, Any]:
    try:
        with open(Path(version_dir) / "meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def list_versions(models_dir: Union[str, Path], include_rejected: bool = False) -> list[str]:
    """Versioni ``vNNN`` complete (con embedding) presenti in ``models_dir``, in ordine.

    Quelle scartate perché peggioravano la validazione restano su disco ma non vengono usate.
    """
    d = Path(models_dir)
    if not d.is_dir():
        return []
    found = [
        p.name for p in d.iterdir()
        if _VERSION_RE.match(p.name) and (p / "embeddings" / "manifest.json").exists()
        and (include_rejected or read_meta(p).get("accepted", True))
    ]
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
    """Api passate al trainer.

    Per le etichettate ``labels`` ha la classe di ognuna. Per le non etichettate è vuoto e
    ``pseudo_labels`` ha la classe predetta dove la confidenza supera la soglia, altrimenti ``None``.
    """

    ids: list[str]
    paths: list[str]
    X: np.ndarray
    labels: list[str] = field(default_factory=list)
    pseudo_labels: list[Optional[str]] = field(default_factory=list)


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

def make_trainer(cfg: ALConfig) -> FeatureExtractorTrainer:
    """Trainer scelto in config: fine-tuning contrastive oppure lo stub."""
    if cfg.retrain.trainer == "stub":
        return StubTrainer()
    from .contrastive import ContrastiveTrainer

    return ContrastiveTrainer(cfg)


class RetrainManager:
    """Lancia i re-train in un thread, senza bloccare l'hub, e versiona il risultato.

    Un job parte dal pulsante dell'hub (``start``) oppure da solo, se ``retrain.enabled``
    è attivo e la policy lo chiede. A fine job: ricalcola tutti gli embedding, confronta
    la metrica di validazione prima e dopo, salva tutto in ``models/vNNN/`` e, se la
    versione non è peggiore, fa ricostruire all'engine bank, grafo e incertezze.
    """

    def __init__(self, engine: "ALEngine", trainer: Optional[FeatureExtractorTrainer] = None) -> None:
        self.engine = engine
        self.trainer = trainer or make_trainer(engine.cfg)
        self.policy = RetrainPolicy(engine.cfg.retrain)
        self.state = "idle"  # idle | running | failed
        self.reason = ""
        self.error = ""
        self._thread: Optional[threading.Thread] = None
        # Lo stato dei job precedenti si ricostruisce dalle versioni su disco, scartate comprese.
        versions = list_versions(engine.cfg.models_dir, include_rejected=True)
        self.n_retrains = len(versions)
        self.last_result: dict[str, Any] = read_meta(engine.cfg.models_dir / versions[-1]) if versions else {}
        self.labels_at_last = self.last_result.get("n_labels", 0)
        self.round_at_last = self.last_result.get("round", 0)

    @classmethod
    def attach(cls, engine: "ALEngine", trainer: Optional[FeatureExtractorTrainer] = None) -> "RetrainManager":
        """Collega il manager all'engine. I job automatici partono solo con ``retrain.enabled``."""
        engine.retrain = cls(engine, trainer)
        return engine.retrain

    def _new_labels(self) -> int:
        return max(0, self.engine.bank.n_labeled - self.labels_at_last)

    def blocked_by(self) -> Optional[str]:
        """Perché un job non può partire adesso, o ``None`` se può."""
        cfg = self.engine.cfg.retrain
        if self.state == "running":
            return "a re-train is already running"
        if self.engine.bank.n_labeled < cfg.min_labels:
            return f"at least {cfg.min_labels} labels are needed ({self.engine.bank.n_labeled} so far)"
        if len(self.engine.bank.classes) < 2:
            return "at least 2 classes are needed"
        return None

    def maybe_start(self) -> Optional[str]:
        """Chiamato dall'engine a ogni refresh: avvia un job se la policy lo chiede."""
        if not self.engine.cfg.retrain.enabled or self.blocked_by():
            return None
        metric = self.engine.cfg.retrain.metric
        metrics = [h.get(metric) for h in self.engine.history if h["round"] > self.round_at_last]
        reason = self.policy.decide(self._new_labels(), self.n_retrains, metrics)
        if reason:
            self.start(reason)
        return reason

    def start(self, reason: str = "manual") -> None:
        """Avvia il job in un thread. I dati sono una fotografia presa adesso.

        Solleva ``ValueError`` se non può partire (job già in corso, poche etichette).
        """
        engine = self.engine
        with engine.lock:
            blocked = self.blocked_by()
            if blocked:
                raise ValueError(blocked)
            bank = engine.bank
            lab = bank.labeled_indices()
            unl = np.setdiff1d(np.arange(len(bank)), lab)
            labeled = TrainingData([bank.ids[i] for i in lab], [bank.paths[i] for i in lab], bank.X[lab], bank.labels_at(lab))

            # Pseudo-label: predizioni correnti molto sicure, ricalcolate a ogni job.
            # Mai per i "Non so" né per le api lontane da ogni esempio etichettato.
            threshold = engine.cfg.retrain.pseudo_label_threshold
            predicted, confidence = engine.final_pred.labels, engine.final_pred.confidence
            skip = set(bank.unknown_indices().tolist()) | set(np.flatnonzero(engine.unassigned).tolist())
            pseudo = [predicted[i] if confidence[i] >= threshold and i not in skip else None for i in unl]
            unlabeled = TrainingData([bank.ids[i] for i in unl], [bank.paths[i] for i in unl], bank.X[unl], pseudo_labels=pseudo)

            snapshot = {
                "ids": list(bank.ids), "paths": list(bank.paths), "X": bank.X, "reason": reason,
                "n_labels": bank.n_labeled, "round": engine.round, "parent_version": engine.model_version,
                "labels": {bank.ids[i]: lbl for i, lbl in zip(lab, labeled.labels)},
                "val_labels": dict(engine.val_labels),
                "n_pseudo": sum(p is not None for p in pseudo),
            }
            self.state, self.reason, self.error = "running", reason, ""
        self._thread = threading.Thread(target=self._run, args=(labeled, unlabeled, snapshot), daemon=True, name="bee-al-retrain")
        self._thread.start()

    @property
    def cache_dir(self) -> Path:
        """Stessa cache dei grafi usata dal motore."""
        return self.engine.cfg.results_dir / "cache"

    def _validation_score(self, snapshot: dict[str, Any], X: np.ndarray) -> Optional[dict[str, Optional[float]]]:
        """Metriche di validazione che si otterrebbero con gli embedding ``X`` e le etichette della fotografia."""
        from .engine import predict_all
        from .memory_bank import MemoryBank
        from .propagation import LabelSpreader
        from .validation import classification_metrics

        cfg = self.engine.cfg
        val = [(i, lbl) for i, lbl in snapshot["val_labels"].items() if lbl != UNKNOWN_LABEL]
        if not val:
            return None
        bank = MemoryBank(snapshot["ids"], X, backend=cfg.knn.backend)
        for image_id, lbl in snapshot["labels"].items():
            bank.add_label(image_id, lbl)
        spreader = None
        if cfg.propagation.enabled:
            p = cfg.propagation
            # Grafo costruito fuori dal processo dell'hub (vedi build_graph_cache_in_subprocess)
            build_graph_cache_in_subprocess(X, p.n_neighbors, self.cache_dir, cfg.knn.backend)
            spreader = LabelSpreader.from_bank(bank, p.n_neighbors, p.alpha, p.max_iter, p.tol, cache_dir=self.cache_dir)
        predicted = predict_all(bank, spreader, cfg)[1].labels
        return classification_metrics([lbl for _, lbl in val], [predicted[bank.index_of(i)] for i, _ in val])

    def _run(self, labeled: TrainingData, unlabeled: TrainingData, snapshot: dict[str, Any]) -> None:
        cfg = self.engine.cfg
        try:
            t0 = time.time()
            extractor = self.trainer.retrain(labeled, unlabeled)
            X = l2_normalize(extractor.embed(snapshot["paths"]))  # tutti gli embedding, da zero

            metric = cfg.retrain.metric
            before = self._validation_score(snapshot, snapshot["X"])
            after = self._validation_score(snapshot, X)
            worse = before is not None and after is not None and after[metric] < before[metric]
            accepted = not (cfg.retrain.require_improvement and worse)

            version = next_version(cfg.models_dir)
            out = cfg.models_dir / version
            out.mkdir(parents=True)
            with open(out / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
            extractor.save(out)
            meta = {k: snapshot[k] for k in ("reason", "n_labels", "n_pseudo", "round", "parent_version")}
            meta.update({
                "version": version, "trainer": type(self.trainer).__name__,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": round(time.time() - t0, 1),
                "metric": metric, "validation_before": before, "validation_after": after, "accepted": accepted,
            })
            with open(out / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            # Gli embedding per ultimi: una versione è "completa" solo quando ci sono loro.
            # Già normalizzati: salvati identici a quelli in memoria, così al riavvio l'hub ritrova il grafo in cache
            save_embeddings(out / "embeddings", snapshot["ids"], snapshot["paths"], X, meta={"model": version}, normalize=False)

            self.n_retrains += 1
            self.labels_at_last = snapshot["n_labels"]
            self.round_at_last = snapshot["round"]
            self.last_result = meta
            if accepted:
                if cfg.propagation.enabled:  # così set_embeddings trova il grafo in cache e non blocca l'hub
                    build_graph_cache_in_subprocess(X, cfg.propagation.n_neighbors, self.cache_dir, cfg.knn.backend)
                self.engine.set_embeddings(Embeddings(snapshot["ids"], snapshot["paths"], X, {"model": version}), version)
            self.state = "idle"
        except Exception as e:  # il job non deve mai far cadere l'hub
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        progress = getattr(self.trainer, "progress", lambda: None)() if self.state == "running" else None
        return {
            "state": self.state,
            "reason": self.reason,
            "error": self.error,
            "progress": progress,
            "blocked_by": self.blocked_by(),
            "auto": self.engine.cfg.retrain.enabled,
            "trainer": type(self.trainer).__name__,
            "n_retrains": self.n_retrains,
            "new_labels": self._new_labels(),
            "next_threshold": self.policy.label_threshold(self.n_retrains),
            "policy": self.engine.cfg.retrain.policy,
            "last": self.last_result or None,
        }
