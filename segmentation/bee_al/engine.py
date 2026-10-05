"""Motore di active learning usato dall'hub: bank, store, batch condiviso e metriche."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from . import REMOVED_LABEL, UNKNOWN_LABEL, audit, snapshot
from .config import DEFAULT_CONFIG_PATH, ALConfig, load_config
from .embeddings import Embeddings, load_embeddings
from .knn import Prediction, knn_predict, nearest_labeled_similarity
from .label_store import LabelStore, is_training_event
from .memory_bank import MemoryBank
from .propagation import LabelSpreader
from .query import is_cold_start, select_batch
from .retrain import BASE_VERSION, latest_embeddings
from .validation import classification_metrics, load_or_create_validation_set

CONFIDENCES = ("High", "Medium", "Low")


class ALError(ValueError):
    """Richiesta non valida (immagine sconosciuta, etichetta vuota, ...)."""


@dataclass
class BatchItem:
    image_id: str
    index: int
    query_type: str
    batch_id: str
    assigned_to: Optional[str] = None
    assigned_at: float = 0.0
    done: bool = False


def predict_all(bank: MemoryBank, spreader: Optional[LabelSpreader], cfg: ALConfig) -> tuple[Prediction, Prediction]:
    """Predizioni su tutte le api: (kNN, finale).

    La finale è la propagazione se abilitata, con il kNN come ripiego per le api
    che nessuna etichetta raggiunge lungo il grafo.
    """
    knn = knn_predict(bank, None, k=cfg.knn.k, temperature=cfg.knn.temperature)
    if spreader is None or not cfg.propagation.enabled or not knn.classes:
        return knn, knn
    prop = spreader.propagate_bank(bank)
    reached = prop.probs.sum(axis=1, keepdims=True) > 0
    return knn, Prediction(knn.classes, np.where(reached, prop.probs, knn.probs))


class ALEngine:
    """Stato in-process dell'active learning, protetto da un lock.

    Tutti gli annotatori condividono un solo batch globale; ogni immagine è
    assegnata a un annotatore alla volta. Lo store su disco è la fonte di verità:
    al riavvio lo stato si ricostruisce da lì.
    """

    def __init__(
        self,
        cfg: ALConfig,
        embeddings: Embeddings,
        store: Optional[LabelStore] = None,
        model_version: str = BASE_VERSION,
        config_path: Union[str, Path, None] = None,
        clock: Callable[[], float] = time.time,
        persist_rounds: bool = True,
    ) -> None:
        self.cfg = cfg
        self.persist_rounds = persist_rounds  # False = non scrive al_rounds.jsonl (script che non sono l'hub)
        self.config_path = Path(config_path) if config_path else None
        self.clock = clock
        self.lock = threading.RLock()
        self.store = store or LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
        self.rounds_path = cfg.results_dir / "al_rounds.jsonl"
        self.history: list[dict[str, Any]] = self._load_history()
        self.round = len(self.history)
        self.batch: list[BatchItem] = []
        self.batch_id = ""
        self.retrain: Any = None  # RetrainManager opzionale, vedi bee_al/retrain.py
        self._val_assigned: dict[str, tuple[str, float]] = {}
        self.val_labels: dict[str, str] = {}
        self.validation_ids = load_or_create_validation_set(
            cfg.validation_path, embeddings.ids, cfg.validation.size, cfg.validation.seed
        )
        self._val_set = set(self.validation_ids)
        self._round_final: Optional[np.ndarray] = None  # etichette finali al round precedente (criteri di stop)
        # Audit attivo: campione di controllo di uno snapshot, mai usato per addestrare.
        self.audit: Optional[dict[str, Any]] = None
        self.audit_labels: dict[str, dict[str, str]] = {}   # image_id -> {annotatore: etichetta}
        self._audit_ids: set[str] = set()
        self._audit_assigned: dict[str, dict[str, float]] = {}
        self._restore_audit(set(embeddings.ids))
        self._install_embeddings(embeddings, model_version)
        self.refresh(reload_config=False)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None, persist_rounds: bool = True) -> "ALEngine":
        """Carica config, embedding dell'ultima versione, store e importa le etichette legacy."""
        cfg = load_config(config_path)
        version, emb_dir = latest_embeddings(cfg)
        store = LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
        store.import_legacy(cfg.results_dir, config_id=cfg.config_id)
        return cls(cfg, load_embeddings(emb_dir), store, version, config_path or DEFAULT_CONFIG_PATH,
                   persist_rounds=persist_rounds)

    # ------------------------------ stato base ------------------------------ #

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.rounds_path.exists():
            return []
        with open(self.rounds_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _install_embeddings(self, embeddings: Embeddings, model_version: str) -> None:
        self.bank = MemoryBank(embeddings.ids, embeddings.X, embeddings.paths, backend=self.cfg.knn.backend)
        self.model_version = model_version
        self.spreader: Optional[LabelSpreader] = None
        if self.cfg.propagation.enabled:
            p = self.cfg.propagation
            self.spreader = LabelSpreader.from_bank(
                self.bank, p.n_neighbors, p.alpha, p.max_iter, p.tol, cache_dir=self.cfg.results_dir / "cache"
            )

    def set_embeddings(self, embeddings: Embeddings, model_version: str) -> None:
        """Sostituisce l'extractor (fine re-train): nuova bank, nuovo grafo, incertezze ricalcolate."""
        with self.lock:
            self._install_embeddings(embeddings, model_version)
            self.refresh(reload_config=False)

    def _reload_labels(self) -> None:
        """Ricostruisce le etichette della bank e del validation set dallo store, con gli alias attuali."""
        self.store.set_aliases(self.cfg.labels.label_aliases)
        self.bank.clear_labels()
        # Le api di validazione non entrano mai nella bank, nemmeno se hanno un'etichetta legacy.
        # Gli eventi di validazione e di audit sono tenuti da parte: non addestrano nulla.
        for image_id, ev in self.store.current(is_training_event).items():
            if image_id in self.bank and image_id not in self._val_set:
                self.bank.add_label(image_id, ev.label)
        self._reload_audit_labels()
        self.val_labels = {
            image_id: ev.label
            for image_id, ev in self.store.current(lambda e: e.query_type == "validation").items()
            if image_id in self._val_set and image_id in self.bank
        }

    # ------------------------------- refresh -------------------------------- #

    def refresh(self, reload_config: bool = True) -> dict[str, Any]:
        """Ricalcola propagazione, incertezze e metriche e genera il batch successivo.

        Rilegge anche la sezione ``labels`` della config: alias, classi suggerite e
        classi di scarto cambiano senza riavviare l'hub e senza toccare i CSV.
        """
        with self.lock:
            if reload_config and self.config_path and self.config_path.exists():
                self.cfg.labels = load_config(self.config_path).labels
            self._reload_labels()
            self.recompute()
            entry = self._evaluate()
            entry.update(self._round_dynamics())
            self.round += 1
            entry["round"] = self.round
            self.history.append(entry)
            if self.persist_rounds:
                self.rounds_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.rounds_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            self._new_batch()
            if self.retrain is not None:
                self.retrain.maybe_start()
            return entry

    def recompute(self) -> None:
        """Ricalcola predizioni e "non assegnate" con le etichette attuali, senza cambiare round né batch."""
        with self.lock:
            self.knn_pred, self.final_pred = predict_all(self.bank, self.spreader, self.cfg)
            self._update_novelty()

    def _update_novelty(self) -> None:
        """Distanza di ogni ape dalle etichettate e gruppo delle "non assegnate".

        Le non assegnate sono la frazione ``knn.unassigned_fraction`` di api non
        etichettate (validation set escluso) più lontane da ogni esempio: restano fuori dai cluster.
        """
        n = len(self.bank)
        self.unassigned = np.zeros(n, dtype=bool)
        self.nearest_sim: Optional[np.ndarray] = None
        if not self.bank.n_labeled:
            return
        self.nearest_sim = nearest_labeled_similarity(self.bank)
        free = self._not_validation()  # le api di validazione restano fuori da questa pagina
        free[self.bank.labeled_indices()] = False
        free[self.bank.unknown_indices()] = False
        free_idx = np.flatnonzero(free)
        k = int(round(self.cfg.knn.unassigned_fraction * len(free_idx)))
        if k:
            self.unassigned[free_idx[np.argsort(self.nearest_sim[free_idx], kind="stable")[:k]]] = True

    def _evaluate(self) -> dict[str, Any]:
        """Metriche sul validation set (solo etichette date in modalità validazione)."""
        val = [(i, lbl) for i, lbl in self.val_labels.items() if lbl != UNKNOWN_LABEL]
        idx = [self.bank.index_of(i) for i, _ in val]
        y_true = [lbl for _, lbl in val]
        entry: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_version": self.model_version,
            "config_id": self.cfg.config_id,
            "n_labels": self.bank.n_labeled,
            "n_classes": len(self.bank.classes),
            "n_validation": len(val),
            "accuracy": None, "macro_f1": None, "knn_accuracy": None, "knn_macro_f1": None,
        }
        if val and self.bank.n_labeled:
            final = classification_metrics(y_true, [self.final_pred.labels[i] for i in idx])
            knn = classification_metrics(y_true, [self.knn_pred.labels[i] for i in idx])
            entry.update(final)
            entry.update({"knn_accuracy": knn["accuracy"], "knn_macro_f1": knn["macro_f1"]})
        return entry

    def _final_labels(self) -> np.ndarray:
        """Etichetta finale di ogni ape: umana se c'è, altrimenti predetta.

        ``None`` per le api di validazione, i "Non so", le non assegnate e quelle senza predizione.
        """
        out = np.empty(len(self.bank), dtype=object)
        pred = self.final_pred
        if pred.classes:
            reached = pred.probs.sum(axis=1) > 0
            out[reached] = np.asarray(pred.classes, dtype=object)[pred.probs.argmax(axis=1)[reached]]
        out[self.unassigned] = None
        labeled = self.bank.labeled_indices()
        out[labeled] = np.asarray(self.bank.labels_at(labeled), dtype=object)
        out[self.bank.unknown_indices()] = None
        out[~self._not_validation()] = None
        return out

    def _round_dynamics(self) -> dict[str, Any]:
        """Campi del round usati dai criteri di stop: stabilità delle predizioni e scoperta di classi.

        - ``pred_change``: frazione di api la cui etichetta finale è cambiata rispetto al round
          precedente (solo api che hanno un'etichetta finale in entrambi i round).
        - ``model_changed``: l'extractor è cambiato, quindi ``pred_change`` non è confrontabile.
        - ``effective``: dal round precedente sono arrivati eventi di etichettatura. Un riavvio
          dell'hub o un ricalcolo a vuoto creano un round ma non dicono nulla sulla convergenza,
          quindi i criteri di stop li ignorano.
        - ``novelty_queries`` / ``novelty_new_class``: query ``novelty`` del batch appena chiuso
          che sono state etichettate, e quante hanno portato una classe che prima non c'era.
        """
        prev = self.history[-1] if self.history else None
        now = self._final_labels()
        pred_change: Optional[float] = None
        if self._round_final is not None and len(self._round_final) == len(now):
            both = np.not_equal(self._round_final, None) & np.not_equal(now, None)
            if both.any():
                pred_change = float(np.mean(self._round_final[both] != now[both]))
        self._round_final = now

        events = [e for e in self.store.events() if is_training_event(e)]
        classes = self.bank.classes
        prev_classes = prev.get("classes") if prev else None
        if prev is None:
            effective, new_classes, since = bool(events), list(classes), 0
        else:
            if prev.get("n_events") is not None:
                effective = len(events) != prev["n_events"]
            else:  # riga scritta da una versione precedente: si confronta il numero di etichette
                effective = self.bank.n_labeled != prev.get("n_labels")
            if prev_classes is None:
                new_classes, since = [], 0
            else:
                new_classes = sorted(set(classes) - set(prev_classes))
                since = prev.get("rounds_since_new_class") or 0
                if new_classes:
                    since = 0
                elif effective:
                    since += 1

        # Ultimo evento per immagine tra le query novelty del batch che si sta chiudendo.
        novelty: dict[str, str] = {}
        if self.batch_id:
            for e in events:
                if e.batch_id == self.batch_id and e.query_type == "novelty":
                    novelty[e.image_id] = self.store.resolve(e.label)
        answered = [lbl for lbl in novelty.values() if lbl not in (REMOVED_LABEL, UNKNOWN_LABEL)]
        known = set(prev_classes or [])
        return {
            "pred_change": pred_change,
            "model_changed": bool(prev is not None and prev.get("model_version") != self.model_version),
            "effective": effective,
            "n_events": len(events),
            "classes": list(classes),
            "new_classes": new_classes,
            "rounds_since_new_class": since,
            "novelty_queries": len(answered),
            "novelty_new_class": sum(lbl not in known for lbl in answered) if prev_classes is not None else 0,
        }

    def stopping_status(self) -> dict[str, Any]:
        """Criteri di stop: per ciascuno ``{"ok": True | False | None, "detail": testo}``.

        ``None`` = non ci sono ancora abbastanza round per dirlo. Si guardano solo i round
        ``effective`` (con etichette nuove). I criteri sono un'indicazione, non fermano nulla.

        - ``f1_plateau``: la macro-F1 di validazione degli ultimi ``window`` round non supera
          di più di ``f1_min_delta`` il massimo dei round precedenti.
        - ``predictions_stable``: ``pred_change < max_pred_change`` negli ultimi ``window`` round
          con lo stesso extractor del round prima.
        - ``no_new_classes``: nessuna classe nuova da almeno ``window`` round.
        """
        with self.lock:
            cfg = self.cfg.stopping
            w = cfg.window
            rounds = [h for h in self.history if h.get("effective")]

            f1 = [h["macro_f1"] for h in rounds if h.get("macro_f1") is not None]
            if len(f1) <= w:
                plateau = {"ok": None, "detail": f"needs {w + 1} rounds with validation metrics, {len(f1)} so far"}
            else:
                best_before, best_recent = max(f1[:-w]), max(f1[-w:])
                plateau = {
                    "ok": bool(best_recent <= best_before + cfg.f1_min_delta),
                    "detail": f"best macro-F1 in the last {w} rounds {best_recent:.3f} vs {best_before:.3f} before "
                              f"(gain {best_recent - best_before:+.3f}, threshold {cfg.f1_min_delta:.3f})",
                }

            changes = [h["pred_change"] for h in rounds
                       if h.get("pred_change") is not None and not h.get("model_changed")]
            if len(changes) < w:
                stable = {"ok": None, "detail": f"needs {w} comparable rounds, {len(changes)} so far"}
            else:
                recent = changes[-w:]
                stable = {
                    "ok": bool(max(recent) < cfg.max_pred_change),
                    "detail": f"labels changed for {', '.join(f'{100 * c:.2f}%' for c in recent)} of the bees "
                              f"in the last {w} rounds (threshold {100 * cfg.max_pred_change:.2f}%)",
                }

            last = self.history[-1] if self.history else {}
            since = last.get("rounds_since_new_class")
            if since is None or len(rounds) < w:
                discovery = {"ok": None, "detail": f"needs {w} rounds with new labels, {len(rounds)} so far"}
            else:
                discovery = {"ok": bool(since >= w),
                             "detail": f"{since} round(s) since the last new class (threshold {w})"}

            return {
                "window": w,
                "f1_plateau": plateau,
                "predictions_stable": stable,
                "no_new_classes": discovery,
                "all_ok": all(c["ok"] is True for c in (plateau, stable, discovery)),
            }

    def _new_batch(self) -> None:
        now = self.clock()
        # Le immagini che qualcuno sta guardando in questo momento restano sue.
        carried = [b for b in self.batch if not b.done and self._is_assigned(b, now)]
        self.batch_id = f"r{self.round:04d}"
        excluded = set(self.bank.labeled_indices().tolist())
        excluded.update(self.bank.unknown_indices().tolist())
        excluded.update(self.bank.index_of(i) for i in self._val_set if i in self.bank)
        excluded.update(self.bank.index_of(i) for i in self._audit_ids if i in self.bank)
        excluded.update(b.index for b in carried)
        source = self.knn_pred if self.cfg.query.uncertainty_source == "knn" else self.final_pred
        rng = np.random.default_rng([self.cfg.seed, self.round])
        queries = select_batch(
            self.bank.X, excluded, source.probs if source.classes else None,
            self.bank.n_labeled, len(self.bank.classes), self.cfg.query, rng,
            novelty=None if self.nearest_sim is None else 1.0 - self.nearest_sim,
        )
        self.batch = carried + [
            BatchItem(self.bank.ids[q.index], q.index, q.query_type, self.batch_id) for q in queries
        ]

    def _is_assigned(self, item: BatchItem, now: float) -> bool:
        return item.assigned_to is not None and now - item.assigned_at < self.cfg.hub.assignment_timeout_s

    # ----------------------------- etichettatura ---------------------------- #

    def next_image(self, annotator: str, mode: str = "train") -> Optional[BatchItem]:
        """Prossima immagine per l'annotatore, o ``None`` se non c'è nulla da proporre.

        Chi ha già un'immagine assegnata riceve di nuovo quella (es. ricaricando la pagina).
        """
        with self.lock:
            now = self.clock()
            if mode == "validation":
                return self._next_validation(annotator, now)
            if mode == "audit":
                return self._next_audit(annotator, now)
            pending = [b for b in self.batch if not b.done]
            for item in pending:
                if item.assigned_to == annotator:
                    item.assigned_at = now
                    return item
            for item in pending:
                if not self._is_assigned(item, now):
                    item.assigned_to, item.assigned_at = annotator, now
                    return item
            return None  # il resto del batch è in mano ad altri annotatori, o non resta nulla

    def _next_validation(self, annotator: str, now: float) -> Optional[BatchItem]:
        timeout = self.cfg.hub.assignment_timeout_s
        free: Optional[str] = None
        for image_id in self.validation_ids:
            if image_id in self.val_labels or image_id not in self.bank:
                continue
            owner = self._val_assigned.get(image_id)
            if owner and owner[0] == annotator:
                free = image_id
                break
            if free is None and (owner is None or now - owner[1] >= timeout):
                free = image_id
        if free is None:
            return None
        self._val_assigned[free] = (annotator, now)
        return BatchItem(free, self.bank.index_of(free), "validation", "validation", annotator, now)

    def submit(
        self, annotator: str, image_id: str, label: str, confidence: str = "", mode: str = "train", manual: bool = False
    ) -> str:
        """Salva un'etichetta e aggiorna subito la bank. Restituisce l'etichetta normalizzata.

        Di norma si etichettano solo api proposte dal batch (o si corregge una propria
        etichetta). Con ``manual=True`` si può etichettare qualsiasi ape, ad esempio dalla
        pagina dei cluster: l'evento ha ``query_type = manual``.

        ``UNKNOWN_LABEL`` ("Non so") mette l'ape da parte; ``REMOVED_LABEL`` annulla l'etichetta.
        A batch finito ricalcola tutto e prepara il batch successivo.
        """
        with self.lock:
            if image_id not in self.bank:
                raise ALError(f"immagine sconosciuta: {image_id}")
            label = self.store.normalize_label(label)
            if not label:
                raise ALError("etichetta vuota")
            if confidence not in CONFIDENCES:
                confidence = ""

            if mode == "validation":
                if image_id not in self._val_set:
                    raise ALError(f"{image_id} non fa parte del validation set")
                self.store.append(image_id, label, annotator, confidence, "validation",
                                  self.cfg.config_id, self.model_version, "validation")
                self._val_assigned.pop(image_id, None)
                resolved = self.store.resolve(label)
                if resolved == REMOVED_LABEL:
                    self.val_labels.pop(image_id, None)
                else:
                    self.val_labels[image_id] = resolved
                return label

            if mode == "audit":
                if self.audit is None or image_id not in self._audit_ids:
                    raise ALError(f"{image_id} non fa parte dell'audit attivo")
                self.store.append(image_id, label, annotator, confidence, "audit", self.cfg.config_id,
                                  self.model_version, audit.batch_id_of(self.audit["snapshot"]))
                self._audit_assigned.get(image_id, {}).pop(annotator, None)
                resolved = self.store.resolve(label)
                answers = self.audit_labels.setdefault(image_id, {})
                if resolved == REMOVED_LABEL:
                    answers.pop(annotator, None)
                else:
                    answers[annotator] = resolved
                return label

            if image_id in self._val_set:
                raise ALError(f"{image_id} è nel validation set: va etichettata in modalità validazione")
            if image_id in self._audit_ids:
                raise ALError(f"{image_id} è nel campione di audit: va etichettata in modalità audit")
            item = next((b for b in self.batch if b.image_id == image_id), None)
            if item is not None:
                query_type, batch_id = item.query_type, item.batch_id
            else:
                # Correzione di un'etichetta data in un batch precedente: stesso tipo di query.
                prev = next(
                    (e for e in reversed(self.store.events()) if e.image_id == image_id and is_training_event(e)),
                    None,
                )
                if prev is not None and prev.query_type != "legacy":
                    query_type, batch_id = prev.query_type, prev.batch_id
                elif manual:
                    query_type, batch_id = "manual", "manual"
                else:
                    raise ALError(f"{image_id} non è stata proposta dall'active learning")
            self.store.append(image_id, label, annotator, confidence, query_type,
                              self.cfg.config_id, self.model_version, batch_id)
            self.bank.add_label(image_id, self.store.resolve(label))
            if item is not None:
                item.done = label != REMOVED_LABEL
                if item.done and all(b.done for b in self.batch):
                    self.refresh()
            return label

    # ------------------------------- lettura -------------------------------- #

    def quick_labels(self) -> list[str]:
        """Classi già viste più quelle suggerite in config."""
        labels = set(self.bank.classes)
        labels.update(lbl for lbl in self.val_labels.values() if lbl != UNKNOWN_LABEL)
        labels.update(self.store.resolve(self.store.normalize_label(c)) for c in self.cfg.labels.classes)
        labels.difference_update({UNKNOWN_LABEL, REMOVED_LABEL, ""})
        return sorted(labels)

    def suggestion(self, image_id: str) -> Optional[tuple[str, float]]:
        """Classe proposta dal kNN e confidenza, solo se ``hub.show_suggestion`` è attivo."""
        if not self.cfg.hub.show_suggestion:
            return None
        with self.lock:
            pred = knn_predict(self.bank, [self.bank.index_of(image_id)], self.cfg.knn.k, self.cfg.knn.temperature)
        if not pred.classes or pred.probs.sum() == 0:
            return None
        return pred.labels[0], float(pred.confidence[0])

    def reject_classes(self) -> set[str]:
        return {self.store.resolve(self.store.normalize_label(c)) for c in self.cfg.labels.reject_classes}

    def final_clusters(self) -> dict[str, list[dict[str, Any]]]:
        """Cluster finali: etichetta umana se c'è, altrimenti quella predetta.

        Restano fuori le classi di scarto, i "Non so", le api senza predizione e le
        "non assegnate" (troppo lontane da ogni esempio etichettato).
        """
        with self.lock:
            reject = self.reject_classes()
            unknown = set(self.bank.unknown_indices().tolist())
            predicted = self.final_pred.labels
            confidence = self.final_pred.confidence
            clusters: dict[str, list[dict[str, Any]]] = {}
            for i, image_id in enumerate(self.bank.ids):
                human = self.bank.label_of(image_id)
                if i in unknown or (not human and self.unassigned[i]):
                    continue
                label = human or predicted[i]
                if label is None or label in reject:
                    continue
                clusters.setdefault(label, []).append({
                    "image_id": image_id,
                    "source": "human" if human else "predicted",
                    "confidence": 1.0 if human else float(confidence[i]),
                })
            return dict(sorted(clusters.items()))

    def assign_many(self, annotator: str, image_ids: Sequence[str], label: str, confidence: str = "") -> int:
        """Etichetta, corregge o rimuove a mano un gruppo di api (pagina dei cluster).

        Ogni modifica è un nuovo evento nello store. Poi ricalcola le predizioni, così i
        cluster riflettono subito il cambiamento. Restituisce quante api sono cambiate.
        """
        with self.lock:
            label = self.store.normalize_label(label)
            if not label:
                raise ALError("etichetta vuota")
            image_ids = list(dict.fromkeys(image_ids))
            for image_id in image_ids:  # si controlla tutto prima di scrivere qualcosa
                if image_id not in self.bank:
                    raise ALError(f"immagine sconosciuta: {image_id}")
                if image_id in self._val_set:
                    raise ALError(f"{image_id} è nel validation set: va etichettata in modalità validazione")
                if image_id in self._audit_ids:
                    raise ALError(f"{image_id} è nel campione di audit: va etichettata in modalità audit")
            resolved = self.store.resolve(label)
            changed = 0
            for image_id in image_ids:
                current = self.bank.label_of(image_id)
                if current == resolved or (resolved == REMOVED_LABEL and current is None):
                    continue  # niente da cambiare: nessun evento inutile nello storico
                self.submit(annotator, image_id, label, confidence, manual=True)
                changed += 1
            if changed:
                self.recompute()
            return changed

    def labeled_samples(self, n: int = 12) -> list[dict[str, Any]]:
        """Per ogni classe, le ultime ``n`` api etichettate a mano; in fondo i "Non so"."""
        with self.lock:
            reject = self.reject_classes()
            groups: dict[str, list[Any]] = {}
            for image_id, ev in self.store.current(is_training_event).items():
                if image_id in self.bank and image_id not in self._val_set:
                    groups.setdefault(ev.label, []).append(ev)
            out = []
            for label in sorted(groups, key=lambda name: (name == UNKNOWN_LABEL, name)):
                events = sorted(groups[label], key=lambda e: e.timestamp, reverse=True)
                out.append({
                    "label": label,
                    "unknown": label == UNKNOWN_LABEL,
                    "reject": label in reject,
                    "n_human": len(events),
                    "samples": [
                        {"image_id": e.image_id, "annotator": e.annotator, "confidence": e.confidence, "query_type": e.query_type}
                        for e in events[:n]
                    ],
                })
            return out

    def _not_validation(self) -> np.ndarray:
        mask = np.ones(len(self.bank), dtype=bool)
        mask[[self.bank.index_of(i) for i in self._val_set if i in self.bank]] = False
        return mask

    def cluster_samples(self, n: int = 12, order: str = "random") -> list[dict[str, Any]]:
        """Per ogni cluster, ``n`` api assegnate dal modello (non dall'annotatore).

        Le predizioni sono quelle dell'ultimo ricalcolo, quindi la pagina cambia a ogni batch.
        Le api del validation set non vengono mostrate: si etichettano solo nella loro modalità.
        ``order``: ``random`` (campione fisso fino al prossimo round), ``confident`` o ``uncertain``.
        """
        with self.lock:
            pred = self.final_pred
            if not pred.classes:
                return []
            assigned = pred.probs.argmax(axis=1)
            confidence = pred.confidence
            predicted = pred.probs.sum(axis=1) > 0
            predicted[self.bank.labeled_indices()] = False
            predicted[self.bank.unknown_indices()] = False
            predicted &= ~self.unassigned & self._not_validation() & ~self._audit_mask()
            human = self.bank.class_counts()
            reject = self.reject_classes()
            rng = np.random.default_rng([self.cfg.seed, self.round])
            clusters = []
            for j, name in enumerate(pred.classes):
                members = np.flatnonzero(predicted & (assigned == j))
                if order == "random":
                    chosen = rng.choice(members, size=min(n, len(members)), replace=False)
                    chosen = chosen[np.argsort(-confidence[chosen], kind="stable")]
                else:
                    sign = 1.0 if order == "uncertain" else -1.0
                    chosen = members[np.argsort(sign * confidence[members], kind="stable")[:n]]
                clusters.append({
                    "label": name,
                    "reject": name in reject,
                    "n_human": human.get(name, 0),
                    "n_predicted": int(len(members)),
                    "mean_confidence": float(confidence[members].mean()) if len(members) else None,
                    "samples": [{"image_id": self.bank.ids[i], "confidence": float(confidence[i])} for i in chosen],
                })
            return clusters

    def unassigned_samples(self, n: int = 12, order: str = "random") -> dict[str, Any]:
        """``n`` api del gruppo "non assegnate", con la similarità all'esempio etichettato più vicino.

        ``order``: ``random``, ``uncertain`` (le più lontane) o ``confident`` (le meno lontane).
        """
        with self.lock:
            members = np.flatnonzero(self.unassigned & ~self._audit_mask())
            if not len(members):
                return {"n": 0, "samples": []}
            sims = self.nearest_sim[members]
            if order == "random":
                rng = np.random.default_rng([self.cfg.seed, self.round, 1])
                pick = rng.choice(len(members), size=min(n, len(members)), replace=False)
                pick = pick[np.argsort(sims[pick], kind="stable")]
            else:
                pick = np.argsort(sims if order == "uncertain" else -sims, kind="stable")[:n]
            nearest = self.final_pred.labels
            return {
                "n": int(len(members)),
                "samples": [
                    {"image_id": self.bank.ids[members[j]], "similarity": float(sims[j]), "nearest_class": nearest[members[j]]}
                    for j in pick
                ],
            }

    # -------------------------------- audit --------------------------------- #

    def _restore_audit(self, known_ids: set[str]) -> None:
        """All'avvio riprende l'audit attivo salvato, altrimenti l'ultimo snapshot che ha un audit set."""
        name = ""
        path = self.cfg.results_dir / "active_audit.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                name = json.load(f).get("snapshot", "")
        candidates = [name] if name else [s["name"] for s in snapshot.list_snapshots(self.cfg) if s["has_audit"]]
        for candidate in candidates:
            folder = snapshot.snapshots_dir(self.cfg) / candidate
            if (folder / audit.AUDIT_FILE).exists():
                self.audit = audit.load_audit_set(folder)
                self._audit_ids = {b["image_id"] for b in self.audit["bees"]} & known_ids
                return

    def _reload_audit_labels(self) -> None:
        self.audit_labels = {}
        if self.audit is None:
            return
        batch_id = audit.batch_id_of(self.audit["snapshot"])
        answers = self.store.per_annotator(lambda e: e.query_type == "audit" and e.batch_id == batch_id)
        self.audit_labels = {
            image_id: {annotator: ev.label for annotator, ev in by_annotator.items()}
            for image_id, by_annotator in answers.items() if image_id in self._audit_ids
        }

    def _audit_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.bank), dtype=bool)
        mask[[self.bank.index_of(i) for i in self._audit_ids if i in self.bank]] = True
        return mask

    def create_audit(self, snapshot_name: str, anomaly_scores: Optional[dict[str, float]] = None) -> dict[str, Any]:
        """Estrae il campione di controllo di uno snapshot (una volta sola) e lo rende l'audit attivo."""
        with self.lock:
            try:
                folder = snapshot.snapshot_path(self.cfg, snapshot_name)
                if not (folder / "predictions.csv").exists():
                    raise ValueError(f"snapshot '{snapshot_name}' does not exist")
                audit.create_audit_set(folder, self.cfg, anomaly_scores)
            except (ValueError, FileExistsError) as e:
                raise ALError(str(e)) from None
            return self.load_audit(snapshot_name)

    def load_audit(self, snapshot_name: str) -> dict[str, Any]:
        """Rende attivo l'audit di uno snapshot e lo ricorda in ``results_dir/active_audit.json``.

        Le api del campione escono dal batch in corso e non vengono più proposte in
        modalità normale: si etichettano solo in modalità audit, alla cieca.
        """
        with self.lock:
            try:
                folder = snapshot.snapshot_path(self.cfg, snapshot_name)
            except ValueError as e:
                raise ALError(str(e)) from None
            if not (folder / audit.AUDIT_FILE).exists():
                raise ALError(f"snapshot '{snapshot_name}' has no audit set")
            self.audit = audit.load_audit_set(folder)
            self._audit_ids = {b["image_id"] for b in self.audit["bees"] if b["image_id"] in self.bank}
            self._audit_assigned = {}
            self._reload_audit_labels()
            self.batch = [b for b in self.batch if b.done or b.image_id not in self._audit_ids]
            path = self.cfg.results_dir / "active_audit.json"
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"snapshot": snapshot_name}, f)
            tmp.replace(path)
            if self.batch and all(b.done for b in self.batch):
                self.refresh()
            return self.audit_progress()

    def _audit_done(self, bee: dict[str, Any]) -> bool:
        """Un'ape singola è completa con una risposta, una doppia con due annotatori diversi."""
        return len(self.audit_labels.get(bee["image_id"], {})) >= (2 if bee["double"] else 1)

    def _next_audit(self, annotator: str, now: float) -> Optional[BatchItem]:
        if self.audit is None:
            return None
        timeout = self.cfg.hub.assignment_timeout_s
        by_id = {b["image_id"]: b for b in self.audit["bees"]}
        free: Optional[str] = None
        for image_id in self.audit["order"]:
            bee = by_id[image_id]
            answers = self.audit_labels.get(image_id, {})
            if image_id not in self._audit_ids or self._audit_done(bee) or annotator in answers:
                continue  # lo stesso annotatore non riceve due volte la stessa ape
            owners = {a: t for a, t in self._audit_assigned.get(image_id, {}).items()
                      if now - t < timeout and a not in answers}
            if annotator in owners:
                free = image_id
                break
            slots = (2 if bee["double"] else 1) - len(answers)
            if free is None and len(owners) < slots:
                free = image_id
        if free is None:
            return None
        self._audit_assigned.setdefault(free, {})[annotator] = now
        batch_id = audit.batch_id_of(self.audit["snapshot"])
        return BatchItem(free, self.bank.index_of(free), "audit", batch_id, annotator, now)

    def audit_label_of(self, image_id: str, annotator: str) -> Optional[str]:
        with self.lock:
            return self.audit_labels.get(image_id, {}).get(annotator)

    def audit_progress(self) -> Optional[dict[str, Any]]:
        """Avanzamento dell'audit attivo: totale, completate, per strato, annotatori. ``None`` senza audit."""
        with self.lock:
            if self.audit is None:
                return None
            strata = {s["id"]: {"id": s["id"], "N_h": s["N_h"], "n_h": s["n_h"], "done": 0} for s in self.audit["strata"]}
            done = 0
            annotators: dict[str, int] = {}
            for bee in self.audit["bees"]:
                for annotator in self.audit_labels.get(bee["image_id"], {}):
                    annotators[annotator] = annotators.get(annotator, 0) + 1
                if self._audit_done(bee):
                    done += 1
                    strata[bee["stratum"]]["done"] += 1
            return {
                "snapshot": self.audit["snapshot"],
                "total": len(self.audit["bees"]),
                "done": done,
                "double": sum(b["double"] for b in self.audit["bees"]),
                "population": self.audit["population"],
                "strata": list(strata.values()),
                "annotators": dict(sorted(annotators.items())),
            }

    # ------------------------------- snapshot ------------------------------- #

    def export_snapshot(self, name: str, annotator: str) -> Path:
        """Congela predizioni, frame di origine e stato in ``results_dir/snapshots/<name>/``.

        Lo snapshot è immutabile: un nome già usato o non valido solleva ``ALError``.
        """
        with self.lock:
            try:
                return snapshot.export_snapshot(self, name, annotator)
            except (ValueError, FileExistsError) as e:
                raise ALError(str(e)) from None

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self.lock:
            return snapshot.list_snapshots(self.cfg)

    def status(self) -> dict[str, Any]:
        with self.lock:
            q = self.cfg.query
            last = self.history[-1] if self.history else {}
            return {
                "round": self.round,
                "model_version": self.model_version,
                "config_id": self.cfg.config_id,
                "n_images": len(self.bank),
                "n_labels": self.bank.n_labeled,
                "n_unknown": int(self.bank.unknown_indices().size),
                "n_unassigned": int(self.unassigned.sum()),
                "class_counts": self.bank.class_counts(),
                "reject_classes": sorted(self.reject_classes()),
                "stopping": self.stopping_status(),
                "snapshots": snapshot.list_snapshots(self.cfg),
                "audit": self.audit_progress(),
                "cold_start": is_cold_start(self.bank.n_labeled, len(self.bank.classes), q),
                "cold_start_threshold": q.cold_start_threshold,
                "batch_id": self.batch_id,
                "batch_size": len(self.batch),
                "batch_done": sum(b.done for b in self.batch),
                "validation_size": len(self.validation_ids),
                "validation_labeled": len(self.val_labels),
                "metrics": {k: last.get(k) for k in ("accuracy", "macro_f1", "knn_accuracy", "knn_macro_f1", "n_validation")},
                "history": list(self.history),
                "retrain": self.retrain.status() if self.retrain is not None else None,
            }
