"""Revisione delle etichette sospette: api etichettate a mano che i vicini smentiscono.

Idea del *confident learning* (Northcutt et al., 2021): un'etichetta è sospetta quando il
modello, senza poterla guardare, assegna con forza un'altra classe a quell'ape. Qui il
"modello" è il k-NN sulle altre api etichettate. Le sospette vengono mostrate a un umano,
che conferma o corregge: niente viene corretto in automatico.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np

from .config import ALConfig
from .label_store import LabelStore, is_training_event
from .memory_bank import MemoryBank

DECISIONS = ("confirmed", "corrected", "set_aside")
CONFIDENCE_ORDER = {"Low": 0, "Medium": 1}   # le etichette date con poca sicurezza si rivedono per prime


@dataclass
class Suspect:
    image_id: str
    index: int
    label: str                 # etichetta umana corrente
    suggested: str             # classe più votata dai vicini, diversa da label
    p_suggested: float         # probabilità k-NN della classe suggerita
    p_label: float             # probabilità k-NN dell'etichetta umana
    annotator: str
    confidence: str            # confidenza dichiarata dall'annotatore (High/Medium/Low/"")
    query_type: str
    timestamp: str
    neighbors: list[dict]      # i vicini etichettati più simili: image_id, label, similarità
    rare_class: bool           # label ha meno di review.min_class_size etichette umane


def find_suspects(bank: MemoryBank, store: LabelStore, cfg: ALConfig, exclude_ids: Iterable[str] = ()) -> list[Suspect]:
    """Etichette di training che i vicini etichettati più simili smentiscono.

    Si controllano **solo le etichette di training**: le api con un'etichetta umana nella
    memory bank, comprese quelle di classi di scarto. Restano fuori, apposta:

    - **validation set e audit**: servono a misurare il modello. Correggerli guardando
      l'opinione del modello li avvicinerebbe alle sue predizioni e lo farebbe sembrare
      migliore di quanto è. Per loro l'unico controllo ammesso è l'accordo tra annotatori,
      già previsto nell'audit;
    - le api messe da parte con "Scarta ape": non hanno una classe da verificare.

    Per ogni ape etichettata si vota con i ``review.k`` vicini etichettati (``knn.k`` se 0),
    pesati con ``exp(similarità / knn.temperature)`` come nel k-NN dell'hub, **esclusa l'ape
    stessa**. È sospetta se la classe più votata è diversa dalla sua etichetta e ha
    probabilità almeno ``review.min_disagreement``.

    Le classi con meno di ``review.min_class_size`` etichette vanno lette a parte
    (``rare_class``): i loro vicini sono quasi tutti di altre classi anche quando l'etichetta
    è giusta, ed è proprio il caso dei fenotipi rari appena scoperti.

    Ordine: prima le etichette date con confidenza ``Low``, poi ``Medium``, poi le altre;
    a parità, disaccordo decrescente; poi ``image_id``. Tutto con una sola ricerca dei vicini.
    """
    r = cfg.review
    excluded = set(exclude_ids)
    idx = np.array([i for i in bank.labeled_indices() if bank.ids[i] not in excluded], dtype=np.int64)
    classes = bank.classes
    if len(idx) == 0 or len(classes) < 2:
        return []
    k = r.k or cfg.knn.k
    sims, nb = bank.search_labeled(bank.X[idx], max(k, r.show_neighbors) + 1)
    sims = np.where(nb == idx[:, None], -np.inf, sims.astype(np.float64))   # un'ape non vota per se stessa
    order = np.argsort(-sims, axis=1, kind="stable")
    sims, nb = np.take_along_axis(sims, order, axis=1), np.take_along_axis(nb, order, axis=1)

    class_pos = {c: j for j, c in enumerate(classes)}
    nb_class = np.array([class_pos[lbl] for lbl in bank.labels_at(nb.ravel())]).reshape(nb.shape)
    vote_sims, vote_class = sims[:, :k], nb_class[:, :k]
    valid = np.isfinite(vote_sims)
    best = np.where(valid, vote_sims, -np.inf).max(axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    weights = np.where(valid, np.exp((np.where(valid, vote_sims, 0.0) - best) / cfg.knn.temperature), 0.0)
    probs = np.zeros((len(idx), len(classes)))
    np.add.at(probs, (np.repeat(np.arange(len(idx)), vote_class.shape[1]), vote_class.ravel()), weights.ravel())
    total = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, total, out=np.zeros_like(probs), where=total > 0)

    own = np.array([class_pos[lbl] for lbl in bank.labels_at(idx)])
    top = probs.argmax(axis=1)
    p_top = probs[np.arange(len(idx)), top]
    flagged = np.flatnonzero((top != own) & (p_top >= r.min_disagreement))
    if len(flagged) == 0:
        return []

    events = store.current(is_training_event)
    counts = bank.class_counts()
    suspects = []
    for row in flagged:
        image_id = bank.ids[idx[row]]
        label = classes[own[row]]
        ev = events.get(image_id)
        shown = [j for j in range(sims.shape[1]) if np.isfinite(sims[row, j])][:r.show_neighbors]
        suspects.append(Suspect(
            image_id=image_id, index=int(idx[row]), label=label, suggested=classes[top[row]],
            p_suggested=float(p_top[row]), p_label=float(probs[row, own[row]]),
            annotator=ev.annotator if ev else "", confidence=ev.confidence if ev else "",
            query_type=ev.query_type if ev else "", timestamp=ev.timestamp if ev else "",
            neighbors=[{"image_id": bank.ids[nb[row, j]], "label": classes[nb_class[row, j]], "similarity": float(sims[row, j])}
                       for j in shown],
            rare_class=counts.get(label, 0) < r.min_class_size,
        ))
    suspects.sort(key=lambda s: (s.rare_class, CONFIDENCE_ORDER.get(s.confidence, 2), -s.p_suggested, s.image_id))
    return suspects


def read_log(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Decisioni di revisione, nell'ordine in cui sono state prese."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_log(path: Union[str, Path], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def confirmations(log: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Conferme ancora valide: ``{image_id: (etichetta confermata, classe che veniva suggerita)}``.

    Una correzione o un "Scarta ape" successivi annullano la conferma di quell'ape.
    """
    out: dict[str, tuple[str, str]] = {}
    for row in log:
        if row.get("decision") == "confirmed":
            out[row["image_id"]] = (row.get("old_label", ""), row.get("suggested", ""))
        else:
            out.pop(row.get("image_id", ""), None)
    return out


def as_dict(suspect: Suspect) -> dict[str, Any]:
    return dataclasses.asdict(suspect)
