"""Review of suspicious labels: hand-labelled bees that their neighbours contradict.

Idea from *confident learning* (Northcutt et al., 2021): a label is suspicious when the
model, without being able to see it, confidently assigns a different class to that bee. Here the
"model" is the k-NN over the other labelled bees. Suspects are shown to a human,
who confirms or corrects them: nothing is corrected automatically.
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
CONFIDENCE_ORDER = {"Low": 0, "Medium": 1}   # labels given with low confidence are reviewed first


@dataclass
class Suspect:
    image_id: str
    index: int
    label: str                 # current human label
    suggested: str             # class most voted by the neighbours, different from label
    p_suggested: float         # k-NN probability of the suggested class
    p_label: float             # k-NN probability of the human label
    annotator: str
    confidence: str            # confidence declared by the annotator (High/Medium/Low/"")
    query_type: str
    timestamp: str
    neighbors: list[dict]      # the most similar labelled neighbours: image_id, label, similarity
    rare_class: bool           # label has fewer than review.min_class_size human labels


def find_suspects(bank: MemoryBank, store: LabelStore, cfg: ALConfig, exclude_ids: Iterable[str] = ()) -> list[Suspect]:
    """Training labels contradicted by the most similar labelled neighbours.

    **Only training labels** are checked: bees with a human label in the
    memory bank, including those of reject classes. Deliberately left out:

    - **validation set and audit**: they are used to measure the model. Correcting them based on
      the model's opinion would pull them towards its predictions and make it look
      better than it is. For them the only check allowed is inter-annotator agreement,
      already built into the audit;
    - bees set aside with "Scarta ape": they have no class to verify.

    For each labelled bee, the ``review.k`` labelled neighbours vote (``knn.k`` if 0),
    weighted by ``exp(similarity / knn.temperature)`` as in the hub's k-NN, **excluding the bee
    itself**. It is suspicious if the most-voted class differs from its label and has
    probability at least ``review.min_disagreement``.

    Classes with fewer than ``review.min_class_size`` labels should be read separately
    (``rare_class``): their neighbours are almost all from other classes even when the label
    is correct, which is exactly the case of newly discovered rare phenotypes.

    Order: first labels given with ``Low`` confidence, then ``Medium``, then the rest;
    ties broken by decreasing disagreement, then by ``image_id``. All with a single neighbour search.
    """
    r = cfg.review
    excluded = set(exclude_ids)
    idx = np.array([i for i in bank.labeled_indices() if bank.ids[i] not in excluded], dtype=np.int64)
    classes = bank.classes
    if len(idx) == 0 or len(classes) < 2:
        return []
    k = r.k or cfg.knn.k
    sims, nb = bank.search_labeled(bank.X[idx], max(k, r.show_neighbors) + 1)
    sims = np.where(nb == idx[:, None], -np.inf, sims.astype(np.float64))   # a bee does not vote for itself
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
    """Review decisions, in the order they were made."""
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
    """Confirmations still valid: ``{image_id: (confirmed label, class that was suggested)}``.

    A later correction or "Scarta ape" cancels the confirmation for that bee.
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
