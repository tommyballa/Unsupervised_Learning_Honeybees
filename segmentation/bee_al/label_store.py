"""Append-only label store (``al_labels.csv``) and import of legacy labels."""
from __future__ import annotations

import csv
import dataclasses
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

from . import REMOVED_LABEL, UNKNOWN_LABEL
from .hub_compat import load_hub_helpers

FIELDS = [
    "image_id", "label", "timestamp", "annotator", "confidence",
    "query_type", "config_id", "model_version", "batch_id",
]
QUERY_TYPES = ("random", "diversity", "uncertainty", "novelty", "manual", "validation", "audit", "holdout", "legacy")
HELD_OUT_TYPES = ("validation", "audit", "holdout")  # events that never enter the memory bank
SENTINELS = (UNKNOWN_LABEL, REMOVED_LABEL)
# Typed by hand in the free-text field it counts as the "Scarta ape" (discard bee) button ("non so" was the old name).
UNKNOWN_TEXTS = ("scarta ape", "non so")


@dataclass(frozen=True)
class LabelEvent:
    image_id: str
    label: str
    timestamp: str = ""
    annotator: str = ""
    confidence: str = ""
    query_type: str = ""
    config_id: str = ""
    model_version: str = ""
    batch_id: str = ""


def is_training_event(event: LabelEvent) -> bool:
    """True for the events that feed the memory bank (all except the held-out ones)."""
    return event.query_type not in HELD_OUT_TYPES


class LabelStore:
    """Labelling events in an append-only CSV.

    Corrections and removals are new events: for each image the last one wins,
    and ``REMOVED_LABEL`` counts as a removal. Aliases are applied on read,
    so renaming or merging classes never touches the history.
    """

    def __init__(
        self,
        path: Union[str, Path],
        aliases: Optional[dict[str, str]] = None,
        normalize: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.path = Path(path)
        self._clean = normalize or load_hub_helpers().clean_label
        self._lock = threading.Lock()
        self.aliases: dict[str, str] = {}
        self.set_aliases(aliases or {})

    # -------------------------------- labels -------------------------------- #

    def normalize_label(self, raw: str) -> str:
        """Same normalisation as the hub (``clean_label``); sentinels pass through unchanged."""
        raw = (raw or "").strip()
        if raw in SENTINELS:
            return raw
        label = self._clean(raw)
        return UNKNOWN_LABEL if label.lower() in UNKNOWN_TEXTS else label

    def set_aliases(self, aliases: dict[str, str]) -> None:
        self.aliases = {self.normalize_label(k): self.normalize_label(v) for k, v in aliases.items()}

    def resolve(self, label: str) -> str:
        """Apply the aliases (including chained ones) to an already-normalised label."""
        seen = {label}
        while label in self.aliases:
            label = self.aliases[label]
            if label in seen:
                break
            seen.add(label)
        return label

    # -------------------------------- writing ------------------------------- #

    def append(
        self,
        image_id: str,
        label: str,
        annotator: str = "",
        confidence: str = "",
        query_type: str = "",
        config_id: str = "",
        model_version: str = "",
        batch_id: str = "",
        timestamp: Optional[str] = None,
    ) -> LabelEvent:
        """Append an event. The label is saved normalised but without aliases."""
        label = self.normalize_label(label)
        if not label:
            raise ValueError("empty label")
        if query_type not in QUERY_TYPES:
            raise ValueError(f"unknown query_type: {query_type!r}")
        event = LabelEvent(
            image_id=image_id,
            label=label,
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            annotator=annotator,
            confidence=confidence,
            query_type=query_type,
            config_id=config_id,
            model_version=model_version,
            batch_id=batch_id,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(dataclasses.asdict(event))
        return event

    # -------------------------------- reading ------------------------------- #

    def events(self) -> list[LabelEvent]:
        """All events in the order they were written (labels without aliases)."""
        if not self.path.exists():
            return []
        with self._lock, open(self.path, "r", newline="", encoding="utf-8") as f:
            return [LabelEvent(**{k: (row.get(k) or "") for k in FIELDS}) for row in csv.DictReader(f)]

    def current(self, keep: Optional[Callable[[LabelEvent], bool]] = None) -> dict[str, LabelEvent]:
        """Current state: last event per image, with aliases applied and removals dropped.

        ``keep`` filters the events before deciding which one wins (e.g. only validation ones).
        """
        latest: dict[str, LabelEvent] = {}
        for ev in self.events():
            if keep is None or keep(ev):
                latest[ev.image_id] = ev
        out: dict[str, LabelEvent] = {}
        for image_id, ev in latest.items():
            label = self.resolve(ev.label)
            if label != REMOVED_LABEL:
                out[image_id] = dataclasses.replace(ev, label=label)
        return out

    def per_annotator(self, keep: Optional[Callable[[LabelEvent], bool]] = None) -> dict[str, dict[str, LabelEvent]]:
        """Labels per annotator: ``{image_id: {annotator: last event of that annotator}}``.

        Unlike ``current``, the last event on the image does not win: each annotator
        keeps their own answer (needed for the audit's double labelling). Annotators
        are in the order in which they first answered; aliases applied, removals
        dropped.
        """
        out: dict[str, dict[str, LabelEvent]] = {}
        for ev in self.events():
            if keep is not None and not keep(ev):
                continue
            label = self.resolve(ev.label)
            answers = out.setdefault(ev.image_id, {})
            if label == REMOVED_LABEL:
                answers.pop(ev.annotator, None)
            else:
                answers[ev.annotator] = dataclasses.replace(ev, label=label)
        return {image_id: answers for image_id, answers in out.items() if answers}

    # -------------------------------- legacy -------------------------------- #

    def import_legacy(self, results_dir: Union[str, Path], config_id: str = "") -> int:
        """Align the store with the hub's ``single_labels.csv`` and ``grid_batch_labels.csv``.

        Idempotent. A legacy label never overwrites a label given in active
        learning; a label removed from the CSVs (``/clean`` page) becomes a
        removal event. Returns the number of events added.
        """
        legacy = read_legacy_labels(results_dir)
        if legacy is None:
            return 0
        latest: dict[str, LabelEvent] = {}
        for ev in self.events():
            if is_training_event(ev):
                latest[ev.image_id] = ev

        added = 0
        for image_id, row in legacy.items():
            label = self.normalize_label(row["label"])
            prev = latest.get(image_id)
            if not label or (prev and (prev.query_type != "legacy" or prev.label == label)):
                continue
            self.append(image_id, label, annotator=row["annotator"], confidence=row["confidence"],
                        query_type="legacy", config_id=config_id, timestamp=row["timestamp"] or None)
            added += 1
        for image_id, prev in latest.items():
            if prev.query_type == "legacy" and prev.label != REMOVED_LABEL and image_id not in legacy:
                self.append(image_id, REMOVED_LABEL, query_type="legacy", config_id=config_id)
                added += 1
        return added


def read_legacy_labels(results_dir: Union[str, Path]) -> Optional[dict[str, dict[str, str]]]:
    """Labels from the hub's historical CSVs: ``image_id -> {label, timestamp, annotator, confidence}``.

    Uses the hub's ``parse_shifted_row_*`` for rows with misaligned columns.
    If an image appears more than once the most recent row wins. ``None`` if the CSVs do not exist.
    """
    helpers = load_hub_helpers()
    d = Path(results_dir)
    rows: list[dict[str, str]] = []
    found = False

    def annotator_of(row: dict, first_col: str) -> str:
        for key in ("user_email", first_col):
            value = str(row.get(key) or "")
            if "@" in value or value == "anonymous":
                return value
        return ""

    single = d / "single_labels.csv"
    if single.is_file():
        found = True
        with open(single, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img, lbl = helpers.parse_shifted_row_single(row)
                shifted = img != str(row.get("image_name", ""))
                if img and lbl:
                    rows.append({
                        "image_id": img.strip(), "label": lbl,
                        "timestamp": str(row.get("timestamp") or ""),
                        "annotator": annotator_of(row, "image_name"),
                        "confidence": "" if shifted else str(row.get("confidence") or ""),
                    })

    grid = d / "grid_batch_labels.csv"
    if grid.is_file():
        found = True
        with open(grid, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                imgs, lbl = helpers.parse_shifted_row_grid(row)
                if not (imgs and lbl):
                    continue
                for img in imgs.split(";"):
                    if img.strip():
                        rows.append({
                            "image_id": img.strip(), "label": lbl,
                            "timestamp": str(row.get("timestamp") or ""),
                            "annotator": annotator_of(row, "selected_images"),
                            "confidence": "",
                        })

    if not found:
        return None
    rows.sort(key=lambda r: r["timestamp"])  # stable sort: on ties the last row read wins
    return {r["image_id"]: r for r in rows}
