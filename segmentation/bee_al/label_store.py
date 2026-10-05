"""Store append-only delle etichette (``al_labels.csv``) e import delle etichette legacy."""
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
QUERY_TYPES = ("random", "diversity", "uncertainty", "novelty", "manual", "validation", "audit", "legacy")
HELD_OUT_TYPES = ("validation", "audit")  # eventi che non entrano mai nella memory bank
SENTINELS = (UNKNOWN_LABEL, REMOVED_LABEL)
# Scritto a mano nel campo libero vale come il pulsante "Scarta ape" ("non so" era il vecchio nome).
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
    """Vero per gli eventi che alimentano la memory bank (tutti tranne quelli tenuti da parte)."""
    return event.query_type not in HELD_OUT_TYPES


class LabelStore:
    """Eventi di etichettatura in un CSV in sola aggiunta.

    Correzioni e rimozioni sono nuovi eventi: per ogni immagine vince l'ultimo,
    e ``REMOVED_LABEL`` vale come rimozione. Gli alias sono applicati in lettura,
    quindi rinominare o unire classi non tocca mai lo storico.
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

    # ------------------------------ etichette ------------------------------ #

    def normalize_label(self, raw: str) -> str:
        """Stessa normalizzazione dell'hub (``clean_label``); le sentinelle passano invariate."""
        raw = (raw or "").strip()
        if raw in SENTINELS:
            return raw
        label = self._clean(raw)
        return UNKNOWN_LABEL if label.lower() in UNKNOWN_TEXTS else label

    def set_aliases(self, aliases: dict[str, str]) -> None:
        self.aliases = {self.normalize_label(k): self.normalize_label(v) for k, v in aliases.items()}

    def resolve(self, label: str) -> str:
        """Applica gli alias (anche in catena) a un'etichetta già normalizzata."""
        seen = {label}
        while label in self.aliases:
            label = self.aliases[label]
            if label in seen:
                break
            seen.add(label)
        return label

    # ------------------------------- scrittura ------------------------------ #

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
        """Aggiunge un evento. L'etichetta è salvata normalizzata ma senza alias."""
        label = self.normalize_label(label)
        if not label:
            raise ValueError("etichetta vuota")
        if query_type not in QUERY_TYPES:
            raise ValueError(f"query_type sconosciuto: {query_type!r}")
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

    # -------------------------------- lettura ------------------------------- #

    def events(self) -> list[LabelEvent]:
        """Tutti gli eventi nell'ordine in cui sono stati scritti (etichette senza alias)."""
        if not self.path.exists():
            return []
        with self._lock, open(self.path, "r", newline="", encoding="utf-8") as f:
            return [LabelEvent(**{k: (row.get(k) or "") for k in FIELDS}) for row in csv.DictReader(f)]

    def current(self, keep: Optional[Callable[[LabelEvent], bool]] = None) -> dict[str, LabelEvent]:
        """Stato attuale: ultimo evento per immagine, con alias applicati e rimozioni tolte.

        ``keep`` filtra gli eventi prima di decidere chi vince (es. solo quelli di validazione).
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
        """Etichette per annotatore: ``{image_id: {annotator: ultimo evento di quell'annotatore}}``.

        A differenza di ``current`` non vince l'ultimo evento sull'immagine: ogni annotatore
        conserva la propria risposta (serve per la doppia etichettatura dell'audit). Gli
        annotatori sono nell'ordine in cui hanno risposto la prima volta; alias applicati,
        rimozioni tolte.
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
        """Allinea lo store a ``single_labels.csv`` e ``grid_batch_labels.csv`` dell'hub.

        Idempotente. Un'etichetta legacy non sovrascrive mai un'etichetta data in
        active learning; un'etichetta tolta dai CSV (pagina ``/clean``) diventa un
        evento di rimozione. Restituisce il numero di eventi aggiunti.
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
    """Etichette dei CSV storici dell'hub: ``image_id -> {label, timestamp, annotator, confidence}``.

    Usa ``parse_shifted_row_*`` dell'hub per le righe con le colonne disallineate.
    Se un'immagine compare più volte vince la riga più recente. ``None`` se i CSV non esistono.
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
    rows.sort(key=lambda r: r["timestamp"])  # ordinamento stabile: a parità vince l'ultima riga letta
    return {r["image_id"]: r for r in rows}
