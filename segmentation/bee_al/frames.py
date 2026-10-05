"""Collegamento ape → frame di origine, dai manifest dell'estrazione e della pulizia.

``extract_anomaly_dataset.py`` numera i crop con un contatore che cresce di 1 per ogni ape,
frame dopo frame, e annota in ``raw_bees/_extraction_manifest.json`` quante api ha salvato
per ogni frame (``processed_sources``, in ordine di elaborazione) e il prossimo id libero
(``next_bee_id``). Sommando i conteggi nell'ordine del file si ricava, per ogni frame,
l'intervallo ``[start, end)`` degli id delle sue api. ``dataset_cleaning.py`` scarta dei crop
ma non li rinumera, quindi il nome del file pulito basta per risalire al frame.
"""
from __future__ import annotations

import bisect
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .config import ID_SEPARATOR, ALConfig

BEE_ID_RE = re.compile(r"^bee_(\d+)_")
EXTRACTION_MANIFEST = "_extraction_manifest.json"
CLEANING_MANIFEST = "_cleaning_manifest.json"


@dataclass
class FrameMap:
    """Intervalli di id delle api per ogni frame di una cartella di dataset."""

    frames: list[str] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    reliable: bool = True
    path: str = ""

    def frame_of(self, filename: str) -> Optional[str]:
        """Frame da cui viene il crop ``bee_<id>_...``.

        ``None`` se il nome non ha quella forma, se l'id è fuori da tutti gli intervalli
        o se la mappa non è affidabile: in quel caso non si tira a indovinare.
        """
        m = BEE_ID_RE.match(Path(filename).name)
        if m is None or not self.reliable:
            return None
        bee_id = int(m.group(1))
        i = bisect.bisect_right(self.starts, bee_id) - 1
        if i < 0 or bee_id >= self.ends[i]:
            return None
        return self.frames[i]

    def raw_bees_per_frame(self) -> dict[str, int]:
        """Api grezze (prima della pulizia) estratte da ogni frame."""
        return {f: e - s for f, s, e in zip(self.frames, self.starts, self.ends)}


def load_frame_map(manifest_path: Union[str, Path]) -> FrameMap:
    """Costruisce gli intervalli di id scorrendo ``processed_sources`` nell'ordine del file.

    Se la somma dei conteggi non coincide con ``next_bee_id`` la numerazione non è quella
    attesa (manifest modificato a mano, estrazioni mescolate, ...): la mappa viene marcata
    ``reliable = False`` con un warning e ``frame_of`` non restituisce più nulla.
    """
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    fmap = FrameMap(path=str(path))
    cursor = 0
    for frame, info in manifest.get("processed_sources", {}).items():
        n = int(info["bees"] if isinstance(info, dict) else info)
        fmap.frames.append(frame)
        fmap.starts.append(cursor)
        cursor += n
        fmap.ends.append(cursor)
    expected = manifest.get("next_bee_id")
    if expected is None or int(expected) != cursor:
        fmap.reliable = False
        warnings.warn(
            f"{path}: la somma delle api per frame ({cursor}) non coincide con next_bee_id ({expected}); "
            "il collegamento ape -> frame di questa cartella non viene usato.",
            stacklevel=2,
        )
    return fmap


def discards_per_frame(cleaning_manifest_path: Union[str, Path], frame_map: FrameMap) -> dict[str, dict[str, Any]]:
    """Per ogni frame: crop accettati e scartati dalla pulizia, e i motivi di scarto.

    Restituisce ``{frame: {"accepted": n, "discarded": n, "reasons": {motivo: n}}}``;
    dizionario vuoto se il manifest della pulizia non esiste.
    """
    path = Path(cleaning_manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    reasons = manifest.get("discard_reasons", {})
    out: dict[str, dict[str, Any]] = {}
    for filename, status in manifest.get("processed_files", {}).items():
        frame = frame_map.frame_of(filename)
        if frame is None:
            continue
        row = out.setdefault(frame, {"accepted": 0, "discarded": 0, "reasons": {}})
        if status == "accepted":
            row["accepted"] += 1
        else:
            row["discarded"] += 1
            reason = reasons.get(filename, "unknown")
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
    return out


def dataset_folders(cfg: ALConfig) -> dict[str, Path]:
    """Cartelle del dataset per nome: quello di ``paths.dataset_dirs``, o il nome della cartella se è una sola."""
    if cfg.paths.dataset_dirs:
        return {name: cfg.resolve(d) for name, d in cfg.paths.dataset_dirs.items()}
    folder = cfg.dataset_dir
    return {folder.parent.name if folder.name == "cleaned_bees" else folder.name: folder}


def dataset_of(image_id: str, cfg: ALConfig) -> str:
    """Nome della cartella di dataset a cui appartiene un image_id."""
    if cfg.paths.dataset_dirs:
        name, sep, _ = image_id.partition(ID_SEPARATOR)
        if sep and name in cfg.paths.dataset_dirs:
            return name
        return ""
    return next(iter(dataset_folders(cfg)))


def extraction_manifest_path(cfg: ALConfig, dataset: str, folder: Path) -> Path:
    """Manifest dell'estrazione: quello in ``frames.manifests``, altrimenti ``<cartella pulita>/../raw_bees/``."""
    if dataset in cfg.frames.manifests:
        return cfg.resolve(cfg.frames.manifests[dataset])
    return folder.parent / "raw_bees" / EXTRACTION_MANIFEST


def load_frame_maps(cfg: ALConfig) -> dict[str, Optional[FrameMap]]:
    """Una ``FrameMap`` per cartella di dataset; ``None`` dove il manifest non si trova."""
    maps: dict[str, Optional[FrameMap]] = {}
    for dataset, folder in dataset_folders(cfg).items():
        path = extraction_manifest_path(cfg, dataset, folder)
        maps[dataset] = load_frame_map(path) if path.exists() else None
    return maps


def frame_groups(frame: str, dataset: str, pattern: str = "") -> tuple[str, str]:
    """Sessione e alveare di un frame: ``(session, colony)``.

    ``pattern`` è una regex con i gruppi nominati ``session`` e/o ``colony`` cercata nel nome
    del frame. Senza regex, o se non corrisponde, la sessione è il nome della cartella di
    dataset e l'alveare è vuoto: dai soli nomi dei file non si inventa nulla.
    """
    session, colony = dataset, ""
    if pattern and frame:
        m = re.search(pattern, frame)
        if m:
            groups = m.groupdict()
            session = groups.get("session") or session
            colony = groups.get("colony") or ""
    return session, colony
