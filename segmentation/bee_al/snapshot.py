"""Snapshot immutabili delle predizioni: il punto fermo da cui partono audit e stime."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from . import UNKNOWN_LABEL
from .config import ALConfig
from .frames import CLEANING_MANIFEST, FrameGrouper, dataset_folders, dataset_of, discards_per_frame, load_frame_maps

if TYPE_CHECKING:
    from .engine import ALEngine

NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$")
PREDICTION_FIELDS = [
    "image_id", "path", "dataset", "filename", "frame", "session", "colony", "status", "final_label", "source",
    "confidence", "propagation_label", "propagation_confidence", "knn_label", "knn_confidence",
    "nearest_labeled_similarity",
]
FRAME_FIELDS = ["dataset", "frame", "session", "colony", "raw_bees", "cleaned_accepted", "cleaned_discarded", "discard_reasons"]
STATUSES = ("labeled", "predicted", "unknown", "unassigned", "rejected", "validation", "no_prediction")


def snapshots_dir(cfg: ALConfig) -> Path:
    return cfg.results_dir / "snapshots"


def snapshot_path(cfg: ALConfig, name: str) -> Path:
    """Cartella di uno snapshot; ``ValueError`` se il nome non è valido."""
    if not NAME_RE.match(name or ""):
        raise ValueError("snapshot name: use only letters, digits, '_', '-' and '.', and do not start with a dot")
    return snapshots_dir(cfg) / name


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def bee_rows(engine: "ALEngine", maps: Optional[dict[str, Any]] = None,
             grouper: Optional[FrameGrouper] = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Una riga per ogni ape della bank, più lo stato delle mappe dei frame.

    ``status``: ``validation`` (validation set), ``unknown`` ("Non so"), ``labeled`` (etichetta
    umana non di scarto), ``rejected`` (classe finale di scarto, umana o predetta),
    ``no_prediction`` (nessuna etichetta la raggiunge), ``unassigned`` (troppo lontana da ogni
    esempio), ``predicted`` (classe assegnata dal modello). ``final_label`` è l'etichetta umana
    se c'è, altrimenti quella predetta; resta vuota quando l'ape non appartiene a nessun cluster.
    ``maps`` e ``grouper`` si possono preparare prima, fuori dal lock: leggerli dal disco è lento.
    """
    cfg, bank = engine.cfg, engine.bank
    maps = load_frame_maps(cfg) if maps is None else maps
    grouper = FrameGrouper(cfg) if grouper is None else grouper
    reject = engine.reject_classes()
    final, knn = engine.final_pred, engine.knn_pred
    final_labels, final_conf = final.labels, final.confidence
    knn_labels, knn_conf = knn.labels, knn.confidence
    reached = final.probs.sum(axis=1) > 0 if final.classes else None
    knn_reached = knn.probs.sum(axis=1) > 0 if knn.classes else None
    unknown = set(bank.unknown_indices().tolist())
    validation = engine._val_set

    linked = {name: 0 for name in maps}
    rows: list[dict[str, str]] = []
    for i, image_id in enumerate(bank.ids):
        path = bank.paths[i]
        dataset = dataset_of(image_id, cfg)
        filename = Path(path).name
        fmap = maps.get(dataset)
        frame = (fmap.frame_of(filename) if fmap is not None else None) or ""
        if frame:
            linked[dataset] += 1
        session, colony = grouper.groups(frame, dataset)

        human = bank.label_of(image_id)
        has_pred = bool(reached is not None and reached[i])
        predicted = final_labels[i] if has_pred else None
        label, source, confidence = "", "", ""
        if image_id in validation:
            status = "validation"
        elif i in unknown:
            status = "unknown"
        elif human:
            status = "rejected" if human in reject else "labeled"
            label, source, confidence = human, "human", _fmt(1.0)
        elif predicted is None:
            status = "no_prediction"
        elif engine.unassigned[i]:
            status = "unassigned"
        else:
            status = "rejected" if predicted in reject else "predicted"
            label, source, confidence = predicted, "predicted", _fmt(float(final_conf[i]))

        has_knn = bool(knn_reached is not None and knn_reached[i])
        rows.append({
            "image_id": image_id, "path": path, "dataset": dataset, "filename": filename,
            "frame": frame, "session": session, "colony": colony,
            "status": status, "final_label": label, "source": source, "confidence": confidence,
            "propagation_label": predicted or "",
            "propagation_confidence": _fmt(float(final_conf[i])) if has_pred else "",
            "knn_label": (knn_labels[i] or "") if has_knn else "",
            "knn_confidence": _fmt(float(knn_conf[i])) if has_knn else "",
            "nearest_labeled_similarity": _fmt(float(engine.nearest_sim[i])) if engine.nearest_sim is not None else "",
        })

    n_by_dataset: dict[str, int] = {}
    for row in rows:
        n_by_dataset[row["dataset"]] = n_by_dataset.get(row["dataset"], 0) + 1
    frame_info = {
        name: {
            "manifest": fmap.path if fmap is not None else "",
            "found": fmap is not None,
            "reliable": bool(fmap is not None and fmap.reliable),
            "frames": len(fmap.frames) if fmap is not None else 0,
            "bees": n_by_dataset.get(name, 0),
            "bees_linked": linked[name],
        }
        for name, fmap in maps.items()
    }
    sessions: dict[str, int] = {}
    for row in rows:
        sessions[row["session"]] = sessions.get(row["session"], 0) + 1
    return rows, {"maps": maps, "info": frame_info, "sessions": dict(sorted(sessions.items())),
                  "session_dirs_missing": grouper.missing_dirs, "frames_in_several_sessions": sorted(grouper.ambiguous)}


def frame_rows(cfg: ALConfig, maps: dict[str, Any], grouper: Optional[FrameGrouper] = None) -> list[dict[str, Any]]:
    """Una riga per frame: api grezze estratte e api accettate/scartate dalla pulizia."""
    folders = dataset_folders(cfg)
    grouper = FrameGrouper(cfg) if grouper is None else grouper
    rows: list[dict[str, Any]] = []
    for dataset, fmap in maps.items():
        if fmap is None or not fmap.reliable:
            continue
        discards = discards_per_frame(folders[dataset] / CLEANING_MANIFEST, fmap)
        for frame, raw in fmap.raw_bees_per_frame().items():
            session, colony = grouper.groups(frame, dataset)
            d = discards.get(frame)
            rows.append({
                "dataset": dataset, "frame": frame, "session": session, "colony": colony, "raw_bees": raw,
                "cleaned_accepted": d["accepted"] if d else ("" if not discards else 0),
                "cleaned_discarded": d["discarded"] if d else ("" if not discards else 0),
                "discard_reasons": json.dumps(d["reasons"] if d else {}, sort_keys=True),
            })
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    # Prima in memoria e poi una sola scrittura: sul disco NTFS (fuse) tante scritture piccole costano il doppio
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())


def export_snapshot(engine: "ALEngine", name: str, annotator: str) -> Path:
    """Scrive lo snapshot in ``results_dir/snapshots/<name>/``.

    Il lock dell'engine si tiene solo per leggere lo stato (qualche secondo); la lettura delle
    cartelle dei frame e la scrittura dei file, che sul disco dei risultati durano minuti,
    avvengono fuori, così l'hub non si blocca.
    Gli snapshot sono immutabili: se il nome esiste già si solleva ``FileExistsError``.
    Si scrive in una cartella temporanea che viene rinominata solo alla fine, così
    un'esportazione interrotta non lascia uno snapshot a metà.
    """
    cfg = engine.cfg
    out = snapshot_path(cfg, name)
    if out.exists():
        raise FileExistsError(f"snapshot '{name}' already exists: snapshots are immutable, pick another name")
    maps, grouper = load_frame_maps(cfg), FrameGrouper(cfg)
    with engine.lock:
        rows, frames = bee_rows(engine, maps, grouper)
        n_events = len(engine.store.events())
        last = engine.history[-1] if engine.history else {}
        state = {
            "round": engine.round, "model_version": engine.model_version, "config_id": cfg.config_id,
            "config": cfg.to_dict(), "reject_classes": sorted(engine.reject_classes()),
            "stopping": engine.stopping_status(),
        }
        # Etichette e validation set copiati com'erano insieme alle predizioni
        copies = {src.name: src.read_bytes() for src in (cfg.al_labels_path, cfg.validation_path) if src.exists()}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".tmp-{name}-{os.getpid()}-{threading.get_ident()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        _write_csv(tmp / "predictions.csv", PREDICTION_FIELDS, rows)
        _write_csv(tmp / "frames.csv", FRAME_FIELDS, frame_rows(cfg, frames["maps"], grouper))

        by_status = {s: 0 for s in STATUSES}
        by_class: dict[str, dict[str, int]] = {}
        for row in rows:
            by_status[row["status"]] += 1
            if row["final_label"]:
                counts = by_class.setdefault(row["final_label"], {"human": 0, "predicted": 0})
                counts[row["source"]] += 1
        meta = {
            "name": name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "annotator": annotator,
            "round": state["round"],
            "model_version": state["model_version"],
            "config_id": state["config_id"],
            "config": state["config"],
            "n_bees": len(rows),
            "n_label_events": n_events,
            "status_counts": by_status,
            "class_counts": dict(sorted(by_class.items())),
            "reject_classes": state["reject_classes"],
            "validation_metrics": {k: last.get(k) for k in ("accuracy", "macro_f1", "knn_accuracy", "knn_macro_f1", "n_validation")},
            "stopping": state["stopping"],
            "frame_maps": frames["info"],
            "bees_per_session": frames["sessions"],
            "session_dirs_missing": frames["session_dirs_missing"],
            "frames_in_several_sessions": frames["frames_in_several_sessions"],
        }
        with open(tmp / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        for filename, data in copies.items():
            (tmp / filename).write_bytes(data)
        if out.exists():   # creato da un'altra richiesta con lo stesso nome mentre si scriveva
            raise FileExistsError(f"snapshot '{name}' already exists: snapshots are immutable, pick another name")
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def list_snapshots(cfg: ALConfig) -> list[dict[str, Any]]:
    """Snapshot esistenti, dal più recente: nome, data, round, modello, numero di api, audit set presente."""
    base = snapshots_dir(cfg)
    if not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        meta_path = d / "meta.json"
        if d.name.startswith(".") or not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        out.append({
            "name": d.name, "created": meta.get("created", ""), "round": meta.get("round"),
            "model_version": meta.get("model_version", ""), "n_bees": meta.get("n_bees"),
            "annotator": meta.get("annotator", ""), "status_counts": meta.get("status_counts", {}),
            "has_audit": (d / "audit_set.json").exists(),
        })
    return sorted(out, key=lambda s: (s["created"], s["name"]), reverse=True)


def read_csv(path: Union[str, Path]) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_meta(snapshot_dir: Union[str, Path]) -> dict[str, Any]:
    with open(Path(snapshot_dir) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "PREDICTION_FIELDS", "FRAME_FIELDS", "STATUSES", "UNKNOWN_LABEL", "bee_rows", "export_snapshot", "frame_rows",
    "list_snapshots", "read_csv", "read_meta", "snapshot_path", "snapshots_dir",
]
