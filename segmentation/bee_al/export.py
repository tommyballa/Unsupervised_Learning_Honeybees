"""Chiusura di una fase: salva configurazione e cluster su disco, come dataset a cartelle."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from .config import ALConfig
from .snapshot import NAME_RE, read_csv, snapshot_path

if TYPE_CHECKING:
    from .engine import ALEngine

CLUSTER_FIELDS = ["image_id", "file", "cluster", "status", "source", "confidence", "frame", "session", "colony", "original_path"]
IN_CLUSTERS = ("labeled", "predicted", "rejected")   # api che hanno una classe finale


def cluster_folder(label: str) -> str:
    """Nome di cartella per una classe: spazi in underscore, niente caratteri strani."""
    name = re.sub(r"[^\w.-]+", "_", label.strip(), flags=re.UNICODE).strip("_.")
    return name or "senza_nome"


def list_exports(cfg: ALConfig) -> list[dict[str, Any]]:
    """Esportazioni già fatte in ``export.clusters_dir``, dalla più recente."""
    base = cfg.clusters_dir
    if base is None or not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        meta = d / "meta.json"
        if d.name.startswith(".") or not meta.exists():
            continue
        with open(meta, "r", encoding="utf-8") as f:
            out.append({**json.load(f), "path": str(d)})
    return sorted(out, key=lambda m: (m.get("created", ""), m.get("name", "")), reverse=True)


class ClusterExporter:
    """Esporta i cluster in un thread, senza bloccare l'hub.

    Prima congela uno snapshot (stesso nome dell'esportazione), poi scrive in
    ``<export.clusters_dir>/<nome>/`` una cartella per classe con le immagini, più
    ``clusters.csv`` (tutte le api, anche quelle senza cluster), ``config.yaml`` e ``meta.json``.
    Si lavora in una cartella temporanea rinominata solo alla fine.
    """

    def __init__(self, engine: "ALEngine") -> None:
        self.engine = engine
        self.state = "idle"   # idle | running | failed
        self.error = ""
        self.progress: dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None

    def start(self, name: str, annotator: str) -> None:
        """Avvia l'esportazione. ``ValueError`` se il nome non va bene o un'altra è in corso."""
        cfg = self.engine.cfg
        if self.state == "running":
            raise ValueError("an export is already running")
        if cfg.clusters_dir is None:
            raise ValueError("export.clusters_dir is not set in the config")
        if not NAME_RE.match(name or ""):
            raise ValueError("name: use only letters, digits, '_', '-' and '.', and do not start with a dot")
        if (cfg.clusters_dir / name).exists():
            raise ValueError(f"'{name}' was already exported: pick another name")
        if not self.engine.bank.n_labeled:
            raise ValueError("there are no labels yet: nothing to export")
        if snapshot_path(cfg, name).exists():
            raise ValueError(f"snapshot '{name}' already exists: snapshots are immutable, pick another name")
        self.state, self.error = "running", ""
        self.progress = {"name": name, "copied": 0, "total": 0, "started": time.time()}
        self._thread = threading.Thread(target=self._run, args=(name, annotator), daemon=True, name="bee-al-export")
        self._thread.start()

    def _run(self, name: str, annotator: str, snapshot_dir: Optional[Path] = None) -> None:
        """Congela lo snapshot (se non è già dato) e ne scrive i cluster; anche lo snapshot
        avviene qui, perché su un disco lento scriverlo richiede minuti."""
        cfg = self.engine.cfg
        out = cfg.clusters_dir / name
        tmp = cfg.clusters_dir / f".tmp-{name}-{os.getpid()}"
        try:
            if snapshot_dir is None:
                snapshot_dir = self.engine.export_snapshot(name, annotator)
            rows = read_csv(snapshot_dir / "predictions.csv")
            members = [r for r in rows if r["status"] in IN_CLUSTERS and r["final_label"]]
            self.progress["total"] = len(members)
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            folders = {label: cluster_folder(label) for label in sorted({r["final_label"] for r in members})}
            if len(set(folders.values())) != len(folders):
                raise ValueError(f"two classes map to the same folder name: {folders}")
            for folder in folders.values():
                (tmp / folder).mkdir()

            mode = cfg.export.mode
            table, missing = [], 0
            counts: dict[str, dict[str, int]] = {label: {"human": 0, "predicted": 0} for label in folders}
            for r in rows:
                file = ""
                if r["status"] in IN_CLUSTERS and r["final_label"]:
                    src, dst = Path(r["path"]), tmp / folders[r["final_label"]] / r["image_id"]
                    if src.is_file():
                        if mode == "symlink":
                            os.symlink(src, dst)
                        elif mode == "hardlink":
                            os.link(src, dst)
                        else:
                            shutil.copy2(src, dst)
                        file = f"{folders[r['final_label']]}/{r['image_id']}"
                        counts[r["final_label"]][r["source"]] += 1
                    else:
                        missing += 1
                    self.progress["copied"] += 1
                table.append({"image_id": r["image_id"], "file": file, "cluster": r["final_label"], "status": r["status"],
                              "source": r["source"], "confidence": r["confidence"], "frame": r["frame"],
                              "session": r["session"], "colony": r["colony"], "original_path": r["path"]})
            with open(tmp / "clusters.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CLUSTER_FIELDS)
                writer.writeheader()
                writer.writerows(table)
            with open(tmp / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
            status_counts: dict[str, int] = {}
            for r in rows:
                status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            meta = {
                "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "annotator": annotator,
                "stage": cfg.stage.name, "stage_title": cfg.stage.title, "config_id": cfg.config_id,
                "config_file": self.engine.config_path.name if self.engine.config_path else "",
                "snapshot": str(snapshot_dir), "mode": mode, "n_bees": len(rows), "n_in_clusters": len(members) - missing,
                "missing_files": missing, "status_counts": status_counts,
                "clusters": {label: {"folder": folders[label], **counts[label], "total": sum(counts[label].values())} for label in folders},
            }
            with open(tmp / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            os.rename(tmp, out)
            self.progress["finished"] = time.time()
            self.state = "idle"
        except Exception as e:  # l'esportazione non deve mai far cadere l'hub
            shutil.rmtree(tmp, ignore_errors=True)
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        cfg = self.engine.cfg
        return {
            "state": self.state, "error": self.error, "progress": dict(self.progress),
            "clusters_dir": str(cfg.clusters_dir) if cfg.clusters_dir else "",
            "mode": cfg.export.mode, "exports": list_exports(cfg),
        }
