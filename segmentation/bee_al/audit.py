"""Audit: campione di controllo stratificato per cluster predetto, etichettato alla cieca.

Lo snapshot dice cosa il modello ha assegnato a ogni ape; l'audit misura quanto spesso ha
ragione, strato per strato, su api che nessuno aveva mai etichettato. Le etichette di audit
non entrano mai nella memory bank: servono solo a correggere le stime (vedi ``estimate``).
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .config import ALConfig, AuditConfig
from .label_store import LabelStore
from .snapshot import read_csv

AUDIT_FILE = "audit_set.json"
STRATA_FILE = "audit_strata.csv"
SCORES_FILE = "anomaly_scores.csv"
UNASSIGNED_STRATUM = "__unassigned__"
POPULATION_STATUSES = ("predicted", "unassigned", "rejected")


def class_normalizer(cfg: ALConfig) -> Callable[[str], str]:
    """Stessa normalizzazione e stessi alias delle etichette, per i nomi di classe scritti in config."""
    store = LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
    return lambda name: store.resolve(store.normalize_label(name))


def confidence_bin(confidence: float, bins: list[float]) -> str:
    """Fascia di confidenza: con soglie ``[0.9]`` restituisce ``<0.9`` oppure ``>=0.9``."""
    edges = sorted(bins)
    if not edges:
        return ""
    for i, edge in enumerate(edges):
        if confidence < edge:
            return f"<{edge:g}" if i == 0 else f"{edges[i - 1]:g}-{edge:g}"
    return f">={edges[-1]:g}"


def parse_scores(text: str) -> dict[str, float]:
    """Legge un CSV ``image_id,score`` (con intestazione)."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"image_id", "score"} <= set(reader.fieldnames):
        raise ValueError("anomaly scores: expected a CSV with the columns image_id,score")
    scores: dict[str, float] = {}
    for row in reader:
        try:
            scores[row["image_id"]] = float(row["score"])
        except (TypeError, ValueError):
            raise ValueError(f"anomaly scores: '{row['score']}' is not a number (image {row['image_id']})") from None
    return scores


def read_scores(path: Union[str, Path]) -> dict[str, float]:
    return parse_scores(Path(path).read_text(encoding="utf-8"))


def in_population(row: dict[str, str]) -> bool:
    """Popolazione dell'audit: api assegnate dal modello (o non assegnate), mai etichettate a mano."""
    return row["status"] in POPULATION_STATUSES and row["source"] != "human"


def assign_strata(
    rows: list[dict[str, str]], audit: AuditConfig, split_class: str = "", anomaly_scores: Optional[dict[str, float]] = None
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Strato di ogni ape della popolazione: ``({image_id: id strato}, {id strato: descrizione})``.

    - classe predetta non di scarto: classe × fascia di confidenza;
    - classe di scarto predetta: uno strato per classe, senza fasce;
    - non assegnate: un solo strato ``__unassigned__``;
    - se ci sono gli anomaly score, gli strati di ``split_class`` si dividono ancora in
      score sopra/sotto il quantile ``anomaly_quantile`` (calcolato sulle api di quella classe;
      un'ape senza score sta sotto).
    """
    population = [r for r in rows if in_population(r)]
    threshold: Optional[float] = None
    if anomaly_scores and split_class:
        values = [anomaly_scores[r["image_id"]] for r in population
                  if r["status"] == "predicted" and r["final_label"] == split_class and r["image_id"] in anomaly_scores]
        if values:
            threshold = float(np.quantile(values, audit.anomaly_quantile))

    of: dict[str, str] = {}
    strata: dict[str, dict[str, Any]] = {}
    for r in population:
        label, conf_bin, anomaly_bin, kind = r["final_label"], "", "", "class"
        if r["status"] == "unassigned":
            sid, label, kind = UNASSIGNED_STRATUM, "", "unassigned"
        elif r["status"] == "rejected":
            sid, kind = f"{label}|reject", "reject"
        else:
            conf_bin = confidence_bin(float(r["confidence"]), audit.confidence_bins)
            sid = f"{label}|{conf_bin}" if conf_bin else label
            if threshold is not None and label == split_class:
                anomaly_bin = "high" if anomaly_scores.get(r["image_id"], -np.inf) >= threshold else "low"
                sid += f"|anomaly_{anomaly_bin}"
        of[r["image_id"]] = sid
        info = strata.setdefault(sid, {"id": sid, "kind": kind, "label": label, "conf_bin": conf_bin,
                                       "anomaly_bin": anomaly_bin, "N_h": 0})
        info["N_h"] += 1
    if threshold is not None:
        for info in strata.values():
            if info["anomaly_bin"]:
                info["anomaly_threshold"] = threshold
    return of, dict(sorted(strata.items()))


def allocation(n_stratum: int, audit: AuditConfig) -> int:
    """Api da campionare in uno strato di ``n_stratum`` api."""
    if n_stratum <= audit.per_stratum:
        return n_stratum
    return min(n_stratum, audit.large_stratum if n_stratum >= audit.large_threshold else audit.per_stratum)


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def create_audit_set(
    snapshot_dir: Union[str, Path], cfg: ALConfig, anomaly_scores: Optional[dict[str, float]] = None
) -> Path:
    """Estrae il campione di controllo di uno snapshot e lo salva in ``audit_set.json``.

    Si crea una volta sola, come il validation set: se esiste già si solleva ``FileExistsError``.
    Dentro ogni strato l'estrazione è casuale uniforme senza reinserimento, con il seed di
    ``audit.seed``; una frazione ``double_fraction`` delle api estratte (almeno una per strato)
    va etichettata da due annotatori diversi; l'ordine di presentazione è una permutazione
    fissa che mescola gli strati. Accanto si salva ``audit_strata.csv`` con lo strato di ogni
    ape della popolazione, che serve alle stime, e gli anomaly score se sono stati forniti.
    """
    snap = Path(snapshot_dir)
    out = snap / AUDIT_FILE
    if out.exists():
        raise FileExistsError(f"snapshot '{snap.name}' already has an audit set: it is created only once")
    audit = cfg.audit
    split_class = class_normalizer(cfg)(audit.anomaly_split_class) if audit.anomaly_split_class else ""
    rows = read_csv(snap / "predictions.csv")
    stratum_of, strata = assign_strata(rows, audit, split_class, anomaly_scores)
    if not stratum_of:
        raise ValueError(f"snapshot '{snap.name}' has no predicted bees to audit")

    members: dict[str, list[str]] = {sid: [] for sid in strata}
    for image_id, sid in stratum_of.items():
        members[sid].append(image_id)
    rng = np.random.default_rng(audit.seed)
    bees: list[dict[str, Any]] = []
    for sid, info in strata.items():
        pool = sorted(members[sid])
        n_h = allocation(len(pool), audit)
        chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=n_h, replace=False))]
        n_double = min(n_h, max(1, int(round(audit.double_fraction * n_h)))) if audit.double_fraction > 0 else 0
        double = set(rng.choice(n_h, size=n_double, replace=False).tolist())
        info["n_h"] = n_h
        info["n_double"] = n_double
        bees += [{"image_id": image_id, "stratum": sid, "double": j in double} for j, image_id in enumerate(chosen)]
    order = [bees[i]["image_id"] for i in rng.permutation(len(bees))]

    with open(snap / (STRATA_FILE + ".tmp"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "stratum"])
        writer.writerows(sorted(stratum_of.items()))
    os.replace(snap / (STRATA_FILE + ".tmp"), snap / STRATA_FILE)
    if anomaly_scores:
        with open(snap / (SCORES_FILE + ".tmp"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "score"])
            writer.writerows(sorted(anomaly_scores.items()))
        os.replace(snap / (SCORES_FILE + ".tmp"), snap / SCORES_FILE)
    _write_json_atomic(out, {
        "snapshot": snap.name,
        "seed": audit.seed,
        "config": {**vars(audit), "anomaly_split_class": split_class},
        "anomaly_scores": bool(anomaly_scores),
        "population": len(stratum_of),
        "strata": list(strata.values()),
        "bees": bees,
        "order": order,
    })
    return out


def load_audit_set(snapshot_dir: Union[str, Path]) -> dict[str, Any]:
    with open(Path(snapshot_dir) / AUDIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_id_of(snapshot_name: str) -> str:
    """``batch_id`` degli eventi di audit di uno snapshot in ``al_labels.csv``."""
    return f"audit:{snapshot_name}"
