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


# --------------------------------------------------------------------------- #
# Stime corrette dall'audit
# --------------------------------------------------------------------------- #

GROUP_LEVELS = ("dataset", "session", "colony", "frame", "colony_session")


def group_key(row: dict[str, str], level: str) -> str:
    """Gruppo di un'ape (o di un frame) a un certo livello; vuoto se l'informazione manca."""
    if level == "frame":
        return f"{row['dataset']}/{row['frame']}" if row.get("frame") else ""
    if level == "colony_session":
        return f"{row['colony']}/{row['session']}" if row.get("colony") else ""
    return row.get(level, "")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """Intervallo di Wilson per una proporzione ``k / n``; ``(None, None)`` se ``n == 0``."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Accordo tra due annotatori sulle api doppie.

    ``kappa = (p_o - p_e) / (1 - p_e)``, con ``p_o`` quota di coppie concordi e ``p_e`` accordo
    atteso per caso dai margini dei due annotatori. Per classe si usa lo stesso indice sul
    problema binario "questa classe contro tutte le altre". ``None`` quando non è definito
    (nessuna coppia, o entrambi hanno usato una sola categoria).
    """
    def kappa(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if len(a) == 0:
            return None
        categories = sorted(set(a) | set(b))
        p_o = float(np.mean(a == b))
        p_e = float(sum(np.mean(a == c) * np.mean(b == c) for c in categories))
        return None if p_e >= 1 - 1e-12 else float((p_o - p_e) / (1 - p_e))

    first = np.array([p[0] for p in pairs], dtype=object)
    second = np.array([p[1] for p in pairs], dtype=object)
    classes = sorted(set(first) | set(second))
    matrix = {c: {d: int(np.sum((first == c) & (second == d))) for d in classes} for c in classes}
    return {
        "n_pairs": len(pairs),
        "agreement": float(np.mean(first == second)) if len(pairs) else None,
        "kappa": kappa(first, second),
        "per_class": {c: kappa(np.where(first == c, c, "~"), np.where(second == c, c, "~")) for c in classes},
        "matrix": matrix,
    }


class EstimationInput:
    """Tutto ciò che serve alle stime, già ridotto a conteggi.

    - ``classes``: tutte le classi (predette, date a mano o viste nell'audit); ``reject``: quelle di scarto.
    - ``strata``: descrizione degli strati (id, kind, label, N_h, n_h campionate).
    - ``counts[h, k]``: api dello strato ``h`` che l'audit ha etichettato come classe ``k``;
      ``unknown[h]``: risposte "Non so", contate a parte.
    - ``human[k]``: api etichettate a mano come ``k`` fuori dall'audit (conteggio esatto).
    - ``cc[k]``: classify and count, cioè api con classe finale ``k`` nello snapshot.
    - ``groups[level]``: per ogni gruppo, ``N[g, h]`` e ``human[g, k]``, più gli stessi conteggi
      per frame (``frame_N``, ``frame_human``, ``frame_group``) per ricampionare i frame.
    - ``pairs``: risposte dei due annotatori sulle api doppie.
    """

    def __init__(self, classes: list[str], reject: set[str], strata: list[dict[str, Any]], counts: np.ndarray,
                 human: np.ndarray, unknown: Optional[np.ndarray] = None, cc: Optional[np.ndarray] = None,
                 groups: Optional[dict[str, dict[str, Any]]] = None, pairs: Optional[list[tuple[str, str]]] = None,
                 frames_ok: bool = True, extra: Optional[dict[str, Any]] = None) -> None:
        self.classes = list(classes)
        self.reject = set(reject)
        self.strata = strata
        self.counts = np.asarray(counts, dtype=float)
        self.human = np.asarray(human, dtype=float)
        self.unknown = np.zeros(len(strata)) if unknown is None else np.asarray(unknown, dtype=float)
        self.cc = None if cc is None else np.asarray(cc, dtype=float)
        self.groups = groups or {}
        self.pairs = pairs or []
        self.frames_ok = frames_ok
        self.extra = extra or {}


def _interval(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def estimate_from_input(inp: EstimationInput, n_boot: int = 1000, seed: int = 11) -> dict[str, Any]:
    """Stime stratificate di prevalenza, globali e per gruppo, con intervalli.

    Notazione: strato ``h`` con ``N_h`` api nella popolazione e ``n_h`` risposte di audit
    (escluse le "Non so"); ``a_hk`` risposte di classe ``k``; ``r_hk = a_hk / n_h``;
    ``H_k`` api etichettate a mano come ``k`` (conteggio esatto, mai ricorretto).

    - Precisione di uno strato: ``r_h,c`` con ``c`` la classe predetta dello strato (per gli
      strati di scarto: quota di risposte che sono una qualsiasi classe di scarto), con
      intervallo di Wilson al 95%.
    - Totale stimato: ``Ñ_k = Σ_h N_h · r_hk + H_k``.
    - Api valide: ``V = Σ_{k non di scarto} Ñ_k``. Prevalenza: ``p_k = Ñ_k / V`` per le classi non di scarto.
    - Varianza: ``Var(Ñ_k) = Σ_h N_h² · (1 − n_h/N_h) · r_hk (1 − r_hk) / (n_h − 1)``, zero quando
      ``n_h = N_h`` (strato censito) o ``n_h ≤ 1``. Per ``p_k``, che è un rapporto, si linearizza:
      ``Var(p_k) ≈ (1/V²) Σ_h N_h² (1 − n_h/N_h) s²_h / (n_h − 1)`` con ``s²_h`` la varianza nello
      strato di ``z = 1[k] − p_k · 1[non scarto]``.
    - Classify and count: quota delle api con classe finale ``k`` sul totale non di scarto,
      senza alcuna correzione.
    - Regola del tre: se la classe ``k`` non compare mai tra le ``n_h`` risposte di uno strato,
      il limite superiore al 95% di ``r_hk`` è circa ``3 / n_h``.
    - Per gruppo ``g`` (frame, sessione, alveare): ``Ñ_g,k = Σ_h N_g,h · r_hk + H_g,k``.
      **Ipotesi**: gli errori del modello dentro uno strato sono stimati sull'audit complessivo
      e assunti uguali in tutti i gruppi. Se un gruppo ha api in uno strato senza risposte di
      audit, per quello strato si usa la classe predetta così com'è e lo si segnala.
    - Bootstrap: a ogni replica si ricampionano con reinserimento le risposte di audit dentro
      ogni strato e, per i gruppi, anche i frame del gruppo. Intervallo = percentili 2.5–97.5.
      Due accorgimenti: la variazione di ogni strato è scalata per la popolazione finita
      (``√(1 − n_h/N_h)``, quindi uno strato censito non varia), e si ricampiona da frequenze
      lisciate ``(a_hk + 1/K) / (n_h + 1)``. Senza lisciatura una classe con zero risposte in
      uno strato grande avrebbe varianza nulla e l'intervallo delle classi rare coprirebbe
      il valore vero molto meno del 95% delle volte (circa 77% nelle simulazioni dei test).
    """
    classes, K, H = inp.classes, len(inp.classes), len(inp.strata)
    col = {c: j for j, c in enumerate(classes)}
    valid = np.array([c not in inp.reject for c in classes], dtype=bool)
    N = np.array([s["N_h"] for s in inp.strata], dtype=float)
    a = inp.counts.reshape(H, K)
    n = a.sum(axis=1)
    r = np.divide(a, n[:, None], out=np.zeros_like(a), where=n[:, None] > 0)
    missing = n == 0                                # strati senza risposte: si tiene la classe predetta
    for h in np.flatnonzero(missing):
        label = inp.strata[h].get("label", "")
        if label in col:
            r[h, col[label]] = 1.0
    fpc = np.clip(1.0 - np.divide(n, N, out=np.ones_like(n), where=N > 0), 0.0, 1.0)
    fpc[missing] = 0.0

    def shares(totals: np.ndarray) -> np.ndarray:
        """Quote sulle sole classi non di scarto (ultimo asse = classi)."""
        v = totals[..., valid].sum(axis=-1, keepdims=True)
        return np.divide(totals, v, out=np.zeros_like(totals), where=v > 0)

    totals = N @ r + inp.human
    V = float(totals[valid].sum())
    p = shares(totals)
    var_scale = np.divide(N**2 * fpc, n - 1, out=np.zeros_like(N), where=n > 1)
    var_total = (var_scale[:, None] * r * (1 - r)).sum(axis=0)
    r_valid = r[:, valid].sum(axis=1)
    se = np.zeros(K)
    for k in np.flatnonzero(valid):
        m1 = r[:, k] * (1 - p[k]) - p[k] * (r_valid - r[:, k])
        m2 = r[:, k] * (1 - p[k]) ** 2 + p[k] ** 2 * (r_valid - r[:, k])
        s2 = np.clip(m2 - m1**2, 0.0, None) * np.divide(n, n - 1, out=np.zeros_like(n), where=n > 1)
        se[k] = np.sqrt((var_scale * s2).sum()) / V if V > 0 else 0.0

    # ------------------------------ bootstrap ------------------------------ #
    rng = np.random.default_rng(seed)
    B = max(1, int(n_boot))
    r_boot = np.broadcast_to(r, (B, H, K)).copy()
    for h in range(H):
        if n[h] > 0 and fpc[h] > 0:
            # Frequenze lisciate con una pseudo-osservazione per strato, divisa tra le classi:
            # una classe mai vista tra n_h risposte non è per questo impossibile nello strato.
            smooth = (a[h] + 1.0 / K) / (n[h] + 1.0)
            draws = rng.multinomial(int(n[h]), smooth, size=B) / n[h]
            r_boot[:, h, :] = np.clip(r[h] + np.sqrt(fpc[h]) * (draws - smooth), 0.0, None)
    totals_boot = np.einsum("h,bhk->bk", N, r_boot) + inp.human
    p_boot = shares(totals_boot)
    all_boot = totals_boot.sum(axis=1)
    reject_share_boot = np.divide(totals_boot[:, ~valid].sum(axis=1), all_boot, out=np.zeros(B), where=all_boot > 0)

    observed = a.sum(axis=0)
    unseen_rate = np.where(n > 0, np.minimum(1.0, 3.0 / np.maximum(n, 1)), 0.0)   # regola del tre, per strato
    upper_rate = np.where(a > 0, r, unseen_rate[:, None])

    strata_out = []
    for h, s in enumerate(inp.strata):
        kind, label = s.get("kind", "class"), s.get("label", "")
        if kind == "reject":
            hits = float(a[h, ~valid].sum())
        elif label in col:
            hits = float(a[h, col[label]])
        else:
            hits = None
        lo, hi = wilson(hits, n[h]) if hits is not None else (None, None)
        strata_out.append({
            "id": s["id"], "kind": kind, "label": label, "conf_bin": s.get("conf_bin", ""),
            "anomaly_bin": s.get("anomaly_bin", ""), "N_h": int(N[h]), "n_sampled": int(s.get("n_h", n[h] + inp.unknown[h])),
            "n_labeled": int(n[h]), "unknown": int(inp.unknown[h]),
            "precision": None if hits is None or n[h] == 0 else float(hits / n[h]), "ci_low": lo, "ci_high": hi,
            "upper_unseen": None if n[h] == 0 else float(unseen_rate[h]),
            "answers": {classes[k]: int(a[h, k]) for k in range(K) if a[h, k] > 0},
        })

    cc = inp.cc if inp.cc is not None else np.zeros(K)
    cc_valid = float(cc[valid].sum())
    prevalence = []
    for k, c in enumerate(classes):
        is_valid = bool(valid[k])
        lo, hi = _interval(p_boot[:, k]) if is_valid else (None, None)
        upper = None
        if observed[k] == 0:
            upper = float((N @ upper_rate[:, k] + inp.human[k]) / V) if V > 0 else None
        prevalence.append({
            "class": c, "reject": not is_valid,
            "cc_count": int(cc[k]), "cc_share": float(cc[k] / cc_valid) if is_valid and cc_valid > 0 else None,
            "n_hat": float(totals[k]), "n_hat_var": float(var_total[k]),
            "share": float(p[k]) if is_valid else None,
            "se": float(se[k]) if is_valid else None,
            "ci_analytic_low": float(max(0.0, p[k] - 1.96 * se[k])) if is_valid else None,
            "ci_analytic_high": float(min(1.0, p[k] + 1.96 * se[k])) if is_valid else None,
            "ci_boot_low": lo, "ci_boot_high": hi,
            "human": int(inp.human[k]), "audit_observed": int(observed[k]), "upper_limit": upper,
        })

    total_all = float(totals.sum())
    result: dict[str, Any] = {
        "classes": classes,
        "reject_classes": sorted(inp.reject),
        "n_boot": B,
        "seed": seed,
        "strata": strata_out,
        "prevalence": prevalence,
        "totals": {
            "population": int(N.sum()), "human": int(inp.human.sum()),
            "valid_hat": V, "reject_hat": float(totals[~valid].sum()),
            "reject_share": float(totals[~valid].sum() / total_all) if total_all > 0 else None,
            "reject_share_ci": list(_interval(reject_share_boot)),
            "audit_labeled": int(n.sum()), "audit_unknown": int(inp.unknown.sum()),
        },
        "strata_without_audit": [inp.strata[h]["id"] for h in np.flatnonzero(missing)],
        "complete": bool(not missing.any() and all(s["n_labeled"] + s["unknown"] >= s["n_sampled"] for s in strata_out)),
        "frame_resampling": bool(inp.frames_ok),
        "agreement": cohen_kappa(inp.pairs),
        "groups": {},
    }
    result.update(inp.extra)
    boot: dict[str, Any] = {"classes": classes, "valid": valid, "global": totals_boot, "groups": {}, "names": {}}

    for level, g in inp.groups.items():
        names, Ng, Hg = g["names"], np.asarray(g["N"], dtype=float), np.asarray(g["human"], dtype=float)
        frame_group = np.asarray(g["frame_group"], dtype=int)
        frame_N, frame_human = np.asarray(g["frame_N"], dtype=float), np.asarray(g["frame_human"], dtype=float)
        confirmed = np.asarray(g.get("confirmed", np.zeros_like(Hg)), dtype=float) + Hg
        totals_g = Ng @ r + Hg
        p_g = shares(totals_g)
        boot_g = np.empty((B, len(names), K))
        resampled = bool(inp.frames_ok and level != "frame")
        for gi in range(len(names)):
            frames_i = np.flatnonzero(frame_group == gi)
            if resampled and len(frames_i) > 1:
                w = rng.multinomial(len(frames_i), np.full(len(frames_i), 1.0 / len(frames_i)), size=B).astype(float)
                n_star, h_star = w @ frame_N[frames_i], w @ frame_human[frames_i]
            else:
                n_star, h_star = np.broadcast_to(Ng[gi], (B, H)), np.broadcast_to(Hg[gi], (B, K))
            boot_g[:, gi, :] = np.einsum("bh,bhk->bk", n_star, r_boot) + h_star
        pb = shares(boot_g)
        all_g = boot_g.sum(axis=2)
        rej_b = np.divide(boot_g[:, :, ~valid].sum(axis=2), all_g, out=np.zeros_like(all_g), where=all_g > 0)

        rows = []
        for gi, name in enumerate(names):
            v_g = float(totals_g[gi, valid].sum())
            tot_g = float(totals_g[gi].sum())
            upper_g = (Ng[gi] @ upper_rate + Hg[gi]) / v_g if v_g > 0 else np.zeros(K)
            per_class = {}
            for k, c in enumerate(classes):
                if not valid[k]:
                    continue
                lo, hi = _interval(pb[:, gi, k])
                per_class[c] = {"share": float(p_g[gi, k]), "ci_low": lo, "ci_high": hi, "n_hat": float(totals_g[gi, k]),
                                "confirmed": int(confirmed[gi, k]), "upper_limit": float(min(1.0, upper_g[k]))}
            rows.append({
                "group": name,
                "population": int(Ng[gi].sum()), "human": int(Hg[gi].sum()), "valid_hat": v_g,
                "reject_share": float(totals_g[gi, ~valid].sum() / tot_g) if tot_g > 0 else None,
                "reject_share_ci": list(_interval(rej_b[:, gi])),
                "n_frames": int(len(np.flatnonzero(frame_group == gi))),
                "missing_strata": [inp.strata[h]["id"] for h in np.flatnonzero(missing & (Ng[gi] > 0))],
                "classes": per_class,
                **{key: values[gi] for key, values in g.get("info", {}).items()},
            })
        result["groups"][level] = {"frame_resampling": resampled, "groups": rows}
        boot["groups"][level] = boot_g
        boot["names"][level] = list(names)
    result["_boot"] = boot
    return result


def load_estimation_input(snapshot_dir: Union[str, Path], cfg: ALConfig, store: Optional[LabelStore] = None,
                          anomaly_scores: Optional[dict[str, float]] = None) -> EstimationInput:
    """Legge snapshot, audit set e risposte di audit e li riduce ai conteggi usati dalle stime."""
    snap = Path(snapshot_dir)
    audit_set = load_audit_set(snap)
    rows = read_csv(snap / "predictions.csv")
    stratum_of = {r["image_id"]: r["stratum"] for r in read_csv(snap / STRATA_FILE)}
    store = store or LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
    normalize = class_normalizer(cfg)
    reject = {normalize(c) for c in cfg.labels.reject_classes}
    reject |= {r["final_label"] for r in rows if r["status"] == "rejected"}

    from . import UNKNOWN_LABEL

    batch = batch_id_of(audit_set["snapshot"])
    answers = store.per_annotator(lambda e: e.query_type == "audit" and e.batch_id == batch)
    first: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    pairs_unknown = 0
    for bee in audit_set["bees"]:
        given = [ev.label for ev in answers.get(bee["image_id"], {}).values()]
        if given:
            first[bee["image_id"]] = given[0]
        if bee["double"] and len(given) >= 2:
            if UNKNOWN_LABEL in given[:2]:
                pairs_unknown += 1
            else:
                pairs.append((given[0], given[1]))

    classes = sorted({r["final_label"] for r in rows if r["final_label"]}
                     | {lbl for lbl in first.values() if lbl != UNKNOWN_LABEL} | {lbl for pair in pairs for lbl in pair})
    col = {c: j for j, c in enumerate(classes)}
    strata = audit_set["strata"]
    row_of = {s["id"]: h for h, s in enumerate(strata)}
    H, K = len(strata), len(classes)

    counts, unknown = np.zeros((H, K)), np.zeros(H)
    for image_id, label in first.items():
        h = row_of[stratum_of[image_id]]
        if label == UNKNOWN_LABEL:
            unknown[h] += 1
        else:
            counts[h, col[label]] += 1

    human, cc = np.zeros(K), np.zeros(K)
    levels = [lv for lv in GROUP_LEVELS if any(group_key(r, lv) for r in rows)]
    names = {lv: sorted({group_key(r, lv) for r in rows} - {""}) for lv in levels}
    index = {lv: {name: i for i, name in enumerate(names[lv])} for lv in levels}
    frames = sorted({(r["dataset"], r["frame"]) for r in rows})
    frame_index = {f: i for i, f in enumerate(frames)}
    frame_N, frame_human = np.zeros((len(frames), H)), np.zeros((len(frames), K))
    grp = {lv: {"N": np.zeros((len(names[lv]), H)), "human": np.zeros((len(names[lv]), K)),
                "confirmed": np.zeros((len(names[lv]), K)),
                "frame_group": np.full(len(frames), -1),
                "info": {key: [0] * len(names[lv]) for key in
                         ("n_bees", "unassigned", "unknown", "validation", "anomaly_scored", "anomaly_above")}}
           for lv in levels}
    threshold = cfg.colony.anomaly_threshold
    frames_ok = True
    for r in rows:
        fi = frame_index[(r["dataset"], r["frame"])]
        h = row_of[stratum_of[r["image_id"]]] if r["image_id"] in stratum_of else None
        k = col.get(r["final_label"])
        is_human = r["source"] == "human"
        if r["final_label"] and r["status"] != "unassigned":
            cc[k] += 1
        if is_human:
            human[k] += 1
            frame_human[fi, k] += 1
        elif h is not None:
            frame_N[fi, h] += 1
        if (is_human or h is not None) and not r["frame"]:
            frames_ok = False
        audit_label = first.get(r["image_id"])
        score = anomaly_scores.get(r["image_id"]) if anomaly_scores else None
        for lv in levels:
            key = group_key(r, lv)
            if not key:
                continue
            gi = index[lv][key]
            g = grp[lv]
            g["frame_group"][fi] = gi
            g["info"]["n_bees"][gi] += 1
            if r["status"] in ("unassigned", "unknown", "validation"):
                g["info"][r["status"]][gi] += 1
            if is_human:
                g["human"][gi, k] += 1
            elif h is not None:
                g["N"][gi, h] += 1
            if audit_label is not None and audit_label != UNKNOWN_LABEL:
                g["confirmed"][gi, col[audit_label]] += 1
            if score is not None and threshold > 0:
                g["info"]["anomaly_scored"][gi] += 1
                g["info"]["anomaly_above"][gi] += int(score >= threshold)

    # Forza della colonia e scarti della pulizia, per gruppo, dal file dei frame.
    frames_csv = read_csv(snap / "frames.csv") if (snap / "frames.csv").exists() else []
    for lv in levels:
        g = grp[lv]
        for key in ("frames_listed", "raw_bees", "cleaned_accepted", "cleaned_discarded"):
            g["info"][key] = [0] * len(names[lv])
        for f in frames_csv:
            key = group_key(f, lv)
            if key not in index[lv]:
                continue
            gi = index[lv][key]
            g["info"]["frames_listed"][gi] += 1
            for column in ("raw_bees", "cleaned_accepted", "cleaned_discarded"):
                g["info"][column][gi] += int(f[column] or 0)
        g["names"], g["frame_N"], g["frame_human"] = names[lv], frame_N, frame_human

    extra = {
        "snapshot": audit_set["snapshot"],
        "audit": {"sampled": len(audit_set["bees"]), "answered": len(first),
                  "double": sum(b["double"] for b in audit_set["bees"]), "pairs_with_unknown": pairs_unknown},
    }
    return EstimationInput(classes, reject, strata, counts, human, unknown, cc, grp, pairs, frames_ok, extra)


def estimate(snapshot_dir: Union[str, Path], cfg: ALConfig, n_boot: int = 1000, store: Optional[LabelStore] = None,
             anomaly_scores: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """Stime corrette per uno snapshot con audit: vedi ``estimate_from_input`` per le formule.

    Le risposte di audit si leggono dallo store delle etichette (``batch_id = audit:<snapshot>``);
    per le api doppie vale la risposta del primo annotatore, le coppie servono solo per il kappa.
    """
    return estimate_from_input(load_estimation_input(snapshot_dir, cfg, store, anomaly_scores), n_boot, cfg.colony.seed)
