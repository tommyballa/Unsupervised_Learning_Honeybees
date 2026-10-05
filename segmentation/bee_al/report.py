"""Report di uno snapshot con audit: stime, indicatori di colonia e file da scaricare."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Union

from .audit import AUDIT_FILE, SCORES_FILE, batch_id_of, estimate_from_input, load_estimation_input, read_scores
from .colony import colony_indicators
from .config import ALConfig
from .label_store import LabelStore

REPORT_DIR = "report"
REPORT_FILES = ("report.md", "strata.csv", "prevalence.csv", "groups.csv", "colony_indicators.csv",
                "colony_indicators.json", "agreement.json", "report.json")


def _scores_path(snap: Path) -> Optional[Path]:
    for path in (snap / REPORT_DIR / SCORES_FILE, snap / SCORES_FILE):
        if path.exists():
            return path
    return None


def fingerprint(snapshot_dir: Union[str, Path], cfg: ALConfig, store: LabelStore, n_boot: int) -> str:
    """Cambia quando cambia qualcosa che entra nel report: risposte di audit, config, anomaly score."""
    snap = Path(snapshot_dir)
    batch = batch_id_of(snap.name)
    h = hashlib.sha1()
    for e in store.events():
        if e.query_type == "audit" and e.batch_id == batch:
            h.update(f"{e.image_id}|{e.annotator}|{e.label}|{e.timestamp}\n".encode("utf-8"))
    config = cfg.to_dict()
    h.update(json.dumps([config["colony"], config["labels"], n_boot], sort_keys=True).encode("utf-8"))
    scores = _scores_path(snap)
    for path in (snap / AUDIT_FILE, scores):
        if path is not None and path.exists():
            h.update(f"{path.name}|{path.stat().st_size}|{path.stat().st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def indicator_rows(indicators: dict[str, Any]) -> list[dict[str, Any]]:
    """Indicatori in forma tabellare: una riga per gruppo × indicatore × voce."""
    rows: list[dict[str, Any]] = []
    for g in indicators["groups"]:
        base = {"group": g["group"]}
        for name, e in g["I1_demographics"].items():
            rows.append({**base, "indicator": "I1_demographics", "item": name, "value": e.get("share"),
                         "ci_low": e.get("ci_low"), "ci_high": e.get("ci_high"), "status": e.get("note", "")})
        for name, e in g["I2_disease"].items():
            rows.append({**base, "indicator": "I2_disease", "item": name, "value": e.get("share"), "ci_low": e.get("ci_low"),
                         "ci_high": e.get("ci_high"), "status": e.get("status", e.get("note", "")),
                         "detail": f"confirmed {e.get('confirmed', 0)}" + (f", upper limit {e['upper_limit']:.4f}" if "upper_limit" in e else "")})
        j = g["I3_jsd"]
        rows.append({**base, "indicator": "I3_jsd", "item": "", "value": j["value"], "ci_low": j["ci_low"], "ci_high": j["ci_high"],
                     "status": "above_threshold" if j["above_threshold"] else "", "detail": f"threshold {j['threshold']:.4f}"})
        u = g["I4_unexplained"]
        rows.append({**base, "indicator": "I4_unexplained", "item": "", "value": u["share"],
                     "detail": f"unassigned {u['unassigned']}, unknown {u['unknown']}, bees {u['n_bees']}"})
        s, q = g["strength"], g["quality"]
        rows.append({**base, "indicator": "strength", "item": "raw_bees_per_frame", "value": s["raw_bees_per_frame"],
                     "detail": f"{s['raw_bees']} raw bees in {s['frames']} frames"})
        rows.append({**base, "indicator": "quality", "item": "reject_share", "value": q["reject_share"],
                     "ci_low": q["reject_share_ci"][0], "ci_high": q["reject_share_ci"][1]})
        rows.append({**base, "indicator": "quality", "item": "cleaning_discard_rate", "value": q["cleaning_discard_rate"],
                     "detail": f"{q['cleaning_discarded']} crops discarded by the cleaning step"})
        if "anomaly" in g:
            a = g["anomaly"]
            rows.append({**base, "indicator": "anomaly", "item": f"score>={a['threshold']:g}", "value": a["share"],
                         "ci_low": a["ci_low"], "ci_high": a["ci_high"], "detail": f"{a['n_scored']} bees with a score"})
        rows.append({**base, "indicator": "light", "item": "", "status": g["light"]["color"],
                     "detail": "; ".join(f"{r['rule']} ({r['detail']})" for r in g["light"]["rules"])})
    for t in indicators["trend"]:
        base = {"group": f"{t['colony']}: {t['from']} -> {t['to']}"}
        if t.get("missing"):
            rows.append({**base, "indicator": "I5_trend", "status": "missing", "detail": ", ".join(t["missing"])})
        for name, d in t["deltas"].items():
            rows.append({**base, "indicator": "I5_trend", "item": name, "value": d["delta"], "ci_low": d["ci_low"],
                         "ci_high": d["ci_high"], "status": "significant" if d["significant"] else ""})
    return rows


def markdown(estimates: dict[str, Any], indicators: dict[str, Any]) -> str:
    """Riassunto leggibile del report."""
    t, audit = estimates["totals"], estimates.get("audit", {})
    lines = [
        f"# Audit report — snapshot `{estimates.get('snapshot', '')}`", "",
        f"- Audit: {audit.get('answered', 0)} of {audit.get('sampled', 0)} sampled bees answered "
        f"({t['audit_unknown']} 'Non so'), {audit.get('double', 0)} to be labeled twice.",
        f"- Population audited: {t['population']} predicted bees; {t['human']} more were labeled by hand (exact counts).",
        f"- Estimated valid bees: {t['valid_hat']:.0f}; estimated reject share: {_pct(t['reject_share'])} "
        f"(95% CI {_pct(t['reject_share_ci'][0])} – {_pct(t['reject_share_ci'][1])}).",
        f"- Bootstrap: {estimates['n_boot']} replicates" + ("" if estimates["frame_resampling"] else
                                                          "; frames could not be resampled (bee-to-frame map missing or unreliable)") + ".",
    ]
    if not estimates["complete"]:
        lines.append("- **The audit is not complete**: the figures below will still move.")
    if estimates["strata_without_audit"]:
        lines.append(f"- **Strata without any audit answer** (the predicted class is used as is): {', '.join(estimates['strata_without_audit'])}.")
    lines += ["", "## Precision per stratum", "", "| Stratum | Bees | Answered | Precision | 95% CI |", "|---|---|---|---|---|"]
    for s in estimates["strata"]:
        ci = "-" if s["ci_low"] is None else f"{_pct(s['ci_low'])} – {_pct(s['ci_high'])}"
        lines.append(f"| {s['id']} | {s['N_h']} | {s['n_labeled']} | {_pct(s['precision'])} | {ci} |")
    lines += ["", "## Prevalence", "", "Share of valid (non-reject) bees. *Classify and count* is the raw share of predicted labels.", "",
              "| Class | Classify and count | Stratified estimate | 95% CI (bootstrap) | Upper limit if never seen |", "|---|---|---|---|---|"]
    for p in estimates["prevalence"]:
        if p["reject"]:
            continue
        lines.append(f"| {p['class']} | {_pct(p['cc_share'])} | {_pct(p['share'])} | {_pct(p['ci_boot_low'])} – {_pct(p['ci_boot_high'])} | "
                     f"{_pct(p['upper_limit']) if p['upper_limit'] is not None else ''} |")
    rejects = [p for p in estimates["prevalence"] if p["reject"]]
    if rejects:
        lines += ["", "Reject classes (left out of the denominator): " + ", ".join(f"{p['class']} ≈ {p['n_hat']:.0f} bees" for p in rejects) + "."]
    k = estimates["agreement"]
    lines += ["", "## Agreement between annotators", ""]
    if k["n_pairs"]:
        kappa = "not defined" if k["kappa"] is None else f"{k['kappa']:.2f}"
        lines.append(f"{k['n_pairs']} bees labeled twice: raw agreement {_pct(k['agreement'])}, Cohen's kappa {kappa}.")
    else:
        lines.append("No bee has been labeled by two annotators yet.")
    lines += ["", f"## Colony indicators by {indicators['group_by']}", ""]
    for note in indicators["notes"]:
        lines.append(f"- Note: {note}")
    if indicators["groups"]:
        ref = indicators["reference"]
        lines += [f"Reference profile: {'given explicitly' if ref['explicit'] else 'mean of ' + ', '.join(ref['groups'])}; "
                  f"JSD threshold {ref['jsd_threshold']:.3f}.", "",
                  "| Group | Light | Rules fired | JSD (I3) | Unexplained (I4) | Estimated reject share | Discarded by cleaning | Raw bees / frame |",
                  "|---|---|---|---|---|---|---|---|"]
        for g in indicators["groups"]:
            rules = "; ".join(f"{r['rule']}: {r['detail']}" for r in g["light"]["rules"]) or "-"
            per_frame = g["strength"]["raw_bees_per_frame"]
            lines.append(f"| {g['group']} | {g['light']['color']} | {rules} | {g['I3_jsd']['value']:.3f} | {_pct(g['I4_unexplained']['share'])} | "
                         f"{_pct(g['quality']['reject_share'])} | {_pct(g['quality']['cleaning_discard_rate'])} | "
                         f"{'-' if per_frame is None else f'{per_frame:.0f}'} |")
        for title, key in (("Demographic profile (I1)", "I1_demographics"), ("Disease signs (I2)", "I2_disease")):
            if any(g[key] for g in indicators["groups"]):
                lines += ["", f"### {title}", ""]
                for g in indicators["groups"]:
                    for name, e in g[key].items():
                        if e.get("share") is None:
                            lines.append(f"- {g['group']} — {name}: {e.get('note', 'not available')}")
                        else:
                            status = f" — **{e['status']}**" if "status" in e else ""
                            lines.append(f"- {g['group']} — {name}: {_pct(e['share'])} ({_pct(e['ci_low'])} – {_pct(e['ci_high'])}){status}")
    if indicators["trend"]:
        lines += ["", "### Trend between sessions (I5)", ""]
        for step in indicators["trend"]:
            head = f"- {step['colony']}: {step['from']} → {step['to']}"
            if step.get("missing"):
                lines.append(f"{head}: no data for {', '.join(step['missing'])}")
            for name, d in step["deltas"].items():
                mark = " **significant**" if d["significant"] else ""
                lines.append(f"{head} — {name}: {d['delta']:+.3f} ({d['ci_low']:+.3f} – {d['ci_high']:+.3f}){mark}")
    lines += ["", "Group estimates assume that the model's errors inside a stratum are the same in every group.", ""]
    return "\n".join(lines)


def build_report(snapshot_dir: Union[str, Path], cfg: ALConfig, store: Optional[LabelStore] = None,
                 anomaly_scores_path: Union[str, Path, None] = None, n_boot: Optional[int] = None,
                 force: bool = False) -> dict[str, Any]:
    """Calcola stime e indicatori di uno snapshot con audit e scrive i file in ``<snapshot>/report/``.

    Il bootstrap costa qualche secondo: il risultato resta su disco e viene riusato finché non
    arrivano nuove risposte di audit o non cambia la config che lo riguarda.
    """
    snap = Path(snapshot_dir)
    if not (snap / AUDIT_FILE).exists():
        raise FileNotFoundError(f"snapshot '{snap.name}' has no audit set")
    store = store or LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
    out = snap / REPORT_DIR
    out.mkdir(exist_ok=True)
    if anomaly_scores_path is not None:
        read_scores(anomaly_scores_path)  # si controlla il formato prima di copiarlo
        shutil.copy2(anomaly_scores_path, out / SCORES_FILE)
    n_boot = n_boot or cfg.colony.n_boot
    key = fingerprint(snap, cfg, store, n_boot)
    cached = out / "report.json"
    if cached.exists() and not force:
        with open(cached, "r", encoding="utf-8") as f:
            report = json.load(f)
        if report.get("fingerprint") == key:
            return report

    scores_file = _scores_path(snap)
    scores = read_scores(scores_file) if scores_file else None
    estimates = estimate_from_input(load_estimation_input(snap, cfg, store, scores), n_boot, cfg.colony.seed)
    indicators = colony_indicators(estimates, cfg)
    estimates.pop("_boot")
    report = {"fingerprint": key, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshot": snap.name,
              "estimates": estimates, "indicators": indicators}

    _write_csv(out / "strata.csv", ["id", "kind", "label", "conf_bin", "anomaly_bin", "N_h", "n_sampled", "n_labeled", "unknown",
                                    "precision", "ci_low", "ci_high", "upper_unseen"], estimates["strata"])
    _write_csv(out / "prevalence.csv", ["class", "reject", "cc_count", "cc_share", "n_hat", "n_hat_var", "share", "se",
                                        "ci_analytic_low", "ci_analytic_high", "ci_boot_low", "ci_boot_high", "human",
                                        "audit_observed", "upper_limit"], estimates["prevalence"])
    group_rows = [
        {"level": level, "group": g["group"], "class": name, **stats, "population": g["population"],
         "reject_share": g["reject_share"], "missing_strata": ";".join(g["missing_strata"])}
        for level, data in estimates["groups"].items() for g in data["groups"] for name, stats in g["classes"].items()
    ]
    _write_csv(out / "groups.csv", ["level", "group", "class", "share", "ci_low", "ci_high", "n_hat", "confirmed", "upper_limit",
                                    "population", "reject_share", "missing_strata"], group_rows)
    _write_csv(out / "colony_indicators.csv", ["group", "indicator", "item", "value", "ci_low", "ci_high", "status", "detail"],
               indicator_rows(indicators))
    for name, data in (("colony_indicators.json", indicators), ("agreement.json", estimates["agreement"])):
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    (out / "report.md").write_text(markdown(estimates, indicators), encoding="utf-8")
    tmp = cached.with_name(cached.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cached)                                            # per ultimo: è la chiave della cache
    return report
