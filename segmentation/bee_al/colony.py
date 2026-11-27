"""Colony indicators computed on the audit-corrected estimates (see ``bee_al/audit.py``)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .audit import class_normalizer, wilson
from .config import ALConfig


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Base-2 Jensen–Shannon divergence, in [0, 1], along the last axis.

    ``JSD(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M)`` with ``M = ½ (P + Q)``. It is 0 for identical profiles.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p = np.divide(p, p.sum(axis=-1, keepdims=True), out=np.zeros_like(p), where=p.sum(axis=-1, keepdims=True) > 0)
    q = np.divide(q, q.sum(axis=-1, keepdims=True), out=np.zeros_like(q), where=q.sum(axis=-1, keepdims=True) > 0)
    p, q = np.broadcast_arrays(p, q)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ratio = np.divide(a, b, out=np.ones_like(a), where=(a > 0) & (b > 0))
        return np.where(a > 0, a * np.log2(ratio), 0.0).sum(axis=-1)

    return np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _profiles(totals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Shares over the non-reject classes; the reject columns stay at zero."""
    t = np.where(valid, totals, 0.0)
    v = t.sum(axis=-1, keepdims=True)
    return np.divide(t, v, out=np.zeros_like(t), where=v > 0)


def colony_indicators(estimates: dict[str, Any], cfg: ALConfig, reference: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """Indicators I1–I5 for each group (``colony.group_by``: frame, session, colony or dataset).

    - **I1 – demographic profile**: corrected shares, with interval, of the ``demographic_classes``.
    - **I2 – disease signs**: for each class in ``disease_classes``, share and interval;
      ``detected`` if the lower bound is > 0 and the bees confirmed by a human in the group
      (audit answers plus hand-given labels) are at least ``min_detected``;
      ``not_detected`` with an upper bound (rule of three) if the estimate is zero; otherwise ``uncertain``.
    - **I3 – Jensen–Shannon divergence** between the group profile and the reference
      profile: mean of the ``reference_groups`` (of all groups if the list is empty), or
      the ``reference`` profile passed explicitly. Anomaly threshold: mean + 2 standard
      deviations of the JSD of the reference groups from their mean.
    - **I4 – unexplained share**: (unassigned + "Scarta ape") / bees in the group.
    - **I5 – trend**: for each hive in ``time_order``, the difference between consecutive sessions
      of I1, I2 and I3, with an interval computed replicate by replicate from the bootstrap;
      ``significant`` if the interval does not contain zero. The key ``"*"`` means the
      sessions of all hives together.
    - Colony strength (raw bees per frame), quality (estimated reject share, rejections
      by the automatic cleaning), share above the anomaly-score threshold if available.
    - Traffic light: each group reports which rules of ``light_rules`` were triggered.
    """
    c = cfg.colony
    normalize = class_normalizer(cfg)
    boot = estimates["_boot"]
    classes, valid = boot["classes"], boot["valid"]
    col = {name: j for j, name in enumerate(classes)}
    demographic = [normalize(x) for x in c.demographic_classes]
    disease = [normalize(x) for x in c.disease_classes]
    level = c.group_by

    out: dict[str, Any] = {"group_by": level, "demographic_classes": demographic, "disease_classes": disease,
                           "groups": [], "trend": [], "notes": []}
    if level not in estimates["groups"]:
        out["notes"].append(f"no '{level}' information in this snapshot: no group indicators")
        return out
    info = estimates["groups"][level]["groups"]
    names = boot["names"][level]
    totals_boot = boot["groups"][level]                               # (B, G, K)
    point = np.array([[g["classes"].get(k, {}).get("n_hat", 0.0) for k in classes] for g in info])
    profile, profile_boot = _profiles(point, valid), _profiles(totals_boot, valid)

    ref_names = [n for n in c.reference_groups if n in names] or list(names)
    missing_refs = [n for n in c.reference_groups if n not in names]
    if missing_refs:
        out["notes"].append(f"reference groups not found: {missing_refs}")
    ref_idx = [names.index(n) for n in ref_names]
    if reference is not None:
        fixed = np.array([reference.get(k, 0.0) if valid[j] else 0.0 for j, k in enumerate(classes)])
        ref, ref_boot = fixed / max(fixed.sum(), 1e-12), None
    else:
        ref, ref_boot = profile[ref_idx].mean(axis=0), profile_boot[:, ref_idx, :].mean(axis=1)
    jsd = jensen_shannon(profile, ref)
    jsd_boot = jensen_shannon(profile_boot, ref if ref_boot is None else ref_boot[:, None, :])
    ref_jsd = jsd[ref_idx]
    threshold = float(ref_jsd.mean() + 2 * ref_jsd.std())
    out["reference"] = {"groups": ref_names if reference is None else [], "explicit": reference is not None,
                        "profile": {k: float(ref[j]) for j, k in enumerate(classes) if valid[j]},
                        "jsd_threshold": threshold, "jsd_mean": float(ref_jsd.mean()), "jsd_std": float(ref_jsd.std())}

    def class_share(gi: int, name: str) -> dict[str, Any]:
        if name not in col or not valid[col[name]]:
            return {"share": None, "note": "class not found among the non-reject classes"}
        lo, hi = _ci(profile_boot[:, gi, col[name]])
        return {"share": float(profile[gi, col[name]]), "ci_low": lo, "ci_high": hi}

    for gi, g in enumerate(info):
        signs = {}
        for name in disease:
            entry = class_share(gi, name)
            if entry["share"] is not None:
                stats = g["classes"][name]
                entry["confirmed"] = stats["confirmed"]
                if entry["ci_low"] > 0 and stats["confirmed"] >= c.min_detected:
                    entry["status"] = "detected"
                elif entry["share"] == 0:
                    entry["status"], entry["upper_limit"] = "not_detected", stats["upper_limit"]
                else:
                    entry["status"] = "uncertain"
            signs[name] = entry
        lo, hi = _ci(jsd_boot[:, gi])
        n_bees = g.get("n_bees", 0)
        unexplained = (g.get("unassigned", 0) + g.get("unknown", 0)) / n_bees if n_bees else None
        processed = g.get("cleaned_accepted", 0) + g.get("cleaned_discarded", 0)
        row: dict[str, Any] = {
            "group": g["group"],
            "I1_demographics": {name: class_share(gi, name) for name in demographic},
            "I2_disease": signs,
            "I3_jsd": {"value": float(jsd[gi]), "ci_low": lo, "ci_high": hi, "threshold": threshold,
                       "above_threshold": bool(jsd[gi] > threshold)},
            "I4_unexplained": {"share": unexplained, "unassigned": g.get("unassigned", 0), "unknown": g.get("unknown", 0),
                               "n_bees": n_bees},
            "strength": {"frames": g.get("frames_listed", 0), "raw_bees": g.get("raw_bees", 0),
                         "raw_bees_per_frame": g["raw_bees"] / g["frames_listed"] if g.get("frames_listed") else None},
            "quality": {"reject_share": g["reject_share"], "reject_share_ci": g["reject_share_ci"],
                        "cleaning_discarded": g.get("cleaned_discarded", 0),
                        "cleaning_discard_rate": g["cleaned_discarded"] / processed if processed else None},
            "missing_strata": g["missing_strata"],
        }
        if g.get("anomaly_scored"):
            a_lo, a_hi = wilson(g["anomaly_above"], g["anomaly_scored"])
            row["anomaly"] = {"share": g["anomaly_above"] / g["anomaly_scored"], "ci_low": a_lo, "ci_high": a_hi,
                              "threshold": c.anomaly_threshold, "n_scored": g["anomaly_scored"]}

        fired = []
        detected = [name for name, e in signs.items() if e.get("status") == "detected"]
        if "disease_detected" in c.light_rules and detected:
            fired.append({"rule": "disease_detected", "detail": ", ".join(detected)})
        if "profile_divergent" in c.light_rules and row["I3_jsd"]["above_threshold"]:
            fired.append({"rule": "profile_divergent", "detail": f"JSD {jsd[gi]:.3f} > {threshold:.3f}"})
        if "unexplained_high" in c.light_rules and c.max_unexplained > 0 and unexplained is not None and unexplained > c.max_unexplained:
            fired.append({"rule": "unexplained_high", "detail": f"{unexplained:.3f} > {c.max_unexplained:.3f}"})
        row["light"] = {"color": "red" if len(fired) >= c.red_min_rules else ("yellow" if fired else "green"), "rules": fired}
        out["groups"].append(row)

    # ---------------------------------- I5 ---------------------------------- #
    for colony_name, sessions in c.time_order.items():
        lv = "session" if colony_name == "*" else "colony_session"
        if lv not in boot["groups"]:
            out["notes"].append(f"trend for '{colony_name}': no {lv.replace('_', ' + ')} information in this snapshot")
            continue
        lv_names = boot["names"][lv]
        lv_point = np.array([[g["classes"].get(k, {}).get("n_hat", 0.0) for k in classes] for g in estimates["groups"][lv]["groups"]])
        prof, prof_boot = _profiles(lv_point, valid), _profiles(boot["groups"][lv], valid)
        ref_b = ref if ref_boot is None else ref_boot[:, None, :]
        d, d_boot = jensen_shannon(prof, ref), jensen_shannon(prof_boot, ref_b)
        keys = [s if colony_name == "*" else f"{colony_name}/{s}" for s in sessions]
        for (s0, k0), (s1, k1) in zip(zip(sessions, keys), zip(sessions[1:], keys[1:])):
            step: dict[str, Any] = {"colony": colony_name, "from": s0, "to": s1, "deltas": {}}
            if k0 not in lv_names or k1 not in lv_names:
                step["missing"] = [k for k in (k0, k1) if k not in lv_names]
                out["trend"].append(step)
                continue
            i0, i1 = lv_names.index(k0), lv_names.index(k1)

            def delta(value: float, samples: np.ndarray) -> dict[str, Any]:
                lo, hi = _ci(samples)
                return {"delta": float(value), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}

            for name in dict.fromkeys(demographic + disease):
                if name in col and valid[col[name]]:
                    j = col[name]
                    step["deltas"][name] = delta(prof[i1, j] - prof[i0, j], prof_boot[:, i1, j] - prof_boot[:, i0, j])
            step["deltas"]["I3_jsd"] = delta(d[i1] - d[i0], d_boot[:, i1] - d_boot[:, i0])
            out["trend"].append(step)
    return out
