"""Stratified estimates from the audit: correctness, interval coverage, edge cases."""
import numpy as np
import pytest

from bee_al.audit import EstimationInput, cohen_kappa, estimate_from_input, wilson
from bee_al.colony import colony_indicators, jensen_shannon
from bee_al.config import config_from_dict

CLASSES = ["Comune", "Media", "Rara"]
TRUE = np.array([0.80, 0.18, 0.02])
# Row = true class, column = predicted class: 5% of the common bees end up among the rare ones (false positives).
CONFUSION = np.array([[0.90, 0.05, 0.05], [0.10, 0.90, 0.00], [0.20, 0.00, 0.80]])


def simulate(seed, n_pop=20000, n_h=150):
    """Population with known prevalences, classifier with known errors, audit drawn per stratum."""
    rng = np.random.default_rng(seed)
    truth = rng.choice(3, size=n_pop, p=TRUE)
    pred = (rng.random(n_pop)[:, None] > np.cumsum(CONFUSION, axis=1)[truth]).sum(axis=1)
    strata, counts = [], np.zeros((3, 3))
    for h in range(3):
        members = np.flatnonzero(pred == h)
        take = rng.choice(members, size=min(n_h, len(members)), replace=False)
        strata.append({"id": CLASSES[h], "kind": "class", "label": CLASSES[h], "N_h": len(members), "n_h": len(take)})
        counts[h] = np.bincount(truth[take], minlength=3)
    inp = EstimationInput(CLASSES, set(), strata, counts, np.zeros(3), cc=np.bincount(pred, minlength=3))
    return inp, np.bincount(truth, minlength=3) / n_pop


def shares(result):
    return {p["class"]: p for p in result["prevalence"]}


def test_stratified_estimate_is_right_where_classify_and_count_is_biased():
    errors, raw_errors, raw = [], [], []
    for seed in range(200):
        inp, truth = simulate(seed)
        p = estimate_from_input(inp, n_boot=5, seed=seed)["prevalence"]
        errors.append(np.array([x["share"] for x in p]) - truth)
        raw_errors.append(np.array([x["cc_share"] for x in p]) - truth)
        raw.append(p[2]["cc_share"])
    bias, raw_bias = np.mean(errors, axis=0), np.mean(raw_errors, axis=0)
    assert (np.abs(bias) < 0.003).all()                                # unbiased on average for all classes
    assert np.mean(raw) == pytest.approx(0.056, abs=0.003)             # the rare class (2%) looks almost three times more frequent
    assert raw_bias[2] > 0.03 and raw_bias[0] < -0.05                  # the raw count is biased, and does not correct itself
    assert np.std(errors, axis=0)[2] < 0.01                            # and the corrected estimate stays precise


def test_interval_coverage_is_close_to_nominal():
    replicas, covered_boot, covered_analytic = 250, np.zeros(3), np.zeros(3)
    for seed in range(replicas):
        inp, truth = simulate(1000 + seed)
        p = estimate_from_input(inp, n_boot=300, seed=seed)["prevalence"]
        covered_boot += [x["ci_boot_low"] <= t <= x["ci_boot_high"] for x, t in zip(p, truth)]
        covered_analytic += [x["ci_analytic_low"] <= t <= x["ci_analytic_high"] for x, t in zip(p, truth)]
    assert (covered_boot / replicas >= 0.88).all()                     # also for the rare class
    assert (covered_analytic[:2] / replicas >= 0.88).all()             # the analytic interval holds for the non-rare classes


def _strata(*specs):
    return [{"id": sid, "kind": kind, "label": label, "N_h": N, "n_h": n} for sid, kind, label, N, n in specs]


def test_census_stratum_has_zero_variance_and_unseen_class_gets_rule_of_three():
    strata = _strata(("A", "class", "A", 30, 30), ("B", "class", "B", 1000, 50))
    counts = np.array([[27, 3, 0], [0, 50, 0]])                        # stratum A fully enumerated; C never seen
    res = estimate_from_input(EstimationInput(["A", "B", "C"], set(), strata, counts, np.zeros(3)), n_boot=400)
    p = shares(res)
    assert p["A"]["n_hat"] == pytest.approx(27) and p["A"]["n_hat_var"] == 0.0
    # The fully enumerated stratum does not vary; A's interval widens only upwards, because among the
    # 1000 never-enumerated "B" some A could be hiding (at most 3/50 of them, rule of three).
    assert p["A"]["ci_boot_low"] == pytest.approx(p["A"]["share"], abs=1e-3)
    assert p["A"]["share"] < p["A"]["ci_boot_high"] < (27 + 1000 * 3 / 50) / 1030
    assert p["B"]["n_hat_var"] == 0.0                                  # r = 1 in the sampled stratum: no analytic variance
    s = {x["id"]: x for x in res["strata"]}
    assert s["A"]["precision"] == pytest.approx(0.9) and s["B"]["precision"] == 1.0
    assert s["B"]["upper_unseen"] == pytest.approx(3 / 50) and s["A"]["upper_unseen"] == pytest.approx(0.1)
    assert s["B"]["ci_low"] > 0.9 and s["B"]["ci_high"] == 1.0         # Wilson, not 1.0 – 1.0

    # C appears in no stratum: estimate 0, but with an upper bound
    assert p["C"]["share"] == 0 and p["C"]["audit_observed"] == 0
    assert p["C"]["upper_limit"] == pytest.approx((30 * 0.1 + 1000 * 0.06) / 1030)
    assert p["A"]["upper_limit"] is None and res["complete"] and res["strata_without_audit"] == []


def test_census_only_population_is_exact():
    strata = _strata(("A", "class", "A", 10, 10), ("B", "class", "B", 5, 5))
    res = estimate_from_input(EstimationInput(["A", "B"], set(), strata, np.array([[8, 2], [1, 4]]), np.zeros(2)), n_boot=200)
    p = shares(res)
    assert p["A"]["share"] == pytest.approx(9 / 15) and p["A"]["se"] == 0.0
    assert p["A"]["ci_boot_low"] == pytest.approx(9 / 15) == pytest.approx(p["A"]["ci_boot_high"])


def test_human_labels_are_exact_counts_and_are_not_corrected():
    strata = _strata(("A", "class", "A", 100, 20), ("B", "class", "B", 100, 20))
    counts = np.array([[10, 10], [0, 20]])                             # half of the predicted "A" are actually B
    base = shares(estimate_from_input(EstimationInput(["A", "B"], set(), strata, counts, np.zeros(2)), n_boot=100))
    with_human = shares(estimate_from_input(EstimationInput(["A", "B"], set(), strata, counts, np.array([40, 0])), n_boot=100))
    assert base["A"]["n_hat"] == pytest.approx(50) and base["B"]["n_hat"] == pytest.approx(150)
    assert with_human["A"]["n_hat"] == pytest.approx(50 + 40)          # the 40 hand-labelled ones count in full, not halved
    assert with_human["A"]["human"] == 40 and with_human["B"]["n_hat"] == pytest.approx(150)
    assert with_human["A"]["share"] == pytest.approx(90 / 240)
    assert with_human["A"]["n_hat_var"] == base["A"]["n_hat_var"]      # and they add no uncertainty


def test_reject_classes_are_estimated_but_left_out_of_the_denominator():
    strata = _strata(("A", "class", "A", 800, 40), ("Sfocata|reject", "reject", "Sfocata", 200, 40))
    counts = np.array([[36, 0, 4], [10, 0, 30]])                       # columns: A, B, Sfocata
    inp = EstimationInput(["A", "B", "Sfocata"], {"Sfocata"}, strata, counts, np.array([0, 50, 10]),
                          cc=np.array([800, 50, 210]))
    res = estimate_from_input(inp, n_boot=300)
    p = shares(res)
    n_a, n_rej = 800 * 0.9 + 200 * 0.25, 800 * 0.1 + 200 * 0.75 + 10
    assert p["A"]["n_hat"] == pytest.approx(n_a) and p["Sfocata"]["n_hat"] == pytest.approx(n_rej)
    assert p["A"]["share"] == pytest.approx(n_a / (n_a + 50)) and p["B"]["share"] == pytest.approx(50 / (n_a + 50))
    assert p["Sfocata"]["share"] is None and p["Sfocata"]["reject"] and p["Sfocata"]["cc_share"] is None
    assert p["A"]["cc_share"] == pytest.approx(800 / 850)
    assert res["totals"]["valid_hat"] == pytest.approx(n_a + 50)
    assert res["totals"]["reject_share"] == pytest.approx(n_rej / (n_a + 50 + n_rej))
    lo, hi = res["totals"]["reject_share_ci"]
    assert lo < res["totals"]["reject_share"] < hi
    s = {x["id"]: x for x in res["strata"]}
    assert s["Sfocata|reject"]["precision"] == pytest.approx(0.75)     # share of true rejects among the predicted rejects
    assert s["Sfocata|reject"]["answers"] == {"A": 10, "Sfocata": 30}  # 10 valid bees rejected by mistake


def test_unknown_answers_are_counted_apart_and_missing_strata_are_flagged():
    strata = _strata(("A", "class", "A", 100, 10), ("B", "class", "B", 60, 10), ("__unassigned__", "unassigned", "", 40, 10))
    counts = np.array([[6, 2], [0, 0], [0, 0]])
    inp = EstimationInput(["A", "B"], set(), strata, counts, np.zeros(2), unknown=np.array([2, 0, 0]))
    res = estimate_from_input(inp, n_boot=100)
    s = {x["id"]: x for x in res["strata"]}
    assert (s["A"]["n_labeled"], s["A"]["unknown"], s["A"]["precision"]) == (8, 2, 0.75)   # "Scarta ape" answers outside n_h
    assert res["strata_without_audit"] == ["B", "__unassigned__"] and res["complete"] is False
    assert s["B"]["precision"] is None and s["__unassigned__"]["ci_low"] is None
    p = shares(res)
    assert p["B"]["n_hat"] == pytest.approx(100 * 0.25 + 60)            # without an audit the predicted class is kept
    assert p["A"]["n_hat"] == pytest.approx(75)                         # and unassigned bees without an audit do not count


def test_wilson_interval():
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0713, abs=1e-3)          # close to the rule of three (0.06)
    lo, hi = wilson(45, 50)
    assert lo == pytest.approx(0.786, abs=2e-3) and hi == pytest.approx(0.957, abs=2e-3)


def test_jensen_shannon_properties():
    rng = np.random.default_rng(0)
    p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))
    assert 0 < jensen_shannon(p, q) <= 1
    assert jensen_shannon([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)    # disjoint profiles: maximum
    assert jensen_shannon([2, 2], [1, 1]) == pytest.approx(0.0)          # normalises by itself
    batch = jensen_shannon(rng.dirichlet(np.ones(4), size=(7, 3)), rng.dirichlet(np.ones(4)))
    assert batch.shape == (7, 3) and (batch >= 0).all() and (batch <= 1).all()


def test_cohen_kappa():
    perfect = cohen_kappa([("A", "A")] * 10 + [("B", "B")] * 6 + [("C", "C")] * 4)
    assert perfect["kappa"] == pytest.approx(1.0) and perfect["agreement"] == 1.0
    assert perfect["per_class"] == {"A": pytest.approx(1.0), "B": pytest.approx(1.0), "C": pytest.approx(1.0)}
    assert perfect["matrix"]["A"] == {"A": 10, "B": 0, "C": 0}

    rng = np.random.default_rng(1)
    random_pairs = list(zip(rng.choice(list("ABC"), 4000), rng.choice(list("ABC"), 4000)))
    assert cohen_kappa(random_pairs)["kappa"] == pytest.approx(0.0, abs=0.04)

    # classic example: 70% agreement, 50% expected -> kappa 0.4
    k = cohen_kappa([("Y", "Y")] * 20 + [("Y", "N")] * 5 + [("N", "Y")] * 10 + [("N", "N")] * 15)
    assert k["agreement"] == pytest.approx(0.7) and k["kappa"] == pytest.approx(0.4) and k["n_pairs"] == 50
    assert k["per_class"]["Y"] == pytest.approx(0.4)
    assert cohen_kappa([])["kappa"] is None and cohen_kappa([("A", "A")] * 5)["kappa"] is None


# ------------------------------ groups and colony --------------------------- #

def grouped_input(per_group, counts, classes=("Sana", "Malata"), reject=(), human=None, n_frames=6, level="session"):
    """``per_group``: {group: bees per stratum}, split across ``n_frames`` frames of different composition."""
    names = list(per_group)
    H, K = len(counts), len(classes)
    strata = [{"id": f"s{h}", "kind": "class", "label": classes[h % K], "N_h": int(sum(per_group[g][h] for g in names)),
               "n_h": int(sum(counts[h]))} for h in range(H)]
    N = np.array([per_group[g] for g in names], dtype=float)
    Hg = np.zeros((len(names), K)) if human is None else np.array([human[g] for g in names], dtype=float)
    frame_group = np.repeat(np.arange(len(names)), n_frames)
    rng = np.random.default_rng(0)
    frame_N = np.concatenate([
        np.stack([rng.multinomial(int(N[g, h]), rng.dirichlet(np.full(n_frames, 2.0))) for h in range(H)], axis=1)
        for g in range(len(names))]).astype(float)
    info = {"n_bees": [int(N[i].sum() + Hg[i].sum()) + 10 for i in range(len(names))], "unassigned": [6] * len(names),
            "unknown": [4] * len(names), "frames_listed": [n_frames] * len(names), "raw_bees": [600] * len(names),
            "cleaned_accepted": [450] * len(names), "cleaned_discarded": [150] * len(names)}
    groups = {level: {"names": names, "N": N, "human": Hg, "frame_group": frame_group,
                      "frame_N": frame_N, "frame_human": np.repeat(Hg / n_frames, n_frames, axis=0),
                      "confirmed": np.round(N @ np.array(counts, dtype=float) / max(1.0, float(np.sum(counts)))), "info": info}}
    return EstimationInput(list(classes), set(reject), strata, np.array(counts, dtype=float), Hg.sum(axis=0), groups=groups)


def test_group_estimates_and_missing_strata_in_a_group():
    counts = [[45, 5], [4, 36], [0, 0]]                                 # the third stratum has no audit answers
    inp = grouped_input({"g1": [300, 20, 0], "g2": [100, 200, 50]}, counts)
    res = estimate_from_input(inp, n_boot=300)
    groups = {g["group"]: g for g in res["groups"]["session"]["groups"]}
    assert groups["g1"]["classes"]["Malata"]["share"] == pytest.approx((300 * 0.1 + 20 * 0.9) / 320)
    assert groups["g1"]["missing_strata"] == [] and groups["g2"]["missing_strata"] == ["s2"]
    for g in groups.values():
        for stats in g["classes"].values():
            assert stats["ci_low"] <= stats["share"] <= stats["ci_high"]
    assert groups["g1"]["classes"]["Malata"]["ci_high"] < groups["g2"]["classes"]["Malata"]["ci_low"]
    assert res["groups"]["session"]["frame_resampling"] is True and groups["g1"]["n_frames"] == 6
    assert res["_boot"]["groups"]["session"].shape == (300, 2, 2)


def colony_cfg(**colony):
    return config_from_dict({"colony": {"demographic_classes": ["sana"], "disease_classes": ["malata", "Mai Vista"],
                                        "n_boot": 400, **colony}})


def test_colony_indicators_per_group():
    counts = [[190, 10], [20, 180]]
    inp = grouped_input({"sano1": [900, 20], "sano2": [880, 30], "sano3": [910, 25], "malato": [300, 700]}, counts)
    res = estimate_from_input(inp, n_boot=400)
    cfg = colony_cfg(reference_groups=["sano1", "sano2", "sano3", "non_esiste"], min_detected=3)
    ind = colony_indicators(res, cfg)
    assert ind["group_by"] == "session" and [g["group"] for g in ind["groups"]] == ["sano1", "sano2", "sano3", "malato"]
    assert ind["disease_classes"] == ["Malata", "Mai Vista"] and "non_esiste" in ind["notes"][0]
    by = {g["group"]: g for g in ind["groups"]}

    i1 = by["sano1"]["I1_demographics"]["Sana"]                         # I1: corrected share with interval
    assert i1["share"] == pytest.approx((900 * 0.95 + 20 * 0.1) / 920) and i1["ci_low"] < i1["share"] < i1["ci_high"]

    sick = by["malato"]["I2_disease"]["Malata"]                         # I2
    assert sick["status"] == "detected" and sick["ci_low"] > 0.5 and sick["confirmed"] >= 3
    assert by["malato"]["I2_disease"]["Mai Vista"] == {"share": None, "note": "class not found among the non-reject classes"}

    ref = ind["reference"]                                              # I3
    assert ref["groups"] == ["sano1", "sano2", "sano3"] and 0 < ref["jsd_threshold"] < 0.01
    assert by["malato"]["I3_jsd"]["above_threshold"] and by["malato"]["I3_jsd"]["value"] > 0.2
    assert not by["sano1"]["I3_jsd"]["above_threshold"]
    assert by["malato"]["I3_jsd"]["ci_low"] <= by["malato"]["I3_jsd"]["value"] <= by["malato"]["I3_jsd"]["ci_high"]

    assert by["sano1"]["I4_unexplained"]["share"] == pytest.approx(10 / 930)          # I4
    assert by["sano1"]["strength"] == {"frames": 6, "raw_bees": 600, "raw_bees_per_frame": 100.0}
    assert by["sano1"]["quality"]["cleaning_discard_rate"] == pytest.approx(0.25)

    # traffic light: two rules together = red, and it says which
    assert by["malato"]["light"]["color"] == "red"
    assert [r["rule"] for r in by["malato"]["light"]["rules"]] == ["disease_detected", "profile_divergent"]
    assert by["sano1"]["light"]["color"] in ("green", "yellow")

    explicit = colony_indicators(res, colony_cfg(), reference={"Sana": 0.3, "Malata": 0.7})
    assert explicit["reference"]["explicit"] and {g["group"]: g for g in explicit["groups"]}["malato"]["I3_jsd"]["value"] < 0.01
    relaxed = colony_indicators(res, colony_cfg(red_min_rules=3, light_rules=["disease_detected"], reference_groups=["sano1", "sano2", "sano3"]))
    assert {g["group"]: g for g in relaxed["groups"]}["malato"]["light"] == {
        "color": "yellow", "rules": [{"rule": "disease_detected", "detail": "Malata"}]}


def test_disease_status_not_detected_and_uncertain():
    counts = [[200, 0], [10, 2]]                                        # almost no diseased bees in the audit
    inp = grouped_input({"pulito": [1000, 0], "dubbio": [900, 40]}, counts)
    ind = colony_indicators(estimate_from_input(inp, n_boot=400), colony_cfg(min_detected=50))
    by = {g["group"]: g for g in ind["groups"]}
    clean = by["pulito"]["I2_disease"]["Malata"]
    assert clean["status"] == "not_detected" and clean["share"] == 0
    assert clean["upper_limit"] == pytest.approx(3 / 200)               # rule of three on the group's stratum
    assert by["dubbio"]["I2_disease"]["Malata"]["status"] == "uncertain"   # estimate > 0 but few confirmed bees
    assert by["pulito"]["light"]["rules"] == [] or by["pulito"]["light"]["rules"][0]["rule"] == "profile_divergent"


def test_trend_between_sessions_is_significant_only_when_it_should():
    counts = [[190, 10], [20, 180]]
    per_group = {"s1": [900, 50], "s2": [880, 60], "s3": [400, 600]}     # between s2 and s3 the diseased bees surge
    res = estimate_from_input(grouped_input(per_group, counts), n_boot=500)
    ind = colony_indicators(res, colony_cfg(time_order={"*": ["s1", "s2", "s3", "s4"]}))
    steps = {(t["from"], t["to"]): t for t in ind["trend"]}
    small, big = steps[("s1", "s2")]["deltas"], steps[("s2", "s3")]["deltas"]
    assert not small["Malata"]["significant"] and small["Malata"]["ci_low"] < 0 < small["Malata"]["ci_high"]
    assert big["Malata"]["significant"] and big["Malata"]["delta"] == pytest.approx(0.456, abs=0.01) and big["Malata"]["ci_low"] > 0.2
    assert big["Sana"]["significant"] and big["Sana"]["delta"] < 0 and big["I3_jsd"]["significant"]
    assert steps[("s3", "s4")]["missing"] == ["s4"] and steps[("s3", "s4")]["deltas"] == {}

    # the interval of the difference is computed replicate by replicate: narrower than the sum of the two intervals
    b = res["_boot"]["groups"]["session"]
    share = b[:, :, 1] / b.sum(axis=2)
    naive = (np.percentile(share[:, 1], 97.5) - np.percentile(share[:, 0], 2.5)) - (np.percentile(share[:, 1], 2.5) - np.percentile(share[:, 0], 97.5))
    assert small["Malata"]["ci_high"] - small["Malata"]["ci_low"] < naive

    per_colony = colony_indicators(res, colony_cfg(time_order={"hiveA": ["s1", "s2"]}))
    assert per_colony["trend"] == [] and "hiveA" in per_colony["notes"][0]   # says so when the snapshot has no hives


def test_indicators_need_the_requested_level():
    res = estimate_from_input(grouped_input({"a": [100, 10]}, [[10, 0], [0, 10]]), n_boot=50)
    ind = colony_indicators(res, colony_cfg(group_by="colony"))
    assert ind["groups"] == [] and "colony" in ind["notes"][0]
    for bad in ({"group_by": "hive"}, {"light_rules": ["boh"]}, {"min_detected": 0}, {"time_order": {"a": "s1"}}):
        with pytest.raises(ValueError):
            config_from_dict({"colony": bad})
    assert config_from_dict({"colony": {"time_order": {"hiveA": ["s1", "s2"]}}}).colony.time_order == {"hiveA": ["s1", "s2"]}
