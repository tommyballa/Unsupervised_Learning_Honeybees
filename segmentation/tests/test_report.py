"""Full workflow on synthetic data: labelling -> snapshot -> audit -> estimates -> indicators -> report."""
import csv
import json
import subprocess
import sys
from collections import Counter

import pytest
import yaml

from bee_al import hub as al_hub
from bee_al import report as report_module
from bee_al.audit import estimate
from bee_al.config import REPO_ROOT
from bee_al.engine import ALEngine
from bee_al.report import REPORT_FILES, build_report
from bee_al.snapshot import read_csv

from conftest import build_colony_dataset
from test_audit import label_audit
from test_engine import label_batch

PATTERN = r"(?P<colony>hive[A-Z])_(?P<session>s\d)"
REJECT = "Class_3"


@pytest.fixture
def flow(tmp_path):
    """Larger dataset, 5 labelling rounds, snapshot and a complete audit done by two annotators."""
    frames = {
        "tr": {"hiveA_s1_001.JPG": 150, "hiveA_s1_002.JPG": 130, "hiveA_s2_003.JPG": 140, "hiveB_s1_004.JPG": 160, "hiveB_s2_005.JPG": 120},
        "te": {"hiveA_s2_006.JPG": 110, "hiveB_s1_007.JPG": 90, "hiveB_s2_008.JPG": 100},
    }
    data = build_colony_dataset(tmp_path, frames, extra_cfg={
        "labels": {"reject_classes": [REJECT]},
        "frames": {"frame_pattern": PATTERN},
        "audit": {"per_stratum": 30, "double_fraction": 0.2},
        "colony": {"demographic_classes": ["class_0", "Class_1"], "disease_classes": ["Class_2"], "n_boot": 300,
                   "time_order": {"hiveA": ["s1", "s2"], "*": ["s1", "s2"]}},
    })
    engine = ALEngine(data.cfg, data.emb)
    for _ in range(5):
        label_batch(engine, data.truth)
    engine.export_snapshot("final", "a@x.it")
    engine.create_audit("final")
    label_audit(engine, data.truth, "a@x.it")
    label_audit(engine, data.truth, "b@x.it")
    data.engine = engine
    data.folder = data.cfg.results_dir / "snapshots" / "final"
    return data


def true_shares(flow):
    """True prevalences among the valid bees of the snapshot (all except validation and "Scarta ape")."""
    rows = read_csv(flow.folder / "predictions.csv")
    truth = Counter(flow.truth[r["image_id"]] for r in rows if r["status"] not in ("validation", "unknown", "no_prediction"))
    valid = sum(n for k, n in truth.items() if k != REJECT)
    return {k: n / valid for k, n in truth.items() if k != REJECT}, truth


def test_end_to_end_estimates_are_close_to_the_truth(flow):
    assert flow.engine.audit_progress()["done"] == flow.engine.audit_progress()["total"]
    report = build_report(flow.folder, flow.cfg, store=flow.engine.store)
    est = report["estimates"]
    assert est["complete"] and est["strata_without_audit"] == [] and est["frame_resampling"] is True
    assert est["reject_classes"] == [REJECT] and est["snapshot"] == "final"

    shares, truth = true_shares(flow)
    prevalence = {p["class"]: p for p in est["prevalence"]}
    assert set(shares) == {p for p, v in prevalence.items() if not v["reject"]}
    for name, true in shares.items():
        p = prevalence[name]
        assert p["share"] == pytest.approx(true, abs=0.05), name
        assert p["ci_boot_low"] - 0.03 <= true <= p["ci_boot_high"] + 0.03, name
        assert p["ci_boot_low"] <= p["share"] <= p["ci_boot_high"]
    assert sum(p["share"] for p in prevalence.values() if not p["reject"]) == pytest.approx(1.0)

    # block E: the reject class is estimated but stays out of the denominator, and has its own audit stratum
    assert prevalence[REJECT]["share"] is None and prevalence[REJECT]["n_hat"] == pytest.approx(truth[REJECT], rel=0.25)
    assert est["totals"]["valid_hat"] == pytest.approx(sum(n for k, n in truth.items() if k != REJECT), rel=0.05)
    strata = {s["id"]: s for s in est["strata"]}
    assert strata[f"{REJECT}|reject"]["kind"] == "reject" and strata[f"{REJECT}|reject"]["n_labeled"] > 0
    assert "__unassigned__" in strata and all(s["n_labeled"] == s["n_sampled"] for s in est["strata"])
    assert REJECT not in flow.engine.final_clusters()
    assert REJECT in flow.engine.status()["reject_counts"]

    # agreement: two annotators answering with the same oracle
    assert est["agreement"]["n_pairs"] == est["audit"]["double"] > 0 and est["agreement"]["kappa"] == pytest.approx(1.0)

    # per-group estimates at all available levels
    assert set(est["groups"]) == {"dataset", "session", "colony", "frame", "colony_session"}
    sessions = {g["group"]: g for g in est["groups"]["session"]["groups"]}
    assert set(sessions) == {"s1", "s2"} and all(not g["missing_strata"] for g in sessions.values())
    assert sum(g["population"] + g["human"] for g in sessions.values()) == est["totals"]["population"] + est["totals"]["human"]
    assert len(est["groups"]["frame"]["groups"]) == 8 and est["groups"]["frame"]["frame_resampling"] is False


def test_indicators_are_produced_for_every_group(flow):
    ind = build_report(flow.folder, flow.cfg, store=flow.engine.store)["indicators"]
    assert ind["group_by"] == "session" and [g["group"] for g in ind["groups"]] == ["s1", "s2"]
    assert ind["demographic_classes"] == ["Class_0", "Class_1"] and ind["disease_classes"] == ["Class_2"]
    rows = read_csv(flow.folder / "predictions.csv")
    frames = read_csv(flow.folder / "frames.csv")
    for g in ind["groups"]:
        assert set(g["I1_demographics"]) == {"Class_0", "Class_1"}
        assert all(0 <= e["ci_low"] <= e["share"] <= e["ci_high"] <= 1 for e in g["I1_demographics"].values())
        assert g["I2_disease"]["Class_2"]["status"] in ("detected", "uncertain", "not_detected")
        assert 0 <= g["I3_jsd"]["value"] <= 1 and g["light"]["color"] in ("green", "yellow", "red")
        mine = [r for r in rows if r["session"] == g["group"]]
        unexplained = sum(r["status"] in ("unassigned", "unknown") for r in mine)
        assert g["I4_unexplained"]["share"] == pytest.approx(unexplained / len(mine)) and g["I4_unexplained"]["n_bees"] == len(mine)
        mine_frames = [f for f in frames if f["session"] == g["group"]]
        assert g["strength"]["frames"] == len(mine_frames) and g["strength"]["raw_bees"] == sum(int(f["raw_bees"]) for f in mine_frames)
        assert g["quality"]["cleaning_discard_rate"] == pytest.approx(1 / 7, abs=0.01)   # the fake cleaning discards 1 in 7
        assert 0 < g["quality"]["reject_share"] < 0.5
    steps = {(t["colony"], t["from"], t["to"]) for t in ind["trend"]}
    assert steps == {("hiveA", "s1", "s2"), ("*", "s1", "s2")}
    assert all({"Class_0", "Class_1", "Class_2", "I3_jsd"} <= set(t["deltas"]) for t in ind["trend"])


def test_report_files_and_cache(flow, monkeypatch):
    report = build_report(flow.folder, flow.cfg, store=flow.engine.store)
    out = flow.folder / "report"
    assert {p.name for p in out.iterdir()} == set(REPORT_FILES)
    strata = read_csv(out / "strata.csv")
    assert {"id", "N_h", "n_labeled", "unknown", "precision", "ci_low", "ci_high"} <= set(strata[0])
    prevalence = read_csv(out / "prevalence.csv")
    assert {"class", "cc_share", "share", "ci_analytic_low", "ci_boot_low", "upper_limit"} <= set(prevalence[0])
    groups = read_csv(out / "groups.csv")
    assert {r["level"] for r in groups} == {"dataset", "session", "colony", "frame", "colony_session"}
    indicators = read_csv(out / "colony_indicators.csv")
    assert {"I1_demographics", "I2_disease", "I3_jsd", "I4_unexplained", "I5_trend", "strength", "quality", "light"} <= {r["indicator"] for r in indicators}
    assert json.loads((out / "agreement.json").read_text())["n_pairs"] > 0
    assert json.loads((out / "colony_indicators.json").read_text())["group_by"] == "session"
    md = (out / "report.md").read_text()
    for text in ("# Audit report", "Precision per stratum", "Classify and count", "Colony indicators by session", "Trend between sessions", "Class_2"):
        assert text in md
    assert "not complete" not in md

    # cached: a second request recomputes nothing
    monkeypatch.setattr(report_module, "estimate_from_input", lambda *a, **k: pytest.fail("should have used the cache"))
    assert build_report(flow.folder, flow.cfg, store=flow.engine.store) == report
    monkeypatch.undo()

    # a new audit answer, or a different config, invalidates the cache
    bee = flow.engine.audit["order"][0]
    flow.engine.submit("a@x.it", bee, "Class_2" if flow.truth[bee] != "Class_2" else "Class_0", mode="audit")
    changed = build_report(flow.folder, flow.cfg, store=flow.engine.store)
    assert changed["fingerprint"] != report["fingerprint"]
    assert changed["estimates"]["prevalence"] != report["estimates"]["prevalence"]
    flow.cfg.colony.min_detected = 99
    assert build_report(flow.folder, flow.cfg, store=flow.engine.store)["fingerprint"] != changed["fingerprint"]


def test_partial_audit_is_reported_as_incomplete(tmp_path):
    data = build_colony_dataset(tmp_path, extra_cfg={"audit": {"per_stratum": 8}, "frames": {"frame_pattern": PATTERN}})
    engine = ALEngine(data.cfg, data.emb)
    for _ in range(3):
        label_batch(engine, data.truth)
    engine.export_snapshot("p", "a@x.it")
    with pytest.raises(FileNotFoundError, match="no audit set"):
        build_report(data.cfg.results_dir / "snapshots" / "p", data.cfg, store=engine.store)
    engine.create_audit("p")
    label_audit(engine, {i: "non so" for i in data.truth}, "a@x.it", limit=2)
    label_audit(engine, data.truth, "a@x.it", limit=6)
    est = estimate(data.cfg.results_dir / "snapshots" / "p", data.cfg, n_boot=50, store=engine.store)
    assert est["complete"] is False and est["audit"]["answered"] == 8 and est["totals"]["audit_unknown"] == 2
    assert est["strata_without_audit"]                                  # some strata have no answers yet
    report = build_report(data.cfg.results_dir / "snapshots" / "p", data.cfg, store=engine.store)
    assert "not complete" in (data.cfg.results_dir / "snapshots" / "p" / "report" / "report.md").read_text()
    assert report["estimates"]["agreement"]["n_pairs"] == 0


def test_anomaly_scores_feed_the_anomaly_indicator(flow, tmp_path):
    scores = tmp_path / "scores.csv"
    with open(scores, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "score"])
        writer.writerows((i, 1.0 if "hiveB_s2" in flow.frame_of[i] else 0.1) for i in flow.emb.ids)
    flow.cfg.colony.anomaly_threshold = 0.5
    ind = build_report(flow.folder, flow.cfg, store=flow.engine.store, anomaly_scores_path=scores)["indicators"]
    by = {g["group"]: g for g in ind["groups"]}
    rows = read_csv(flow.folder / "predictions.csv")
    s2 = [r for r in rows if r["session"] == "s2"]
    assert by["s1"]["anomaly"]["share"] == 0.0 and by["s1"]["anomaly"]["ci_high"] < 0.01
    assert by["s2"]["anomaly"]["share"] == pytest.approx(sum("hiveB" in r["frame"] for r in s2) / len(s2))
    assert by["s2"]["anomaly"]["ci_low"] < by["s2"]["anomaly"]["share"] < by["s2"]["anomaly"]["ci_high"]
    assert (flow.folder / "report" / "anomaly_scores.csv").exists()
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        build_report(flow.folder, flow.cfg, store=flow.engine.store, anomaly_scores_path=bad)


def test_audit_report_script(flow, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    data = flow.cfg.to_dict()
    data["paths"] = {k: (str(flow.cfg.resolve(v)) if v else "") if isinstance(v, str) else {n: str(flow.cfg.resolve(d)) for n, d in v.items()}
                     for k, v in data["paths"].items()}
    config_path.write_text(yaml.safe_dump(data))
    script = REPO_ROOT / "scripts" / "audit_report.py"
    run = subprocess.run([sys.executable, str(script), "--snapshot", "final", "--config", str(config_path), "--n-boot", "100"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "classify and count" in run.stdout and "Class_0" in run.stdout and "report" in run.stdout
    assert {p.name for p in (flow.folder / "report").iterdir()} == set(REPORT_FILES)
    assert json.loads((flow.folder / "report" / "report.json").read_text())["estimates"]["n_boot"] == 100
    missing = subprocess.run([sys.executable, str(script), "--snapshot", "boh", "--config", str(config_path)], capture_output=True, text=True)
    assert missing.returncode != 0 and "Report not created" in missing.stderr


def test_hub_results_page_and_downloads(flow, hub_app, monkeypatch, tmp_path):
    monkeypatch.setattr(al_hub.STATE, "engine", flow.engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    assert c.get("/al-audit-results?snapshot=final").status_code == 302 and c.get("/al-audit-file/final/report.md").status_code == 403
    c.post("/", data={"email": "t@x.it"})

    r = c.get("/al-audit-results?snapshot=final")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    for text in ("Audit results", "Precision per stratum", "Classify and count", "Stratified estimate", "Colony indicators by session",
                 "Demographic profile (I1)", "Disease signs (I2)", "Trend between sessions (I5)", "Cohen", "Class_2", "reject"):
        assert text in html, text
    assert "not complete" not in html
    assert c.get("/al-audit-results").status_code == 200                # without a name: the active audit
    assert "results</a>" in c.get("/al-status").get_data(as_text=True)

    for name in REPORT_FILES:
        d = c.get(f"/al-audit-file/final/{name}")
        assert d.status_code == 200 and "attachment" in d.headers["Content-Disposition"] and f"final_{name}" in d.headers["Content-Disposition"]
    for url in ("/al-audit-file/final/predictions.csv", "/al-audit-file/final/..%2Fmeta.json", "/al-audit-file/boh/report.md",
                "/al-audit-results?snapshot=boh", "/al-audit-results?snapshot=../x"):
        assert c.get(url).status_code == 404


def test_sessions_from_folders_drive_group_estimates_and_trend(tmp_path):
    """Two sessions spanning the dataset folders: estimates and trends are per session, not per folder."""
    from test_frames import _session_dirs

    data = build_colony_dataset(tmp_path, extra_cfg={"audit": {"per_stratum": 10},
                                                     "colony": {"n_boot": 100, "time_order": {"*": ["giorno1", "giorno2"]}}})
    data.cfg.frames.session_dirs = _session_dirs(tmp_path, data, lambda f: "giorno1" if int(f[-7:-4]) % 2 else "giorno2")
    engine = ALEngine(data.cfg, data.emb)
    for _ in range(4):
        label_batch(engine, data.truth)
    out = engine.export_snapshot("s", "a@x.it")
    rows = read_csv(out / "predictions.csv")
    expected = {i: ("giorno1" if int(f[-7:-4]) % 2 else "giorno2") for i, f in data.frame_of.items()}
    assert {r["image_id"]: r["session"] for r in rows} == expected
    assert {r["dataset"] for r in rows if r["session"] == "giorno1"} == {"tr", "te"}      # the sessions span the folders
    meta = json.loads((out / "meta.json").read_text())
    assert meta["bees_per_session"] == dict(Counter(expected.values())) and meta["session_dirs_missing"] == []
    assert {f["session"] for f in read_csv(out / "frames.csv")} == {"giorno1", "giorno2"}

    engine.create_audit("s")
    label_audit(engine, data.truth, "a@x.it")
    label_audit(engine, data.truth, "b@x.it")
    report = build_report(out, data.cfg, store=engine.store)
    sessions = {g["group"]: g for g in report["estimates"]["groups"]["session"]["groups"]}
    assert set(sessions) == {"giorno1", "giorno2"}
    for name, g in sessions.items():                                                        # the estimate of each session is close to its truth
        truth = Counter(data.truth[r["image_id"]] for r in rows if r["session"] == name and r["status"] not in ("validation", "unknown", "no_prediction"))
        for label, stats in g["classes"].items():
            assert stats["share"] == pytest.approx(truth[label] / sum(truth.values()), abs=0.1), (name, label)
    assert [(t["from"], t["to"]) for t in report["indicators"]["trend"]] == [("giorno1", "giorno2")]
    assert [g["group"] for g in report["indicators"]["groups"]] == ["giorno1", "giorno2"]
