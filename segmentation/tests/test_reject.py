"""Reject classes across the whole flow: clusters, snapshot, audit, estimates, report, status page."""
import pytest

from bee_al import hub as al_hub
from bee_al.audit import load_audit_set
from bee_al.config import DEFAULT_CONFIG_PATH
from bee_al.engine import ALEngine
from bee_al.report import build_report
from bee_al.snapshot import read_csv

from conftest import build_colony_dataset
from test_audit import label_audit
from test_engine import label_batch

REJECT = "Class_1"


@pytest.fixture
def flow(tmp_path):
    data = build_colony_dataset(tmp_path, extra_cfg={
        "labels": {"reject_classes": ["class_1"]},                       # written in lower case: goes through normalisation
        "frames": {"frame_pattern": r"(?P<colony>hive[A-Z])_(?P<session>s\d)"},
        "audit": {"per_stratum": 12}, "colony": {"n_boot": 100},
    })
    engine = ALEngine(data.cfg, data.emb)
    for _ in range(4):
        label_batch(engine, data.truth)
    engine.export_snapshot("s", "a@x.it")
    engine.create_audit("s")
    data.engine, data.folder = engine, data.cfg.results_dir / "snapshots" / "s"
    return data


def test_a_reject_bee_through_the_whole_flow(flow):
    engine = flow.engine
    assert engine.reject_classes() == {REJECT} and REJECT in engine.bank.classes     # stays in the kNN
    assert REJECT not in engine.final_clusters()                                     # but not in the final clusters

    rows = {r["image_id"]: r for r in read_csv(flow.folder / "predictions.csv")}
    by_hand = [r for r in rows.values() if r["status"] == "rejected" and r["source"] == "human"]
    by_model = [r for r in rows.values() if r["status"] == "rejected" and r["source"] == "predicted"]
    assert by_hand and by_model and all(r["final_label"] == REJECT for r in by_hand + by_model)
    in_clusters = {m["image_id"] for members in engine.final_clusters().values() for m in members}
    assert not in_clusters & {r["image_id"] for r in by_hand + by_model}

    # audit: one stratum for the predicted reject class, containing only bees rejected by the model
    audit = load_audit_set(flow.folder)
    stratum = next(s for s in audit["strata"] if s["id"] == f"{REJECT}|reject")
    assert stratum["kind"] == "reject" and stratum["N_h"] == len(by_model) and stratum["conf_bin"] == ""
    sampled = [b["image_id"] for b in audit["bees"] if b["stratum"] == stratum["id"]]
    assert sampled and all(rows[i]["source"] == "predicted" for i in sampled)
    assert not {r["image_id"] for r in by_hand} & {b["image_id"] for b in audit["bees"]}

    label_audit(engine, flow.truth, "a@x.it")
    label_audit(engine, flow.truth, "b@x.it")
    report = build_report(flow.folder, flow.cfg, store=engine.store)
    est = report["estimates"]
    prevalence = {p["class"]: p for p in est["prevalence"]}
    assert prevalence[REJECT]["reject"] and prevalence[REJECT]["share"] is None and prevalence[REJECT]["n_hat"] > len(by_hand)
    assert prevalence[REJECT]["human"] == len(by_hand)                                # hand-rejected bees count exactly
    valid = [p for p in est["prevalence"] if not p["reject"]]
    assert sum(p["share"] for p in valid) == pytest.approx(1.0)                       # the denominator does not contain the rejects
    assert est["totals"]["valid_hat"] == pytest.approx(sum(p["n_hat"] for p in valid))
    assert est["totals"]["reject_hat"] == pytest.approx(prevalence[REJECT]["n_hat"])

    # valid bees rejected by mistake: the non-reject answers in the reject stratum
    s = next(x for x in est["strata"] if x["id"] == stratum["id"])
    wrongly = sum(n for label, n in s["answers"].items() if label != REJECT)
    assert s["precision"] == pytest.approx(1 - wrongly / s["n_labeled"])

    # report: estimated rejects per group next to the automatic-cleaning discards
    for g in report["indicators"]["groups"]:
        assert 0 < g["quality"]["reject_share"] < 1 and g["quality"]["cleaning_discard_rate"] == pytest.approx(1 / 7, abs=0.02)
        assert g["quality"]["reject_share_ci"][0] <= g["quality"]["reject_share"] <= g["quality"]["reject_share_ci"][1]
        assert REJECT not in g["I1_demographics"] and all(REJECT not in step["deltas"] for step in report["indicators"]["trend"])
    md = (flow.folder / "report" / "report.md").read_text()
    assert "Estimated reject share" in md and "Discarded by cleaning" in md and f"{REJECT} ≈" in md
    frames = {r["group"]: r for r in read_csv(flow.folder / "report" / "groups.csv") if r["level"] == "frame"}
    assert len(frames) == 7 and all(0 <= float(r["reject_share"]) <= 1 for r in frames.values())   # one frame has no bees


def test_status_page_counts_reject_classes(flow, hub_app, monkeypatch, tmp_path):
    engine = flow.engine
    counts = engine.reject_counts()
    rows = read_csv(flow.folder / "predictions.csv")
    assert counts == engine.status()["reject_counts"] and set(counts) == {REJECT}
    assert counts[REJECT]["labeled"] == sum(r["status"] == "rejected" and r["source"] == "human" for r in rows)
    assert counts[REJECT]["predicted"] == sum(r["status"] == "rejected" and r["source"] == "predicted" for r in rows)

    monkeypatch.setattr(al_hub.STATE, "engine", engine)
    monkeypatch.chdir(tmp_path)
    c = hub_app.app.test_client()
    c.post("/", data={"email": "t@x.it"})
    html = c.get("/al-status").get_data(as_text=True)
    assert "Reject classes" in html and "Assigned by the model" in html and "label what it is" in html
    assert f"<td>{REJECT}</td><td>{counts[REJECT]['labeled']}</td><td>{counts[REJECT]['predicted']}</td>" in html

    engine.cfg.labels.reject_classes = []
    engine.refresh(reload_config=False)
    assert engine.reject_counts() == {} and "Reject classes" not in c.get("/al-status").get_data(as_text=True)


def test_config_documents_the_rule_for_annotators():
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert '["Blurry", "Bad Seg", "Multiple Bees", "No Bee"]' in text
    assert "if the defect prevents judging the phenotype, it is a reject" in text
