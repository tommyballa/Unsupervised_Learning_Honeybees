"""Per-batch model counts (scripts/batch_counts.py)."""
import importlib.util

import pytest

from bee_al.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("batch_counts", REPO_ROOT / "scripts" / "batch_counts.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def bee(session, frame, status, final="", propagation=""):
    return {"session": session, "frame": frame, "status": status, "final_label": final, "propagation_label": propagation}


def test_counts_per_batch_follow_time_order_and_keep_extras_apart():
    rows = [
        bee("b2", "f3", "predicted", "A"), bee("b2", "f3", "predicted", "B"),
        bee("b1", "f1", "labeled", "A"), bee("b1", "f1", "predicted", "A"), bee("b1", "f2", "predicted", "B"),
        bee("b1", "f2", "validation", propagation="A"),            # validation: counts with the predicted class
        bee("b1", "f2", "unassigned"), bee("b1", "f1", "unknown"), bee("b1", "f1", "no_prediction"),
    ]
    frames = [{"session": "b1", "frame": "f1"}, {"session": "b1", "frame": "f2"}, {"session": "b1", "frame": "vuota"},
              {"session": "b2", "frame": "f3"}]
    meta = {"config": {"colony": {"time_order": {"*": ["b2", "b1"]}}}}
    by_batch, per_photo, classes = bc.count(rows, frames, meta, n_boot=200)

    assert classes == ["A", "B"] and [b["batch"] for b in by_batch] == ["b2", "b1"]
    b1 = by_batch[1]
    assert b1["photos"] == 3 and b1["bees"] == 7 and b1["bees_in_classes"] == 4      # the empty photo still counts as a photo
    assert (b1["A"], b1["B"]) == (3, 1) and b1["A %"] == pytest.approx(75.0)
    assert b1[bc.UNASSIGNED] == b1[bc.SET_ASIDE] == b1[bc.NO_PREDICTION] == 1
    assert b1["A % lo"] <= 75.0 <= b1["A % hi"] and b1["A per photo"] == pytest.approx(1.0)
    assert by_batch[0]["A %"] == pytest.approx(50.0)
    assert len(per_photo) == 4 and sum(p["bees"] for p in per_photo) == len(rows)


def test_reject_classes_stay_out_and_profile_uses_only_demographic_classes():
    rows = [bee("b1", "f1", "predicted", "Giovane"), bee("b1", "f1", "predicted", "Adulte"),
            bee("b1", "f1", "predicted", "Adulte"), bee("b1", "f1", "predicted", "Adulte"),
            bee("b1", "f1", "predicted", "Malata"), bee("b1", "f1", "rejected", "Scarto"),
            bee("b1", "f1", "validation", propagation="Scarto")]
    by_batch, _, classes = bc.count(rows, [{"session": "b1", "frame": "f1"}], {}, n_boot=50,
                                    reject=("Scarto",), demographic=("Giovane", "Adulte"))
    b = by_batch[0]
    assert classes == ["Adulte", "Giovane", "Malata"] and b["Scarto"] == 2 and b["bees_in_classes"] == 5
    assert b["Malata %"] == pytest.approx(20.0) and "Malata % profile" not in b
    assert b["Adulte % profile"] == pytest.approx(75.0) and b["Giovane % profile"] == pytest.approx(25.0)
