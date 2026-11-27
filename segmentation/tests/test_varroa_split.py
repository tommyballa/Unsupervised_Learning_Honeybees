import os

import pytest

pytest.importorskip("cv2")

from build_varroa_split import plan_counts, seed_for, split_files


def test_plan_counts_full_dataset():
    train, test, one, two = plan_counts(54182)
    assert (train, test) == (43346, 10836)
    assert (one, two) == (217, 108)
    assert one + two == round(test * 0.03)


@pytest.mark.parametrize("n", [0, 1, 7, 100, 2000, 54182])
def test_plan_counts_are_consistent(n):
    train, test, one, two = plan_counts(n)
    assert train + test == n
    assert min(train, test, one, two) >= 0
    assert one + two <= test


def test_split_is_disjoint_complete_and_reproducible():
    files = [f"/d/bee_{i:05d}.png" for i in range(1000)]
    train, test = split_files(files, seed=42)
    assert len(test) == 200 and len(train) == 800
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(files)
    assert split_files(list(reversed(files)), seed=42) == (train, test)
    assert split_files(files, seed=43)[1] != test


def test_seed_for_depends_on_name_not_directory():
    assert seed_for("/a/bee_1.png", 42) == seed_for("/b/bee_1.png", 42)
    assert seed_for("/a/bee_1.png", 42) != seed_for("/a/bee_2.png", 42)


def test_plan_counts_five_percent():
    train, test, one, two = plan_counts(127879, anomalous_fraction=0.05)
    assert (train, test) == (102303, 25576)
    assert one + two == round(25576 * 0.05)


def test_collect_crops_keeps_cluster_and_rejects_duplicates(tmp_path):
    from build_varroa_split import collect_crops

    for cluster, names in {"Giovane": ["a.png", "b.png"], "Fuco": ["c.png"]}.items():
        (tmp_path / cluster).mkdir()
        for n in names:
            (tmp_path / cluster / n).write_bytes(b"")
    found = collect_crops([str(tmp_path / "Giovane"), str(tmp_path / "Fuco") + "/"])
    assert {os.path.basename(p): c for p, c in found.items()} == {"a.png": "Giovane", "b.png": "Giovane", "c.png": "Fuco"}

    (tmp_path / "Malata").mkdir()
    (tmp_path / "Malata" / "a.png").write_bytes(b"")
    with pytest.raises(ValueError):
        collect_crops([str(tmp_path / "Giovane"), str(tmp_path / "Malata")])
