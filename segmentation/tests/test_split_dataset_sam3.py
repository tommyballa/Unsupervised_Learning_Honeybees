import csv
import os

import numpy as np
import pytest
from PIL import Image

import split_dataset_sam3 as sds


def make_photos(n=160, day="2026-09-12", start=0):
    """Come le foto vere: una panoramica affollata ogni otto, poi i primi piani."""
    return [{"name": f"DSC_{start + i:04d}.JPG", "day": day, "n_bees": 1200 if i % 8 == 0 else 200}
            for i in range(n)]


def test_assign_photos_is_balanced_and_reproducible():
    n_bees = [p["n_bees"] for p in make_photos()]
    splits = sds.assign_photos(n_bees, sds.FRACTIONS, np.random.default_rng(0), trials=5000)
    assert [splits.count(s) for s in ("train", "val", "test")] == [112, 24, 24]
    for split, fraction in sds.FRACTIONS.items():
        bees = sum(n for n, s in zip(n_bees, splits) if s == split)
        assert bees / sum(n_bees) == pytest.approx(fraction, abs=0.01)
    assert splits == sds.assign_photos(n_bees, sds.FRACTIONS, np.random.default_rng(0), trials=5000)


def test_plan_split_balances_every_day():
    photos = sds.plan_split(make_photos() + make_photos(n=80, day="2026-10-11", start=1000), trials=5000)
    for day, n in [("2026-09-12", 160), ("2026-10-11", 80)]:
        for split, fraction in sds.FRACTIONS.items():
            assert sum((p["day"], p["split"]) == (day, split) for p in photos) / n == pytest.approx(fraction, abs=0.02)


def test_plan_split_is_reproducible_and_seed_dependent():
    first = [p["split"] for p in sds.plan_split(make_photos(), trials=2000)]
    assert first == [p["split"] for p in sds.plan_split(list(reversed(make_photos())), trials=2000)][::-1]
    assert first != [p["split"] for p in sds.plan_split(make_photos(), seed=7, trials=2000)]


def make_dataset(tmp_path):
    source = tmp_path / "sam3"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for photo in make_photos() + make_photos(day="2026-09-26", start=1000):
        exif = Image.Exif()
        exif[306] = photo["day"].replace("-", ":") + " 18:00:00"
        Image.new("RGB", (8, 8)).save(source / "images" / photo["name"], exif=exif)
        label = source / "labels" / (os.path.splitext(photo["name"])[0] + ".txt")
        label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n" * (3 if photo["n_bees"] > 1000 else 1))
    return str(source)


def test_build_writes_a_disjoint_yolo_dataset(tmp_path):
    source = make_dataset(tmp_path)
    out = tmp_path / "split"
    sds.build(source, str(out))

    names = {}
    for split in sds.FRACTIONS:
        images = sorted(os.listdir(out / split / "images"))
        labels = sorted(os.listdir(out / split / "labels"))
        assert images and [os.path.splitext(n)[0] for n in images] == [os.path.splitext(n)[0] for n in labels]
        names[split] = set(images)
    assert sum(len(n) for n in names.values()) == len(set().union(*names.values())) == 320

    with open(out / "split.csv") as f:
        rows = list(csv.DictReader(f))
    assert {row["name"] for row in rows if row["split"] == "test"} == names["test"]
    assert {row["day"] for row in rows} == {"2026-09-12", "2026-09-26"}

    with pytest.raises(FileExistsError):
        sds.build(source, str(out))


def test_build_dry_run_writes_nothing(tmp_path):
    sds.build(make_dataset(tmp_path), str(tmp_path / "split"), dry_run=True)
    assert not (tmp_path / "split").exists()


def test_build_detects_a_label_without_its_photo(tmp_path):
    source = make_dataset(tmp_path)
    os.remove(os.path.join(source, "images", "DSC_0005.JPG"))
    with pytest.raises(ValueError):
        sds.build(source, str(tmp_path / "split"), dry_run=True)
