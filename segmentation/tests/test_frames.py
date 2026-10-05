"""Collegamento ape -> frame dai manifest dell'estrazione e della pulizia."""
import json

import pytest

from bee_al.config import config_from_dict
from bee_al.frames import (
    dataset_folders,
    dataset_of,
    discards_per_frame,
    frame_groups,
    load_frame_map,
    load_frame_maps,
)


def _manifest(tmp_path, frames, next_bee_id=None):
    path = tmp_path / "_extraction_manifest.json"
    total = sum(frames.values())
    path.write_text(json.dumps({"processed_sources": {f: {"bees": n} for f, n in frames.items()},
                                "next_bee_id": total if next_bee_id is None else next_bee_id}))
    return path


def test_intervals_follow_the_order_of_the_file(tmp_path):
    fmap = load_frame_map(_manifest(tmp_path, {"b.JPG": 3, "a.JPG": 0, "c.JPG": 2, "z.JPG": 4}))
    assert fmap.reliable
    assert list(zip(fmap.frames, fmap.starts, fmap.ends)) == [("b.JPG", 0, 3), ("a.JPG", 3, 3), ("c.JPG", 3, 5), ("z.JPG", 5, 9)]
    assert fmap.raw_bees_per_frame() == {"b.JPG": 3, "a.JPG": 0, "c.JPG": 2, "z.JPG": 4}

    name = "bee_{:05d}_conf0.87.png".format
    assert [fmap.frame_of(name(i)) for i in (0, 2, 3, 4, 5, 8)] == ["b.JPG", "b.JPG", "c.JPG", "c.JPG", "z.JPG", "z.JPG"]
    assert fmap.frame_of(name(9)) is None                        # primo id fuori dall'ultimo intervallo
    assert fmap.frame_of("bee_123456_conf0.50.png") is None
    assert fmap.frame_of("ape_00001.png") is None and fmap.frame_of("bee_x_conf.png") is None
    assert fmap.frame_of("/some/dir/bee_00004_conf0.90.png") == "c.JPG"


def test_count_mismatch_makes_the_map_unreliable(tmp_path):
    with pytest.warns(UserWarning, match="next_bee_id"):
        fmap = load_frame_map(_manifest(tmp_path, {"a.JPG": 3, "b.JPG": 2}, next_bee_id=7))
    assert fmap.reliable is False
    assert fmap.frame_of("bee_00001_conf0.90.png") is None       # non si tira a indovinare
    assert fmap.raw_bees_per_frame() == {"a.JPG": 3, "b.JPG": 2}


def test_cleaned_bees_are_a_subset_with_missing_ids(colony):
    maps = load_frame_maps(colony.cfg)
    assert set(maps) == {"tr", "te"} and all(m.reliable for m in maps.values())
    for image_id, frame in colony.frame_of.items():
        dataset = dataset_of(image_id, colony.cfg)
        assert maps[dataset].frame_of(image_id.split("__", 1)[1]) == frame
    assert not any(i.endswith("bee_00000_conf0.90.png") for i in colony.frame_of)   # scartata dalla pulizia
    assert maps["tr"].frame_of("bee_00000_conf0.90.png") == "hiveA_s1_001.JPG"      # ma il frame si ricava lo stesso


def test_each_folder_uses_its_own_manifest(colony):
    maps = load_frame_maps(colony.cfg)
    same_file = "bee_00010_conf0.90.png"                          # la numerazione riparte da 0 in ogni cartella
    assert maps["tr"].frame_of(same_file) == "hiveA_s1_001.JPG"
    assert maps["te"].frame_of(same_file) == "hiveA_s2_006.JPG"
    assert dataset_of(f"te__{same_file}", colony.cfg) == "te"
    assert set(dataset_folders(colony.cfg)) == {"tr", "te"}


def test_discards_per_frame(colony, tmp_path):
    fmap = load_frame_maps(colony.cfg)["tr"]
    discards = discards_per_frame(tmp_path / "tr" / "cleaned_bees" / "_cleaning_manifest.json", fmap)
    first = discards["hiveA_s1_001.JPG"]                          # id 0..59: scartati 0, 7, ..., 56
    assert first["accepted"] == 51 and first["discarded"] == 9
    assert first["reasons"] == {"border": 5, "blur": 4}
    assert "hiveA_s1_002.JPG" not in discards                     # frame senza api
    assert sum(d["accepted"] + d["discarded"] for d in discards.values()) == 220
    assert discards_per_frame(tmp_path / "non_esiste.json", fmap) == {}


def test_manifest_override_and_missing_manifest(colony, tmp_path):
    other = tmp_path / "altrove.json"
    other.write_text(json.dumps({"processed_sources": {"x.JPG": {"bees": 220}}, "next_bee_id": 220}))
    colony.cfg.frames.manifests = {"tr": str(other)}
    (tmp_path / "te" / "raw_bees" / "_extraction_manifest.json").unlink()
    maps = load_frame_maps(colony.cfg)
    assert maps["tr"].frames == ["x.JPG"] and maps["te"] is None

    single = config_from_dict({"paths": {"dataset_dir": "tr/cleaned_bees"}}, root=tmp_path)
    assert list(dataset_folders(single)) == ["tr"] and dataset_of("bee_00001_conf0.90.png", single) == "tr"


def test_session_and_colony_from_the_frame_name():
    pattern = r"(?P<colony>hive[A-Z])_(?P<session>s\d)"
    assert frame_groups("hiveA_s2_003.JPG", "tr", pattern) == ("s2", "hiveA")
    assert frame_groups("DSC_5365.JPG", "tr", pattern) == ("tr", "")       # non corrisponde: nessuna invenzione
    assert frame_groups("hiveA_s2_003.JPG", "tr", "") == ("tr", "")
    assert frame_groups("hiveA_s2_003.JPG", "tr", r"(?P<session>s\d)") == ("s2", "")
    assert frame_groups("", "tr", pattern) == ("tr", "")

    assert config_from_dict({"frames": {"frame_pattern": pattern}}).frames.frame_pattern == pattern
    for bad in ("(", r"(?P<hive>\d+)", r"\d+"):
        with pytest.raises(ValueError):
            config_from_dict({"frames": {"frame_pattern": bad}})


def _session_dirs(tmp_path, colony, split):
    """Copia finta delle cartelle dei frame originali: ``split`` dice a quale sessione va ogni frame."""
    for name, frames in colony.folders.items():
        for frame in frames:
            folder = tmp_path / "orig" / split(frame) / "favi"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / frame).write_bytes(b"jpg")
    return {s: f"orig/{s}" for s in sorted({split(f) for frames in colony.folders.values() for f in frames})}


def test_session_from_the_folder_that_holds_the_frame(colony, tmp_path):
    from bee_al.frames import FrameGrouper

    # frame della stessa cartella di dataset finiscono in sessioni diverse, come train/test rispetto a batch_3/batch_4
    colony.cfg.frames.session_dirs = _session_dirs(tmp_path, colony, lambda f: "giorno1" if int(f[-7:-4]) % 2 else "giorno2")
    colony.cfg.frames.colony = "alveare_1"
    grouper = FrameGrouper(colony.cfg)
    assert grouper.groups("hiveA_s1_001.JPG", "tr") == ("giorno1", "alveare_1")
    assert grouper.groups("hiveA_s1_002.JPG", "tr") == ("giorno2", "alveare_1")
    assert grouper.groups("hiveA_s2_006.JPG", "te") == ("giorno2", "alveare_1")
    assert grouper.groups("mai_visto.JPG", "tr") == ("tr", "alveare_1")       # non trovato: resta la cartella di dataset
    assert grouper.groups("", "te") == ("te", "alveare_1")

    colony.cfg.frames.frame_pattern = r"(?P<colony>hive[A-Z])_(?P<session>s\d)"   # la regex, se corrisponde, vince
    assert FrameGrouper(colony.cfg).groups("hiveA_s1_001.JPG", "tr") == ("s1", "hiveA")
    assert FrameGrouper(colony.cfg).groups("DSC_1.JPG", "tr") == ("tr", "alveare_1")


def test_ambiguous_or_missing_session_dirs(colony, tmp_path):
    from bee_al.frames import FrameGrouper

    colony.cfg.frames.session_dirs = _session_dirs(tmp_path, colony, lambda f: "giorno1")
    (tmp_path / "orig" / "giorno2").mkdir()
    (tmp_path / "orig" / "giorno2" / "hiveA_s1_001.JPG").write_bytes(b"jpg")          # stesso nome in due sessioni
    colony.cfg.frames.session_dirs.update({"giorno2": "orig/giorno2", "giorno3": "orig/non_esiste"})
    with pytest.warns(UserWarning, match="più cartelle di sessione"):
        grouper = FrameGrouper(colony.cfg)
    assert grouper.groups("hiveA_s1_001.JPG", "tr") == ("tr", "")                    # ambiguo: nessuna sessione inventata
    assert grouper.groups("hiveA_s2_003.JPG", "tr") == ("giorno1", "")
    assert grouper.ambiguous == {"hiveA_s1_001.JPG"} and grouper.missing_dirs == [str(tmp_path / "orig" / "non_esiste")]


def test_default_config_maps_batch_3_and_batch_4():
    from bee_al.config import load_config

    cfg = load_config()
    assert list(cfg.frames.session_dirs) == ["batch_3", "batch_4"]
    assert cfg.colony.time_order == {"*": ["batch_3", "batch_4"]}
