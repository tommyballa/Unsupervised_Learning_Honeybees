from pathlib import Path

import numpy as np
import pytest

from bee_al.embeddings import extract_embeddings, load_embeddings, read_manifest, save_embeddings


class FakeEncoder:
    """Embedding deterministico dal nome del file; conta le immagini elaborate."""

    def __init__(self, fail_after=None, bad=()):
        self.seen = []
        self.fail_after = fail_after
        self.bad = set(bad)

    def __call__(self, paths):
        if self.fail_after is not None and len(self.seen) >= self.fail_after:
            raise KeyboardInterrupt
        self.seen.extend(p.name for p in paths)
        out = np.stack([np.random.default_rng(abs(hash(p.name)) % 2**32).normal(size=8) for p in paths])
        for i, p in enumerate(paths):
            if p.name in self.bad:
                out[i] = np.nan
        return out


def _paths(n):
    return [Path(f"/data/bee_{i:05d}_conf0.90.png") for i in range(n)]


def test_extract_saves_normalized_float32_with_ids_and_paths(tmp_path):
    paths = _paths(10)
    added = extract_embeddings(paths, tmp_path, FakeEncoder(), meta={"model": "fake"}, shard_size=4)
    emb = load_embeddings(tmp_path)
    assert added == 10
    assert emb.ids == [p.name for p in paths]
    assert emb.paths == [str(p) for p in paths]
    assert emb.X.dtype == np.float32 and emb.X.shape == (10, 8)
    np.testing.assert_allclose(np.linalg.norm(emb.X, axis=1), 1.0, atol=1e-5)
    assert len(read_manifest(tmp_path)["shards"]) == 3


def test_extract_resumes_after_interruption(tmp_path):
    paths = _paths(10)
    enc = FakeEncoder(fail_after=8)  # si interrompe al terzo shard
    with pytest.raises(KeyboardInterrupt):
        extract_embeddings(paths, tmp_path, enc, meta={"model": "fake"}, shard_size=4)
    assert len(load_embeddings(tmp_path)) == 8

    enc2 = FakeEncoder()
    assert extract_embeddings(paths, tmp_path, enc2, meta={"model": "fake"}, shard_size=4) == 2
    assert enc2.seen == [p.name for p in paths[8:]]  # le prime 8 non vengono rifatte
    assert load_embeddings(tmp_path).ids == [p.name for p in paths]
    assert extract_embeddings(paths, tmp_path, FakeEncoder(), meta={"model": "fake"}, shard_size=4) == 0


def test_unreadable_images_are_skipped_and_not_retried(tmp_path):
    paths = _paths(6)
    extract_embeddings(paths, tmp_path, FakeEncoder(bad=[paths[2].name]), meta={}, shard_size=4)
    emb = load_embeddings(tmp_path)
    assert paths[2].name not in emb.ids and len(emb) == 5
    assert read_manifest(tmp_path)["failed"] == [paths[2].name]
    enc = FakeEncoder()
    extract_embeddings(paths, tmp_path, enc, meta={}, shard_size=4)
    assert enc.seen == []


def test_different_model_in_same_dir_is_refused(tmp_path):
    extract_embeddings(_paths(3), tmp_path, FakeEncoder(), meta={"model": "a"})
    with pytest.raises(ValueError, match="estratti con"):
        extract_embeddings(_paths(3), tmp_path, FakeEncoder(), meta={"model": "b"})


def test_save_load_roundtrip_and_missing_dir(tmp_path):
    X = np.random.default_rng(0).normal(size=(5, 4))
    save_embeddings(tmp_path / "e", ["a", "b", "c", "d", "e"], ["pa"] * 5, X, meta={"model": "m"})
    emb = load_embeddings(tmp_path / "e")
    assert emb.ids == ["a", "b", "c", "d", "e"] and emb.meta == {"model": "m"}
    np.testing.assert_allclose(np.linalg.norm(emb.X, axis=1), 1.0, atol=1e-5)
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path / "nope")
