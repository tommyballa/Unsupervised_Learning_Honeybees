"""Filtro di lunghezza e orientamento testa in alto di dataset_cleaning (senza GPU né immagini reali)."""
import sys
import types

import cv2
import numpy as np
import pytest

import dataset_cleaning as dc
from bee_orientation import head_up_in_frame


def synthetic_bee(length, width=None, angle=30, size=None):
    """Crop con un'ape ellittica ruotata: metà alta (nel crop) chiara, metà bassa scura."""
    width = width or length // 2
    size = size or int(length * 1.6)
    img = np.full((size, size, 3), 40, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    center = (size // 2, size // 2)
    cv2.ellipse(mask, center, (length // 2, width // 2), angle, 0, 360, 255, -1)
    rng = np.random.default_rng(0)
    img[mask > 0] = rng.integers(60, 200, (int((mask > 0).sum()), 3), dtype=np.uint8)  # texture, per il filtro blur
    return img, mask


def test_short_bees_are_discarded():
    img, mask = synthetic_bee(80)
    assert dc.clean_bee(img, mask, True).reason == "short"


def test_rotate_frame_180_keeps_matrix_consistent():
    img, mask = synthetic_bee(200)
    result = dc.clean_bee(img, mask, True)
    assert result.reason is None

    rotated = dc.rotate_frame_180(result)
    size = (dc.TARGET_SIZE, dc.TARGET_SIZE)
    # Una maschera esterna (es. varroa) portata nel frame con la nuova affine
    # deve coincidere con quella portata con la vecchia e poi ruotata
    other = np.zeros_like(mask)
    cv2.circle(other, (150, 120), 12, 255, -1)
    via_new = cv2.warpAffine(other, rotated.matrix, size, flags=cv2.INTER_NEAREST)
    via_old = cv2.rotate(cv2.warpAffine(other, result.matrix, size, flags=cv2.INTER_NEAREST), cv2.ROTATE_180)
    assert np.array_equal(via_new, via_old)
    assert np.array_equal(rotated.image, cv2.rotate(result.image, cv2.ROTATE_180))
    assert np.array_equal(rotated.mask, cv2.rotate(result.mask, cv2.ROTATE_180))


def test_head_up_in_frame_follows_rotation():
    img, mask = synthetic_bee(200)
    result = dc.clean_bee(img, mask, True)
    head_dir = np.linalg.solve(result.matrix[:, :2], [0.0, -1.0])  # testa in alto nel frame
    assert head_up_in_frame(result.matrix, head_dir)
    assert not head_up_in_frame(dc.rotate_frame_180(result).matrix, head_dir)


class FakeOrienter:
    """Testa = metà più chiara dell'immagine."""
    weights_id = "fake"
    device = "cpu"

    def __init__(self, **kwargs):
        pass

    def scores(self, images):
        h = dc.TARGET_SIZE // 2
        return np.array([float(im[:h].mean()) - float(im[h:].mean()) for im in images])


@pytest.fixture
def fake_orienter(monkeypatch):
    monkeypatch.setitem(sys.modules, "bee_orientation", types.SimpleNamespace(HeadOrienter=FakeOrienter))


def test_orient_saved_bees_is_idempotent(tmp_path, fake_orienter):
    s = dc.TARGET_SIZE
    head_up = np.zeros((s, s, 3), np.uint8)
    head_up[: s // 2] = 200
    files = {"up.png": head_up, "down.png": cv2.rotate(head_up, cv2.ROTATE_180)}
    for name, img in files.items():
        cv2.imwrite(str(tmp_path / name), img)
    manifest = dc.new_cleaning_manifest()
    manifest["processed_files"] = {"up.png": "accepted", "down.png": "accepted", "gone.png": "discarded"}

    dc.orient_saved_bees(str(tmp_path), manifest, workers=2)
    for name in files:
        assert np.array_equal(cv2.imread(str(tmp_path / name)), head_up)
    assert manifest["orientation"]["scores"]["down.png"] < 0 < manifest["orientation"]["scores"]["up.png"]
    assert "gone.png" not in manifest["orientation"]["scores"]

    # Interruzione simulata: il registro si perde ma le api sono già ruotate, non vanno ricapovolte
    manifest["orientation"]["scores"].clear()
    dc.orient_saved_bees(str(tmp_path), manifest, workers=2)
    for name in files:
        assert np.array_equal(cv2.imread(str(tmp_path / name)), head_up)
