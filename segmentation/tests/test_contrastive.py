"""Contrastive fine-tuning: loss, batch composition, training on a tiny model, subprocess."""
import json
import sys
from collections import Counter

import numpy as np
import pytest

from bee_al.config import config_from_dict
from bee_al.contrastive import ContrastiveTrainer, PrecomputedExtractor, plan_batches
from bee_al.embeddings import load_embeddings, save_embeddings
from bee_al.retrain import TrainingData, make_trainer


def _job(tmp_path, n_images=0, **kw):
    job = {
        "model": "tiny", "image_size": 32, "seed": 0, "embed_batch_size": 8, "shard_size": 100,
        "unfreeze_blocks": 1, "steps": 4, "batch_size": 8, "labeled_fraction": 0.5,
        "lr_backbone": 1e-2, "lr_head": 1e-2, "temperature": 0.1, "projection_dim": 8,
        "num_workers": 0, "device": "cpu", "out_dir": str(tmp_path / "work"),
        "labeled": [], "pseudo": [], "unlabeled": [], "all_paths": [],
    }
    job.update(kw)
    return job


def test_plan_batches_mixes_labeled_pseudo_and_unlabeled(tmp_path):
    job = _job(
        tmp_path, steps=50, batch_size=20,
        labeled=[[f"a{i}", "A"] for i in range(30)] + [["b0", "B"]],          # B has a single example
        pseudo=[[f"pa{i}", "A"] for i in range(10)] + [[f"pc{i}", "C"] for i in range(5)],
        unlabeled=[f"u{i}" for i in range(200)],
    )
    plan, class_id = plan_batches(job, np.random.default_rng(0))
    assert len(plan) == 50 * 20 and class_id == {"A": 0, "B": 1, "C": 2}
    groups = np.array([g for _, g in plan]).reshape(50, 20)
    assert ((groups >= 0).sum(axis=1) == 10).all()                            # half the batch is supervised
    unl = groups[groups < 0]
    assert len(unl) == len(set(unl.tolist()))                                 # each unlabelled bee has its own group
    human = Counter(g for p, g in plan if p.startswith(("a", "b")))
    assert human[1] > 0.3 * (human[0] + human[1])                             # balanced classes: the rare one does not vanish
    assert any(p.startswith("pc") for p, _ in plan)                           # pseudo-labels enter the batch

    only_labeled, _ = plan_batches(_job(tmp_path, labeled=[["a", "A"], ["b", "B"]]), np.random.default_rng(0))
    assert len(only_labeled) == 4 * 8 and all(g >= 0 for _, g in only_labeled)


def test_supcon_loss():
    torch = pytest.importorskip("torch")
    from bee_al.contrastive import supcon_loss

    torch.manual_seed(0)
    a, b = torch.nn.functional.normalize(torch.randn(2, 16), dim=1)
    groups = torch.tensor([0, 0, 1, 1])
    aligned = torch.stack([a, a, b, b])                 # identical positives: minimal loss
    mixed = torch.stack([a, b, a, b])                   # swapped positives: high loss
    assert supcon_loss(aligned, groups) < 0.1 < supcon_loss(mixed, groups)

    z = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    simclr = supcon_loss(z, torch.tensor([-1, -2, -3, -1, -2, -3]), 0.5)      # only the two views: NT-Xent
    sim = (z @ z.T / 0.5).masked_fill(torch.eye(6, dtype=torch.bool), float("-inf"))
    expected = torch.nn.functional.cross_entropy(sim, torch.tensor([3, 4, 5, 0, 1, 2]))
    assert torch.allclose(simclr, expected, atol=1e-5)

    lonely = supcon_loss(z, torch.tensor([0, 0, 1, 2, 3, 4]))                 # anchors without positives do not count
    assert torch.isfinite(lonely)
    z.requires_grad_(True)
    supcon_loss(z, torch.tensor([0, 0, 1, 1, 2, 2])).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_train_contrastive_end_to_end_on_tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from bee_al.contrastive import train_contrastive

    rng = np.random.default_rng(0)
    paths = []
    for i in range(16):                                  # two "classes" of images: reddish and bluish
        img = rng.integers(0, 60, size=(40, 40, 3), dtype=np.uint8)
        img[..., 0 if i % 2 == 0 else 2] += 150
        p = tmp_path / f"bee_{i:02d}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, stride=2)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, x):
            x = self.conv(x).mean(dim=(2, 3))
            for block in self.blocks:
                x = torch.relu(block(x))
            return self.norm(x)

    torch.manual_seed(0)
    model = Tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    job = _job(
        tmp_path, steps=6,
        labeled=[[p, "Red" if i % 2 == 0 else "Blue"] for i, p in enumerate(paths[:6])],
        pseudo=[[paths[6], "Red"]], unlabeled=paths[6:], all_paths=paths,
    )
    summary = train_contrastive(job, backbone=model, log=lambda s: None)

    out = tmp_path / "work"
    assert summary["n_labeled"] == 6 and summary["n_pseudo"] == 1 and len(summary["losses"]) == 6
    assert np.isfinite(summary["losses"]).all()
    assert json.loads((out / "progress.json").read_text())["phase"] == "done"
    emb = load_embeddings(out / "embeddings")              # all embeddings recomputed
    assert emb.paths == paths and emb.X.shape == (16, 8)
    np.testing.assert_allclose(np.linalg.norm(emb.X, axis=1), 1.0, atol=1e-5)

    after = torch.load(out / "backbone.pt")
    assert torch.equal(after["conv.weight"], before["conv.weight"])          # frozen part untouched
    assert torch.equal(after["blocks.0.weight"], before["blocks.0.weight"])
    assert not torch.equal(after["blocks.1.weight"], before["blocks.1.weight"])  # last block trained


FAKE_SCRIPT = """
import json, sys, numpy as np
sys.path.insert(0, {root!r})
from pathlib import Path
from bee_al.embeddings import save_embeddings
job = json.load(open(sys.argv[sys.argv.index("--job") + 1]))
out = Path(job["out_dir"])
if job["steps"] == 13:
    print("CUDA out of memory"); sys.exit(3)
paths = job["all_paths"][::-1]                             # order different from the requested one
X = np.array([[float(Path(p).stem.split("_")[1]), 1.0] for p in paths])
save_embeddings(out / "embeddings", [Path(p).name for p in paths], paths, X)
(out / "backbone.pt").write_text("weights")
(out / "train_log.json").write_text(json.dumps({{"n_pseudo": len(job["pseudo"])}}))
(out / "progress.json").write_text(json.dumps({{"phase": "done", "step": 1, "total": 1, "loss": 0.5, "elapsed_s": 1}}))
"""


def test_contrastive_trainer_runs_a_subprocess_and_reads_its_output(tmp_path):
    from bee_al.config import REPO_ROOT

    cfg = config_from_dict({}, root=tmp_path)
    assert isinstance(make_trainer(cfg), ContrastiveTrainer)
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_SCRIPT.format(root=str(REPO_ROOT)))
    trainer = ContrastiveTrainer(cfg, command=[sys.executable, str(script)])
    assert trainer.progress() is None

    paths = [f"/d/bee_{i}.png" for i in range(6)]
    labeled = TrainingData(["bee_0.png", "bee_1.png"], paths[:2], np.zeros((2, 2)), ["A", "B"])
    unlabeled = TrainingData([f"bee_{i}.png" for i in range(2, 6)], paths[2:], np.zeros((4, 2)),
                             pseudo_labels=["A", None, None, "B"])
    job = trainer.build_job(labeled, unlabeled)
    assert job["labeled"] == [[paths[0], "A"], [paths[1], "B"]]
    assert job["pseudo"] == [[paths[2], "A"], [paths[5], "B"]] and job["unlabeled"] == paths[2:]
    assert job["steps"] == cfg.retrain.steps and job["model"] == cfg.extractor.model

    extractor = trainer.retrain(labeled, unlabeled)
    assert isinstance(extractor, PrecomputedExtractor) and trainer.progress()["phase"] == "done"
    X = extractor.embed(paths)                               # rows in the requested order, not the file's
    np.testing.assert_allclose(X[:, 0] / X[:, 1], [0, 1, 2, 3, 4, 5], atol=1e-5)
    with pytest.raises(RuntimeError, match="without embedding"):
        extractor.embed(paths + ["/d/bee_99.png"])

    version = tmp_path / "v001"
    version.mkdir()
    extractor.save(version)
    assert (version / "backbone.pt").read_text() == "weights" and (version / "train_log.json").exists()
    assert not trainer.work_dir.exists()                     # work folder cleaned up

    cfg.retrain.steps = 13                                   # the process fails: the error carries the log tail
    with pytest.raises(RuntimeError, match="exit code 3.*CUDA out of memory"):
        trainer.retrain(labeled, unlabeled)
