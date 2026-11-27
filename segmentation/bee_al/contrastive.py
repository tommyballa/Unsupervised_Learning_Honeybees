"""Heavy re-train: contrastive fine-tuning of the extractor.

A single loss (generalised SupCon) on mixed batches:
- hand-labelled bees and high-confidence pseudo-labels: bees of the same class
  are positives of each other (supervised contrastive);
- unlabelled bees: the only positive is the other augmented view of the same bee
  (SimCLR), i.e. consistency across augmentations.

Pseudo-labels come from the current predictions at each job, so they are recomputed
from scratch every round. The weights also restart every time from the pre-trained model,
so that errors from previous rounds do not accumulate.

Training runs in a separate process (``scripts/retrain_contrastive.py``) launched
by ``ContrastiveTrainer``: the hub does not need torch and is not blocked.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import ALConfig, REPO_ROOT
from .embeddings import extract_embeddings, load_embeddings
from .retrain import FeatureExtractorTrainer, TrainingData

SCRIPT = REPO_ROOT / "scripts" / "retrain_contrastive.py"
WEIGHTS_NAME = "backbone.pt"


# --------------------------------------------------------------------------- #
# Hub side: launching the process and reading the result (no torch here)
# --------------------------------------------------------------------------- #

class PrecomputedExtractor:
    """Extractor whose work is already on disk: recomputed embeddings and weights in the work folder."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)

    def embed(self, paths: Sequence[str]) -> np.ndarray:
        emb = load_embeddings(self.work_dir / "embeddings")
        row = {p: i for i, p in enumerate(emb.paths)}
        missing = [p for p in paths if p not in row]
        if missing:
            raise RuntimeError(f"{len(missing)} images without embedding after the re-train (e.g. {missing[0]})")
        return emb.X[[row[p] for p in paths]]

    def save(self, out_dir: Path) -> None:
        for name in (WEIGHTS_NAME, "train_log.json", "train.log"):
            if (self.work_dir / name).exists():
                shutil.copy2(self.work_dir / name, Path(out_dir) / name)
        shutil.rmtree(self.work_dir, ignore_errors=True)


class ContrastiveTrainer(FeatureExtractorTrainer):
    """Launch the contrastive fine-tuning in a subprocess and wait for it to finish."""

    def __init__(self, cfg: ALConfig, command: Optional[Sequence[str]] = None) -> None:
        self.cfg = cfg
        self.work_dir = cfg.models_dir / "_work"
        python = cfg.resolve(cfg.retrain.python)
        self.command = list(command) if command else [str(python) if python.exists() else sys.executable, str(SCRIPT)]

    def build_job(self, labeled: TrainingData, unlabeled: TrainingData) -> dict[str, Any]:
        r, ex = self.cfg.retrain, self.cfg.extractor
        pseudo = [(p, lbl) for p, lbl in zip(unlabeled.paths, unlabeled.pseudo_labels) if lbl is not None]
        return {
            "model": ex.model, "image_size": ex.image_size, "seed": self.cfg.seed,
            "embed_batch_size": ex.batch_size, "shard_size": ex.shard_size,
            "unfreeze_blocks": r.unfreeze_blocks, "steps": r.steps, "batch_size": r.batch_size,
            "labeled_fraction": r.labeled_fraction, "lr_backbone": r.lr_backbone, "lr_head": r.lr_head,
            "temperature": r.temperature, "projection_dim": r.projection_dim,
            "num_workers": r.num_workers, "device": r.device,
            "labeled": [[p, lbl] for p, lbl in zip(labeled.paths, labeled.labels)],
            "pseudo": [[p, lbl] for p, lbl in pseudo],
            "unlabeled": list(unlabeled.paths),
            "all_paths": labeled.paths + unlabeled.paths,
            "all_ids": labeled.ids + unlabeled.ids,
            "out_dir": str(self.work_dir),
        }

    def retrain(self, labeled: TrainingData, unlabeled: TrainingData) -> PrecomputedExtractor:
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        job_path = self.work_dir / "job.json"
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(self.build_job(labeled, unlabeled), f)
        log_path = self.work_dir / "train.log"
        with open(log_path, "w", encoding="utf-8") as log:
            code = subprocess.run([*self.command, "--job", str(job_path)], stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            tail = " | ".join(log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:])
            raise RuntimeError(f"re-train failed (exit code {code}): {tail}")
        return PrecomputedExtractor(self.work_dir)

    def progress(self) -> Optional[dict[str, Any]]:
        """Progress written by the subprocess (phase, step, loss), if available."""
        try:
            with open(self.work_dir / "progress.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


# --------------------------------------------------------------------------- #
# Training side (torch imported only in here)
# --------------------------------------------------------------------------- #

def supcon_loss(z, groups, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al., 2020) on already-normalised embeddings.

    ``groups[i] == groups[j]`` makes ``i`` and ``j`` positives. Giving each unlabelled
    bee a group of its own (shared only by its two views) turns the same formula
    into SimCLR's NT-Xent.
    """
    import torch

    z = z.float()
    sim = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (groups[:, None] == groups[None, :]) & ~eye
    n_pos = positives.sum(dim=1)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0)).sum(dim=1) / n_pos.clamp(min=1)
    return per_anchor[n_pos > 0].mean()


def build_train_transform(image_size: int = 224):
    """Augmentation for the two views.

    Bees have no natural orientation, so rotations and flips are unrestricted.
    Colour is barely touched (only brightness and contrast): it can be a health signal.
    """
    import torchvision.transforms as T

    from .embeddings import IMAGENET_MEAN, IMAGENET_STD

    return T.Compose([
        T.RandomRotation(180),
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def plan_batches(job: dict[str, Any], rng: np.random.Generator) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """Sequence of (path, group) for all steps, plus the class -> group map.

    In each batch one part comes from labelled bees and pseudo-labels, sampled with
    balanced classes (half from human labels if there are pseudo-labels); the rest is
    unlabelled and gets a unique negative group.
    """
    classes = sorted({lbl for _, lbl in job["labeled"]} | {lbl for _, lbl in job["pseudo"]})
    class_id = {c: i for i, c in enumerate(classes)}

    def by_class(items: list) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for path, lbl in items:
            out.setdefault(lbl, []).append(path)
        return out

    human, pseudo = by_class(job["labeled"]), by_class(job["pseudo"])
    batch = job["batch_size"]
    n_sup = max(2, int(round(batch * job["labeled_fraction"]))) if human else 0
    if not job["unlabeled"]:
        n_sup = batch
    n_human = n_sup if not pseudo else (n_sup + 1) // 2

    def draw(pool: dict[str, list[str]], k: int) -> list[tuple[str, int]]:
        names = list(pool)
        picked = rng.choice(len(names), size=k)
        return [(pool[names[c]][rng.integers(len(pool[names[c]]))], class_id[names[c]]) for c in picked]

    plan: list[tuple[str, int]] = []
    unique = 0
    for _ in range(job["steps"]):
        items = draw(human, n_human) if human else []
        if pseudo:
            items += draw(pseudo, n_sup - n_human)
        for path in rng.choice(job["unlabeled"], size=batch - len(items)) if batch > len(items) else []:
            unique += 1
            items.append((str(path), -unique))
        plan += items
    return plan, class_id


def set_trainable(model: Any, unfreeze_blocks: int) -> list:
    """Freeze the backbone except the last ``unfreeze_blocks`` blocks (0 = train everything)."""
    params = list(model.parameters())
    blocks = getattr(model, "blocks", None)
    if not unfreeze_blocks or blocks is None:
        for p in params:
            p.requires_grad = True
        return params
    for p in params:
        p.requires_grad = False
    trainable = list(blocks[-unfreeze_blocks:].parameters())
    if getattr(model, "norm", None) is not None:
        trainable += list(model.norm.parameters())
    for p in trainable:
        p.requires_grad = True
    return trainable


def train_contrastive(
    job: dict[str, Any],
    backbone: Any = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Contrastive fine-tuning, then recompute all embeddings in ``job['out_dir']``.

    Writes ``backbone.pt``, ``train_log.json``, ``progress.json`` and ``embeddings/``.
    ``backbone`` is for tests; normally the pre-trained model of ``job['model']`` is loaded.
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from .embeddings import build_torch_encoder, load_backbone, pick_device

    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def progress(phase: str, step: int = 0, total: int = 0, loss: Optional[float] = None) -> None:
        tmp = out / "progress.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"phase": phase, "step": step, "total": total, "loss": loss, "elapsed_s": round(time.time() - t0)}, f)
        tmp.replace(out / "progress.json")

    progress("loading")
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    device = pick_device(job["device"])
    model = (backbone if backbone is not None else load_backbone(job["model"])).to(device)
    model.eval()  # DINOv2 has no active batch norm or dropout: stays in eval, only the chosen weights are updated
    trainable = set_trainable(model, job["unfreeze_blocks"])

    plan, class_id = plan_batches(job, rng)
    transform = build_train_transform(job["image_size"])

    class _TwoViews(Dataset):
        def __len__(self) -> int:
            return len(plan)

        def __getitem__(self, i: int):
            path, group = plan[i]
            with Image.open(path) as im:
                img = im.convert("RGB")
            return transform(img), transform(img), group

    loader = DataLoader(_TwoViews(), batch_size=job["batch_size"], shuffle=False, num_workers=job["num_workers"])

    with torch.no_grad():
        dim = model(torch.zeros(1, 3, job["image_size"], job["image_size"], device=device)).shape[1]
    head = torch.nn.Sequential(
        torch.nn.Linear(dim, dim), torch.nn.GELU(), torch.nn.Linear(dim, job["projection_dim"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": job["lr_backbone"]}, {"params": head.parameters(), "lr": job["lr_head"]}],
        weight_decay=0.05,
    )
    steps = job["steps"]
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup)))
    )
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    n_pseudo = len(job["pseudo"])
    log(f"device {device}; {len(job['labeled'])} labelled, {n_pseudo} pseudo-labels, {len(job['unlabeled'])} unlabelled; "
        f"{len(class_id)} classes; {sum(p.numel() for p in trainable) / 1e6:.1f}M trained parameters")
    losses: list[float] = []
    for step, (v1, v2, groups) in enumerate(loader, start=1):
        x = torch.cat([v1, v2]).to(device, non_blocking=True)
        groups = torch.cat([groups, groups]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            z = head(model(x))
        loss = supcon_loss(torch.nn.functional.normalize(z.float(), dim=1), groups, job["temperature"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0 or step == steps:
            progress("training", step, steps, float(np.mean(losses[-10:])))
            log(f"step {step}/{steps} loss {np.mean(losses[-10:]):.4f}")
    if not all(np.isfinite(losses)):
        raise RuntimeError("non-finite loss during training")

    model.eval()
    torch.save(model.state_dict(), out / WEIGHTS_NAME)
    progress("embedding", steps, steps, float(np.mean(losses[-10:])))
    encoder = build_torch_encoder(model, job["image_size"], job["embed_batch_size"], job["num_workers"], device)
    extract_embeddings(job["all_paths"], out / "embeddings", encoder,
                       meta={"model": job["model"], "image_size": job["image_size"], "weights": "contrastive"},
                       shard_size=job["shard_size"], log=log, image_ids=job.get("all_ids"))
    summary = {
        "method": "supcon + simclr (pseudo-label)", "device": device, "steps": steps,
        "n_labeled": len(job["labeled"]), "n_pseudo": n_pseudo, "n_unlabeled": len(job["unlabeled"]),
        "classes": sorted(class_id), "loss_first": float(np.mean(losses[:10])), "loss_last": float(np.mean(losses[-10:])),
        "losses": [round(v, 4) for v in losses], "duration_s": round(time.time() - t0, 1),
    }
    with open(out / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    progress("done", steps, steps, summary["loss_last"])
    return summary
