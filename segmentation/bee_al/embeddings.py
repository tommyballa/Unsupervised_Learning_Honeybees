"""Extraction and persistence of embeddings (shards + manifest, resumable).

The on-disk format does not depend on torch: the hub reads the embeddings with numpy only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MANIFEST_NAME = "manifest.json"

# Takes a list of paths and returns an (n, dim) array; a row of NaN = unreadable image.
Encoder = Callable[[Sequence[Path]], np.ndarray]


@dataclass
class Embeddings:
    """L2-normalised float32 embeddings, with image_id (file name) and path."""

    ids: list[str]
    paths: list[str]
    X: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def list_images(dataset_dir: Union[str, Path]) -> list[Path]:
    """Dataset images in alphabetical order (same extensions as the hub)."""
    d = Path(dataset_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def list_dataset(sources: Sequence[tuple[str, Union[str, Path]]]) -> tuple[list[str], list[Path]]:
    """Image_ids and paths of all folders (see ``ALConfig.dataset_sources``).

    The image_id is the folder prefix plus the file name, so it stays unique
    even when two folders contain files with the same name.
    """
    ids: list[str] = []
    paths: list[Path] = []
    for prefix, folder in sources:
        if not Path(folder).is_dir():
            continue
        for p in list_images(folder):
            ids.append(f"{prefix}{p.name}")
            paths.append(p)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image_id across the dataset folders")
    return ids, paths


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _write_shard(out_dir: Path, name: str, ids: Sequence[str], paths: Sequence[str], X: np.ndarray) -> None:
    tmp = out_dir / (name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, embeddings=X, image_ids=np.asarray(ids, dtype=str), paths=np.asarray(paths, dtype=str))
    os.replace(tmp, out_dir / name)


def read_manifest(emb_dir: Union[str, Path]) -> Optional[dict[str, Any]]:
    path = Path(emb_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _done_ids(emb_dir: Path, manifest: dict[str, Any]) -> set[str]:
    done: set[str] = set(manifest.get("failed", []))
    for shard in manifest["shards"]:
        with np.load(emb_dir / shard["file"]) as z:
            done.update(z["image_ids"].tolist())
    return done


def extract_embeddings(
    image_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    encoder: Encoder,
    meta: Optional[dict[str, Any]] = None,
    shard_size: int = 2048,
    log: Callable[[str], None] = lambda s: None,
    image_ids: Optional[Sequence[str]] = None,
) -> int:
    """Extract the missing embeddings and save them in shards. Returns how many were added.

    It can be interrupted and restarted: images already in the manifest are skipped.
    ``image_ids`` gives an id to each path; if missing, the id is the file name.
    ``meta`` (model, image size, ...) must match that of any previous extraction.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    manifest = read_manifest(out) or {"meta": meta, "shards": [], "failed": [], "dim": None}
    if manifest["meta"] != meta:
        raise ValueError(
            f"{out} contains embeddings extracted with {manifest['meta']}, requested {meta}. "
            "Use another folder or delete the existing one."
        )

    paths = [Path(p) for p in image_paths]
    ids = list(image_ids) if image_ids is not None else [p.name for p in paths]
    if len(ids) != len(paths) or len(set(ids)) != len(ids):
        raise ValueError("image_ids must have a distinct id for each image")
    done = _done_ids(out, manifest)
    todo = [(i, p) for i, p in zip(ids, paths) if i not in done]
    log(f"{len(done)} images already done, {len(todo)} to extract")

    added = 0
    for start in range(0, len(todo), shard_size):
        chunk = todo[start:start + shard_size]
        emb = np.asarray(encoder([p for _, p in chunk]), dtype=np.float32)
        ok = np.isfinite(emb).all(axis=1)
        manifest["failed"].extend(i for (i, _), good in zip(chunk, ok) if not good)
        kept = [item for item, good in zip(chunk, ok) if good]
        if kept:
            X = l2_normalize(emb[ok])
            name = f"shard_{len(manifest['shards']):05d}.npz"
            _write_shard(out, name, [i for i, _ in kept], [str(p) for _, p in kept], X)
            manifest["shards"].append({"file": name, "count": len(kept)})
            manifest["dim"] = int(X.shape[1])
            added += len(kept)
        # The manifest is updated only after the shard: an interruption never leaves a half-written state.
        _write_json_atomic(out / MANIFEST_NAME, manifest)
        log(f"  {min(start + shard_size, len(todo))}/{len(todo)}")
    if not todo:
        _write_json_atomic(out / MANIFEST_NAME, manifest)
    return added


def save_embeddings(
    out_dir: Union[str, Path],
    ids: Sequence[str],
    paths: Sequence[str],
    X: np.ndarray,
    meta: Optional[dict[str, Any]] = None,
    normalize: bool = True,
) -> None:
    """Save already computed embeddings in one go (used by re-training and by the tests).

    ``normalize=False`` for already normalised vectors: normalising them again changes the last
    bits, and with them the kNN graph cache key (the graph must be reused even after a restart).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if normalize:
        X = l2_normalize(X)
    _write_shard(out, "shard_00000.npz", ids, paths, X)
    manifest = {
        "meta": dict(meta or {}),
        "shards": [{"file": "shard_00000.npz", "count": len(ids)}],
        "failed": [],
        "dim": int(X.shape[1]),
    }
    _write_json_atomic(out / MANIFEST_NAME, manifest)


def load_embeddings(emb_dir: Union[str, Path]) -> Embeddings:
    """Load all shards listed in the manifest."""
    d = Path(emb_dir)
    manifest = read_manifest(d)
    if manifest is None or not manifest["shards"]:
        raise FileNotFoundError(f"No embeddings in {d}: run scripts/extract_embeddings.py first")
    ids: list[str] = []
    paths: list[str] = []
    chunks: list[np.ndarray] = []
    for shard in manifest["shards"]:
        with np.load(d / shard["file"]) as z:
            ids.extend(z["image_ids"].tolist())
            paths.extend(z["paths"].tolist())
            chunks.append(z["embeddings"])
    X = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    return Embeddings(ids=ids, paths=paths, X=X, meta=manifest["meta"])


def restrict_to_datasets(emb: Embeddings, dataset_names: Sequence[str], separator: str = "__") -> Embeddings:
    """Keep only the bees from the given dataset folders (the image_id prefix).

    Embeddings may have been extracted for more folders than the ones to be used:
    removing a folder from ``paths.dataset_dirs`` is enough to exclude its bees.
    """
    prefixes = tuple(f"{name}{separator}" for name in dataset_names)
    keep = [i for i, image_id in enumerate(emb.ids) if image_id.startswith(prefixes)]
    if len(keep) == len(emb.ids):
        return emb
    if not keep:
        raise ValueError(f"No embeddings for the folders {list(dataset_names)}: run scripts/extract_embeddings.py")
    return Embeddings([emb.ids[i] for i in keep], [emb.paths[i] for i in keep],
                      np.ascontiguousarray(emb.X[keep]), {**emb.meta, "datasets": list(dataset_names)})


def restrict_embeddings(emb: Embeddings, subset_path: Union[str, Path]) -> Embeddings:
    """Keep only the bees listed in a ``{"image_ids": [...]}`` file, in embedding order."""
    path = Path(subset_path)
    if not path.exists():
        raise FileNotFoundError(f"Subset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        wanted = set(json.load(f)["image_ids"])
    keep = [i for i, image_id in enumerate(emb.ids) if image_id in wanted]
    if not keep:
        raise ValueError(f"{path} lists none of the bees in this dataset")
    return Embeddings([emb.ids[i] for i in keep], [emb.paths[i] for i in keep],
                      np.ascontiguousarray(emb.X[keep]), {**emb.meta, "subset": str(path), "subset_of": len(emb)})


# --------------------------------------------------------------------------- #
# Torch encoder (imported only by extraction code: the hub does not need it)
# --------------------------------------------------------------------------- #

def pick_device(device: str = "auto") -> str:
    """``auto`` picks the GPU with the most free memory, the CPU otherwise."""
    import torch

    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


def build_transform(image_size: int = 224):
    """Same preprocessing as ``dataset_clustering.py``: square resize + ImageNet normalisation."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_backbone(model_name: str = "dinov2_vits14"):
    """Pretrained DINOv2 model via torch.hub (e.g. ``dinov2_vits14``, ``dinov2_vitb14``)."""
    import torch

    return torch.hub.load("facebookresearch/dinov2", model_name)


def build_torch_encoder(
    model: Any,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
) -> Encoder:
    """Wrap a frozen torch model in a batch-wise ``Encoder``."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    dev = pick_device(device)
    model = model.to(dev).eval()
    transform = build_transform(image_size)

    class _Images(Dataset):
        def __init__(self, paths: Sequence[Path]) -> None:
            self.paths = list(paths)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int):
            try:
                with Image.open(self.paths[i]) as im:
                    img = im.convert("RGB")  # drop the alpha channel, like cv2.imread
                return transform(img), True
            except Exception:
                return torch.zeros(3, image_size, image_size), False

    def encode(paths: Sequence[Path]) -> np.ndarray:
        loader = DataLoader(_Images(paths), batch_size=batch_size, num_workers=num_workers)
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for batch, ok in loader:
                feats = model(batch.to(dev)).float().cpu().numpy()
                feats[~ok.numpy()] = np.nan
                out.append(feats)
        return np.concatenate(out)

    return encode
