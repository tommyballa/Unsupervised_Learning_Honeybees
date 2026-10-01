"""Estrazione e persistenza degli embedding (shard + manifest, riprendibile).

Il formato su disco non dipende da torch: l'hub legge gli embedding con solo numpy.
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

# Riceve una lista di percorsi e restituisce un array (n, dim); una riga di NaN = immagine illeggibile.
Encoder = Callable[[Sequence[Path]], np.ndarray]


@dataclass
class Embeddings:
    """Embedding float32 normalizzati L2, con image_id (nome file) e percorso."""

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
    """Immagini del dataset in ordine alfabetico (stesse estensioni dell'hub)."""
    d = Path(dataset_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


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
) -> int:
    """Estrae gli embedding mancanti e li salva a shard. Restituisce quanti ne ha aggiunti.

    Si può interrompere e rilanciare: le immagini già presenti nel manifest vengono saltate.
    ``meta`` (modello, dimensione immagine, ...) deve coincidere con quello di un'estrazione precedente.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    manifest = read_manifest(out) or {"meta": meta, "shards": [], "failed": [], "dim": None}
    if manifest["meta"] != meta:
        raise ValueError(
            f"{out} contiene embedding estratti con {manifest['meta']}, richiesti {meta}. "
            "Usa un'altra cartella o cancella quella esistente."
        )

    done = _done_ids(out, manifest)
    todo = [Path(p) for p in image_paths if Path(p).name not in done]
    log(f"{len(done)} immagini già fatte, {len(todo)} da estrarre")

    added = 0
    for start in range(0, len(todo), shard_size):
        chunk = todo[start:start + shard_size]
        emb = np.asarray(encoder(chunk), dtype=np.float32)
        ok = np.isfinite(emb).all(axis=1)
        manifest["failed"].extend(p.name for p, good in zip(chunk, ok) if not good)
        kept = [p for p, good in zip(chunk, ok) if good]
        if kept:
            X = l2_normalize(emb[ok])
            name = f"shard_{len(manifest['shards']):05d}.npz"
            _write_shard(out, name, [p.name for p in kept], [str(p) for p in kept], X)
            manifest["shards"].append({"file": name, "count": len(kept)})
            manifest["dim"] = int(X.shape[1])
            added += len(kept)
        # Il manifest è aggiornato solo dopo lo shard: un'interruzione non lascia stati a metà.
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
) -> None:
    """Salva in un colpo solo embedding già calcolati (usato dai re-train e dai test)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    """Carica tutti gli shard elencati nel manifest."""
    d = Path(emb_dir)
    manifest = read_manifest(d)
    if manifest is None or not manifest["shards"]:
        raise FileNotFoundError(f"Nessun embedding in {d}: esegui prima scripts/extract_embeddings.py")
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


# --------------------------------------------------------------------------- #
# Encoder torch (importato solo da chi estrae: l'hub non ne ha bisogno)
# --------------------------------------------------------------------------- #

def pick_device(device: str = "auto") -> str:
    """``auto`` sceglie la GPU con più memoria libera, altrimenti la CPU."""
    import torch

    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


def build_transform(image_size: int = 224):
    """Stesso preprocessing di ``dataset_clustering.py``: resize quadrato + normalizzazione ImageNet."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_backbone(model_name: str = "dinov2_vits14"):
    """Modello DINOv2 pre-addestrato via torch.hub (es. ``dinov2_vits14``, ``dinov2_vitb14``)."""
    import torch

    return torch.hub.load("facebookresearch/dinov2", model_name)


def build_torch_encoder(
    model: Any,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
) -> Encoder:
    """Avvolge un modello torch congelato in un ``Encoder`` che lavora a batch."""
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
                    img = im.convert("RGB")  # scarta l'alpha, come cv2.imread
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
