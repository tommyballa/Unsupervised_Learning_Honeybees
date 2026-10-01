"""Memory bank: tutti gli embedding più le etichette note, con ricerca dei k vicini."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union

import numpy as np

from . import REMOVED_LABEL, UNKNOWN_LABEL

BACKENDS = ("faiss", "sklearn", "numpy")


class NeighborIndex(Protocol):
    """Interfaccia unica dei backend: similarità coseno decrescenti e indici dei vicini."""

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str = "auto") -> str:
    """``auto`` usa FAISS se installato, altrimenti sklearn."""
    if backend == "auto":
        return "faiss" if faiss_available() else "sklearn"
    if backend not in BACKENDS:
        raise ValueError(f"Backend sconosciuto: {backend}")
    if backend == "faiss" and not faiss_available():
        raise ImportError("backend 'faiss' richiesto ma faiss non è installato")
    return backend


class _FaissIndex:
    def __init__(self, X: np.ndarray) -> None:
        import faiss

        self.n = len(X)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(np.ascontiguousarray(X, dtype=np.float32))

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims, idx = self.index.search(np.ascontiguousarray(Q, dtype=np.float32), min(k, self.n))
        return sims, idx.astype(np.int64)


class _SklearnIndex:
    def __init__(self, X: np.ndarray) -> None:
        from sklearn.neighbors import NearestNeighbors

        self.n = len(X)
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self.nn.kneighbors(Q, n_neighbors=min(k, self.n))
        return (1.0 - dist).astype(np.float32), idx.astype(np.int64)


class _NumpyIndex:
    def __init__(self, X: np.ndarray, chunk: int = 2048) -> None:
        self.X = X
        self.chunk = chunk

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.X))
        sims_out, idx_out = [], []
        for s in range(0, len(Q), self.chunk):
            sims = Q[s:s + self.chunk] @ self.X.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_sims = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-top_sims, axis=1, kind="stable")
            sims_out.append(np.take_along_axis(top_sims, order, axis=1))
            idx_out.append(np.take_along_axis(top, order, axis=1))
        return np.concatenate(sims_out).astype(np.float32), np.concatenate(idx_out).astype(np.int64)


def build_index(X: np.ndarray, backend: str = "auto") -> NeighborIndex:
    """Indice esatto sui vettori (già normalizzati L2) con il backend scelto."""
    backend = resolve_backend(backend)
    if backend == "faiss":
        return _FaissIndex(X)
    if backend == "sklearn":
        return _SklearnIndex(X)
    return _NumpyIndex(X)


class MemoryBank:
    """Embedding di tutte le api e, per quelle etichettate, la label.

    Le classi non sono fissate: sono quelle delle etichette presenti in questo momento.
    ``add_label`` non addestra nulla: aggiorna un dizionario e invalida l'indice
    delle etichettate, che viene ricostruito alla ricerca successiva.
    """

    def __init__(
        self,
        ids: Sequence[str],
        X: np.ndarray,
        paths: Optional[Sequence[str]] = None,
        backend: str = "auto",
    ) -> None:
        if len(ids) != len(X):
            raise ValueError("ids e X hanno lunghezze diverse")
        self.ids: list[str] = list(ids)
        self.paths: list[str] = list(paths) if paths is not None else list(ids)
        self.X: np.ndarray = np.ascontiguousarray(X, dtype=np.float32)
        self.backend: str = resolve_backend(backend)
        self._pos: dict[str, int] = {image_id: i for i, image_id in enumerate(self.ids)}
        if len(self._pos) != len(self.ids):
            raise ValueError("image_id duplicati")
        self._labels: dict[int, str] = {}
        self._unknown: set[int] = set()
        self._full_index: Optional[NeighborIndex] = None
        self._labeled_index: Optional[NeighborIndex] = None
        self._labeled_idx: np.ndarray = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._pos

    def index_of(self, image_id: str) -> int:
        return self._pos[image_id]

    # ------------------------------ etichette ------------------------------ #

    def add_label(self, image_id: str, label: str) -> None:
        """Aggiunge o aggiorna un'etichetta. ``UNKNOWN_LABEL`` mette da parte, ``REMOVED_LABEL`` rimuove."""
        i = self._pos[image_id]
        self._labels.pop(i, None)
        self._unknown.discard(i)
        if label == UNKNOWN_LABEL:
            self._unknown.add(i)
        elif label != REMOVED_LABEL:
            self._labels[i] = label
        self._labeled_index = None

    def remove_label(self, image_id: str) -> None:
        self.add_label(image_id, REMOVED_LABEL)

    def clear_labels(self) -> None:
        self._labels.clear()
        self._unknown.clear()
        self._labeled_index = None

    def label_of(self, image_id: str) -> Optional[str]:
        """Label dell'immagine, ``UNKNOWN_LABEL`` se "Non so", ``None`` se non etichettata."""
        i = self._pos[image_id]
        return UNKNOWN_LABEL if i in self._unknown else self._labels.get(i)

    @property
    def classes(self) -> list[str]:
        """Classi attuali, in ordine alfabetico ("Non so" non è una classe)."""
        return sorted(set(self._labels.values()))

    @property
    def n_labeled(self) -> int:
        return len(self._labels)

    def labeled_indices(self) -> np.ndarray:
        return np.fromiter(sorted(self._labels), dtype=np.int64, count=len(self._labels))

    def labels_at(self, indices: Sequence[int]) -> list[str]:
        return [self._labels[int(i)] for i in indices]

    def unknown_indices(self) -> np.ndarray:
        return np.fromiter(sorted(self._unknown), dtype=np.int64, count=len(self._unknown))

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lbl in self._labels.values():
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------- ricerca ------------------------------- #

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k vicini tra tutte le api: (similarità, indici)."""
        if self._full_index is None:
            self._full_index = build_index(self.X, self.backend)
        return self._full_index.search(np.atleast_2d(Q), k)

    def search_labeled(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k vicini tra le sole api etichettate: (similarità, indici globali)."""
        Q = np.atleast_2d(Q)
        if not self._labels:
            return np.empty((len(Q), 0), dtype=np.float32), np.empty((len(Q), 0), dtype=np.int64)
        if self._labeled_index is None:
            self._labeled_idx = self.labeled_indices()
            self._labeled_index = build_index(self.X[self._labeled_idx], self.backend)
        sims, local = self._labeled_index.search(Q, k)
        return sims, self._labeled_idx[local]

    # ----------------------------- persistenza ----------------------------- #

    def save(self, out_dir: Union[str, Path]) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "bank.npz", X=self.X, ids=np.asarray(self.ids, dtype=str), paths=np.asarray(self.paths, dtype=str))
        state = {
            "labels": {self.ids[i]: lbl for i, lbl in sorted(self._labels.items())},
            "unknown": [self.ids[i] for i in sorted(self._unknown)],
        }
        with open(out / "labels.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, in_dir: Union[str, Path], backend: str = "auto") -> "MemoryBank":
        d = Path(in_dir)
        with np.load(d / "bank.npz") as z:
            bank = cls(z["ids"].tolist(), z["X"], z["paths"].tolist(), backend=backend)
        with open(d / "labels.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        for image_id, lbl in state["labels"].items():
            bank.add_label(image_id, lbl)
        for image_id in state["unknown"]:
            bank.add_label(image_id, UNKNOWN_LABEL)
        return bank
