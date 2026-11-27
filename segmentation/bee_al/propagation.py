"""Label spreading on a sparse kNN graph (Zhou et al., 2004). No network to train."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import scipy.sparse as sp

from .knn import Prediction
from .memory_bank import MemoryBank


def build_knn_graph(bank: MemoryBank, n_neighbors: int = 15) -> sp.csr_matrix:
    """Symmetric, normalised kNN graph ``S = D^-1/2 W D^-1/2`` over all bees.

    Weights are the cosine similarities (clipped at 0). It depends only on the embeddings:
    it is built once and reused until the extractor changes.
    """
    n = len(bank)
    sims, idx = bank.search(bank.X, min(n_neighbors + 1, n))
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    vals = np.clip(sims.ravel(), 0.0, None).astype(np.float64)
    keep = rows != cols
    W = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    D = sp.diags(inv_sqrt)
    return (D @ W @ D).tocsr()


def load_or_build_graph(bank: MemoryBank, n_neighbors: int = 15, cache_dir: Union[str, Path, None] = None) -> sp.csr_matrix:
    """Like ``build_knn_graph``, but reuses the graph saved on disk if the embeddings are the same.

    On large datasets building the graph takes about a minute; the cached file is tied
    to the content of the embeddings and to the number of neighbours, so it never needs manual invalidation.
    """
    if cache_dir is None:
        return build_knn_graph(bank, n_neighbors)
    path = graph_cache_path(bank.X, n_neighbors, cache_dir)
    if path.exists():
        try:
            graph = sp.load_npz(path).tocsr()
            if graph.shape == (len(bank), len(bank)):
                return graph
        except Exception:
            pass  # corrupted file (e.g. interrupted write): rebuild it
    graph = build_knn_graph(bank, n_neighbors)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    sp.save_npz(tmp, graph)
    os.replace(tmp, path)
    return graph


def graph_cache_path(X: np.ndarray, n_neighbors: int, cache_dir: Union[str, Path]) -> Path:
    """Graph cache file: tied to the content of the embeddings (float32, as in the bank)."""
    key = hashlib.sha1(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()[:16]
    return Path(cache_dir) / f"knn_graph_{key}_k{n_neighbors}.npz"


def build_graph_cache_in_subprocess(
    X: np.ndarray, n_neighbors: int, cache_dir: Union[str, Path], backend: str = "auto"
) -> Path:
    """Build the graph of ``X`` in a separate Python process and leave it in the cache.

    Needed by the hub: the faiss search over all bees does not release the GIL, so building
    the graph inside the server process blocks it for minutes (even pages that do not use the
    engine). The calling process waits without holding the GIL; then ``load_or_build_graph``
    finds the cached file and loads it in no time.
    """
    from .config import REPO_ROOT

    path = graph_cache_path(X, n_neighbors, cache_dir)
    if path.exists():
        return path
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        x_path = Path(tmp) / "X.npy"
        np.save(x_path, np.ascontiguousarray(X, dtype=np.float32))
        subprocess.run([sys.executable, "-m", "bee_al.propagation", str(x_path), str(n_neighbors), str(cache_dir), backend],
                       cwd=REPO_ROOT, check=True)
    return path


class LabelSpreader:
    """Propagate labels along the graph: ``F <- alpha * S F + (1 - alpha) * Y0``."""

    def __init__(self, graph: sp.csr_matrix, alpha: float = 0.8, max_iter: int = 30, tol: float = 1e-4) -> None:
        self.graph = graph
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    @classmethod
    def from_bank(
        cls, bank: MemoryBank, n_neighbors: int = 15, alpha: float = 0.8, max_iter: int = 30, tol: float = 1e-4,
        cache_dir: Union[str, Path, None] = None,
    ) -> "LabelSpreader":
        return cls(load_or_build_graph(bank, n_neighbors, cache_dir), alpha=alpha, max_iter=max_iter, tol=tol)

    def propagate(self, labeled_idx: Sequence[int], labels: Sequence[str]) -> Prediction:
        """Class distribution for every node of the graph.

        Classes are those present in ``labels`` (even just one, or none).
        Labelled bees stay fixed to their label; those not reached
        by any label get a row of zeros.
        """
        n = self.graph.shape[0]
        classes = sorted(set(labels))
        if not classes:
            return Prediction(classes, np.zeros((n, 0)))
        class_pos = {c: j for j, c in enumerate(classes)}
        labeled_idx = np.asarray(labeled_idx, dtype=np.int64)
        cols = np.array([class_pos[lbl] for lbl in labels], dtype=np.int64)

        Y0 = np.zeros((n, len(classes)))
        Y0[labeled_idx, cols] = 1.0
        F = Y0.copy()
        for _ in range(self.max_iter):
            F_new = self.alpha * (self.graph @ F) + (1.0 - self.alpha) * Y0
            delta = np.abs(F_new - F).max()
            F = F_new
            if delta < self.tol:
                break

        F[labeled_idx] = Y0[labeled_idx]
        mass = F.sum(axis=1, keepdims=True)
        probs = np.divide(F, mass, out=np.zeros_like(F), where=mass > 1e-12)
        return Prediction(classes, probs)

    def propagate_bank(self, bank: MemoryBank) -> Prediction:
        idx = bank.labeled_indices()
        return self.propagate(idx, bank.labels_at(idx))


if __name__ == "__main__":
    # Used by build_graph_cache_in_subprocess: python -m bee_al.propagation X.npy n_neighbors cache_dir backend
    x_file, k, cache, backend_name = sys.argv[1:5]
    X_all = np.load(x_file)
    load_or_build_graph(MemoryBank([str(i) for i in range(len(X_all))], X_all, backend=backend_name), int(k), cache)
