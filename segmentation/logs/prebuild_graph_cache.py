"""Builds in advance (in a separate process) the kNN graph that the hub will use at its next start.

Repeats the steps of ALEngine.from_config up to bank.X, so that the cache key matches.
    PYTHONPATH=. venv/bin/python logs/prebuild_graph_cache.py
"""
import sys
import time

from bee_al.config import load_config
from bee_al.embeddings import load_embeddings, restrict_embeddings, restrict_to_datasets
from bee_al.memory_bank import MemoryBank
from bee_al.propagation import build_graph_cache_in_subprocess, graph_cache_path
from bee_al.retrain import latest_embeddings

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/al_config.yaml")
version, emb_dir = latest_embeddings(cfg)
emb = load_embeddings(emb_dir)
if cfg.paths.dataset_dirs:
    emb = restrict_to_datasets(emb, list(cfg.paths.dataset_dirs))
if cfg.subset_path is not None:
    emb = restrict_embeddings(emb, cfg.subset_path)
X = MemoryBank(emb.ids, emb.X, emb.paths, backend=cfg.knn.backend).X
cache_dir, k = cfg.results_dir / "cache", cfg.propagation.n_neighbors
path = graph_cache_path(X, k, cache_dir)
print(version, len(X), path, "already cached" if path.exists() else "to be built", flush=True)
t = time.time()
build_graph_cache_in_subprocess(X, k, cache_dir, cfg.knn.backend)
print(f"ready in {time.time() - t:.0f} s:", path.exists())
