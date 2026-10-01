"""Selezione del batch di api da far etichettare (active learning)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .config import QueryConfig
from .uncertainty import uncertainty_scores

RANDOM = "random"
DIVERSITY = "diversity"
UNCERTAINTY = "uncertainty"
NOVELTY = "novelty"


@dataclass(frozen=True)
class Query:
    index: int        # posizione dell'ape nella memory bank
    query_type: str   # random | diversity | uncertainty | novelty


def is_cold_start(n_labeled: int, n_classes: int, cfg: QueryConfig) -> bool:
    """Cold start finché le etichette sono poche o c'è meno di 2 classi."""
    return n_labeled < cfg.cold_start_threshold or n_classes < 2


def kmeans_representatives(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indici (in ``X``) di ``k`` punti distinti, ognuno il più vicino a un centroide k-means."""
    n = len(X)
    if k >= n:
        return np.arange(n)
    seed = int(rng.integers(2**31 - 1))
    if n > 10_000:
        km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed)
    else:
        km = KMeans(n_clusters=k, n_init=1, random_state=seed)
    km.fit(X)
    # Distanza al quadrato punto-centroide, a meno di ||x||^2 che non serve per l'argmin per centroide.
    centers = km.cluster_centers_
    dist = (centers**2).sum(axis=1)[:, None] - 2.0 * centers @ X.T + (X**2).sum(axis=1)[None, :]
    chosen: list[int] = []
    taken = np.zeros(n, dtype=bool)
    for row in dist:
        row = np.where(taken, np.inf, row)
        j = int(np.argmin(row))
        chosen.append(j)
        taken[j] = True
    return np.asarray(chosen, dtype=np.int64)


def select_batch(
    X: np.ndarray,
    excluded: Iterable[int],
    probs: Optional[np.ndarray],
    n_labeled: int,
    n_classes: int,
    cfg: QueryConfig,
    rng: np.random.Generator,
    novelty: Optional[np.ndarray] = None,
) -> list[Query]:
    """Sceglie il prossimo batch tra le api non in ``excluded``.

    ``excluded`` deve contenere etichettate, "Non so", validation set e batch in corso.
    ``probs`` è la distribuzione sulle classi di tutte le api (kNN o propagazione),
    ignorata in cold start. ``novelty`` è la distanza di ogni ape dall'esempio
    etichettato più vicino; se manca, la quota novelty va all'incertezza.

    - Quota random: ``random_fraction`` del batch, a caso tra le candidate.
    - Cold start: k-means sulle candidate, le api più vicine ai centroidi (``diversity``).
    - Regime normale: top-M più incerte, k-means su quelle, un rappresentante per
      cluster (``uncertainty``). Mai le top-N in blocco.
    - Quota novelty (solo regime normale): ``novelty_fraction`` del batch tra le api più
      lontane da tutto ciò che è etichettato, per far emergere classi rare mai viste.
      Anche qui top-M e poi un rappresentante per cluster.
    """
    mask = np.ones(len(X), dtype=bool)
    mask[np.fromiter(excluded, dtype=np.int64)] = False
    candidates = np.flatnonzero(mask)
    batch = min(cfg.batch_size, len(candidates))
    if batch == 0:
        return []

    n_random = min(batch, int(round(cfg.random_fraction * batch)))
    random_pick = rng.choice(candidates, size=n_random, replace=False)
    queries = [Query(int(i), RANDOM) for i in random_pick]

    pool = np.setdiff1d(candidates, random_pick)
    n_main = batch - n_random
    if n_main == 0:
        return queries

    cold = probs is None or is_cold_start(n_labeled, n_classes, cfg)
    if not cold and novelty is not None:
        n_novelty = min(n_main, int(round(cfg.novelty_fraction * batch)))
        if n_novelty:
            order = np.lexsort((rng.random(len(pool)), -novelty[pool]))
            far = pool[order[:cfg.candidate_multiplier * n_novelty]]
            picked = far[kmeans_representatives(X[far], n_novelty, rng)]
            queries += [Query(int(i), NOVELTY) for i in picked]
            pool = np.setdiff1d(pool, picked)
            n_main -= n_novelty

    if n_main == 0:
        pass
    elif cold:
        pool_for_kmeans, kind = pool, DIVERSITY
    else:
        unc = uncertainty_scores(probs[pool], cfg.uncertainty)
        # Parità di incertezza risolta a caso, non per ordine di file.
        order = np.lexsort((rng.random(len(pool)), -unc))
        pool_for_kmeans, kind = pool[order[:cfg.candidate_multiplier * n_main]], UNCERTAINTY

    if n_main:
        reps = kmeans_representatives(X[pool_for_kmeans], n_main, rng)
        queries += [Query(int(i), kind) for i in pool_for_kmeans[reps]]
    order = rng.permutation(len(queries))  # l'annotatore non deve riconoscere il tipo dall'ordine
    return [queries[i] for i in order]
