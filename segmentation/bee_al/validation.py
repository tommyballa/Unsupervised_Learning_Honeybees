"""Fixed validation set and evaluation metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score

NO_PREDICTION = "__none__"


def load_or_create_validation_set(
    path: Union[str, Path], image_ids: Sequence[str], size: int, seed: int
) -> list[str]:
    """Random sample fixed by a seed and saved to file.

    If the file exists it is reused as-is: changing ``size`` or ``seed`` in the
    config does not regenerate it (it must be deleted by hand, losing comparability across rounds).
    """
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        known = set(image_ids)
        kept = [i for i in saved["image_ids"] if i in known]
        if saved["image_ids"] and not kept:
            raise ValueError(
                f"{path} elenca api che non sono in questo dataset: è il validation set di un altro "
                "dataset. Usa un'altra paths.results_dir oppure sposta i vecchi risultati."
            )
        return kept
    rng = np.random.default_rng(seed)
    pool = sorted(image_ids)
    chosen = sorted(rng.choice(pool, size=min(size, len(pool)), replace=False).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "size": len(chosen), "image_ids": chosen}, f, indent=2)
    return chosen


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[Optional[str]]) -> dict[str, Optional[float]]:
    """Accuracy and macro-F1 (averaged over the classes present in the reference)."""
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None}
    pred = [NO_PREDICTION if p is None else p for p in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
    }


def clustering_metrics(y_true: Sequence[str], y_pred: Sequence[Optional[str]]) -> dict[str, Optional[float]]:
    """NMI and ARI of the predicted clusters against the reference labels."""
    if len(y_true) == 0:
        return {"nmi": None, "ari": None}
    pred = [NO_PREDICTION if p is None else p for p in y_pred]
    return {
        "nmi": float(normalized_mutual_info_score(y_true, pred)),
        "ari": float(adjusted_rand_score(y_true, pred)),
    }
