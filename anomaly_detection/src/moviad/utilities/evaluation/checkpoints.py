"""
Saving the best configuration of a model and loading it
for experiments/plot_heatmaps.py.

Each model has its own folder:

    checkpoints/<single_bee|full_image>/<model>/
        best.pth      model state at the best epoch (or after fitting)
        config.json   script hyperparameters, epoch and metrics of the best

During training the best is chosen by BestTracker, comparing the metrics
in lexicographic order (default: img_roc_auc, then pxl_avg_prec on ties)
on the selection data (selection_data): the single_bee scripts use the
validation set, so the final metrics on the held-out test set are not
optimistic; with the test set, the best's metrics are (same data).
"""

import json
import math
from pathlib import Path

import torch


BEST_METRICS = ("img_roc_auc", "pxl_avg_prec")
STATE_FILE = "best.pth"
CONFIG_FILE = "config.json"


def script_config(namespace: dict) -> dict:
    """UPPERCASE constants of a script (its hyperparameters) that are JSON-serializable."""
    config = {}
    for key, value in namespace.items():
        if not key.isupper() or key.startswith("_"):
            continue
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, tuple):
            value = list(value)
        try:
            json.dumps(value)
        except TypeError:
            continue
        config[key] = value
    return config


def clean_state_dict(state: dict) -> dict:
    """CPU copy of the state dict, without the "_orig_mod." prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in state.items()}


def save_checkpoint(ckpt_dir, state, config: dict, **info):
    """Writes best.pth and config.json to ckpt_dir (info: epoch, metrics, ...)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / STATE_FILE)
    with open(ckpt_dir / CONFIG_FILE, "w") as f:
        json.dump({**info, "config": config}, f, indent=2)


def load_checkpoint(ckpt_dir, map_location="cpu"):
    """Returns (state, info) saved by save_checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    state_path = ckpt_dir / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {state_path} (run the training first)")
    # weights_only=False: some states contain numpy arrays (e.g. PaDiM)
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    info = {}
    if (ckpt_dir / CONFIG_FILE).exists():
        with open(ckpt_dir / CONFIG_FILE) as f:
            info = json.load(f)
    return state, info


class BestTracker:
    """
    Keeps track of the best epoch and saves its checkpoint.

    Example:
        tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))
        ...
        tracker.update(epoch + 1, results, lambda: model_state(model))
    """

    def __init__(self, ckpt_dir, config: dict, metrics=BEST_METRICS, selection_data="test set"):
        self.ckpt_dir = Path(ckpt_dir)
        self.config = config
        self.metrics = tuple(metrics)
        self.selection_data = selection_data
        self.best_key = None
        self.best_epoch = None
        self.best_results = None

    def _key(self, results):
        values = []
        for name in self.metrics:
            value = results.get(name, float("-inf"))
            values.append(float("-inf") if value is None or math.isnan(value) else value)
        return tuple(values)

    def update(self, epoch: int, results: dict, state_fn) -> bool:
        """If results is the best so far, saves state_fn() as best. Returns True if it improves."""
        key = self._key(results)
        if self.best_key is not None and key <= self.best_key:
            print(f"  best unchanged: epoch {self.best_epoch} "
                  f"({', '.join(f'{m}={v:.4f}' for m, v in zip(self.metrics, self.best_key))})")
            return False

        self.best_key, self.best_epoch, self.best_results = key, epoch, dict(results)
        save_checkpoint(
            self.ckpt_dir, state_fn(), self.config,
            best_epoch=epoch,
            selection_metrics=list(self.metrics),
            selection_data=self.selection_data,
            metrics=self.best_results,
        )
        print(f"  ↑ new best at epoch {epoch} "
              f"({', '.join(f'{m}={v:.4f}' for m, v in zip(self.metrics, key))}) -> {self.ckpt_dir}")
        return True

    def summary(self):
        if self.best_epoch is None:
            return "no evaluation run"
        return (f"epoch {self.best_epoch}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in self.best_results.items()))
