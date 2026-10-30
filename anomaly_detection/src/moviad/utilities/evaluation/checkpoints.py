"""
Salvataggio della configurazione migliore (best) di un modello e caricamento
per experiments/plot_heatmaps.py.

Ogni modello ha la sua cartella:

    checkpoints/<single_bee|full_image>/<modello>/
        best.pth      stato del modello all'epoca migliore (o dopo il fitting)
        config.json   iperparametri dello script, epoca e metriche della best

Durante il training la best viene scelta con BestTracker confrontando le
metriche in ordine lessicografico (default: img_roc_auc, a parità
pxl_avg_prec) sui dati di selezione (selection_data): gli script single_bee
usano il validation set, così le metriche finali sul test held-out non sono
ottimistiche; con il test set quelle della best lo sono (stessi dati).
"""

import json
import math
from pathlib import Path

import torch


BEST_METRICS = ("img_roc_auc", "pxl_avg_prec")
STATE_FILE = "best.pth"
CONFIG_FILE = "config.json"


def script_config(namespace: dict) -> dict:
    """Costanti MAIUSCOLE di uno script (i suoi iperparametri) serializzabili in JSON."""
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
    """Copia su CPU dello state dict, senza il prefisso "_orig_mod." di torch.compile."""
    return {k.replace("_orig_mod.", ""): v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in state.items()}


def save_checkpoint(ckpt_dir, state, config: dict, **info):
    """Scrive best.pth e config.json in ckpt_dir (info: epoca, metriche, ...)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / STATE_FILE)
    with open(ckpt_dir / CONFIG_FILE, "w") as f:
        json.dump({**info, "config": config}, f, indent=2)


def load_checkpoint(ckpt_dir, map_location="cpu"):
    """Restituisce (state, info) salvati da save_checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    state_path = ckpt_dir / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {state_path} (lanciare prima il training)")
    # weights_only=False: alcuni stati contengono array numpy (es. PaDiM)
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    info = {}
    if (ckpt_dir / CONFIG_FILE).exists():
        with open(ckpt_dir / CONFIG_FILE) as f:
            info = json.load(f)
    return state, info


class BestTracker:
    """
    Tiene traccia dell'epoca migliore e salva il suo checkpoint.

    Esempio:
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
        """Se results è la migliore finora salva state_fn() come best. Restituisce True se migliora."""
        key = self._key(results)
        if self.best_key is not None and key <= self.best_key:
            print(f"  best invariata: epoca {self.best_epoch} "
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
        print(f"  ↑ nuova best all'epoca {epoch} "
              f"({', '.join(f'{m}={v:.4f}' for m, v in zip(self.metrics, key))}) -> {self.ckpt_dir}")
        return True

    def summary(self):
        if self.best_epoch is None:
            return "nessuna valutazione eseguita"
        return (f"epoca {self.best_epoch}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in self.best_results.items()))
