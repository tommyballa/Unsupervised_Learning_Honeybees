"""
Ablation dei modelli single-bee: numero di api di training e backbone.

I main_<modello>.py leggono due variabili d'ambiente (assenti = configurazione
ufficiale, invariata):

    ABL_TRAIN_N=5000            sottoinsieme casuale di N api del training set
    ABL_BACKBONE=resnet18       backbone al posto di quello del recipe

    ABL_TRAIN_N=5000 python experiments/single_bee/main_cfa.py
    ABL_TRAIN_N=5000 python experiments/plot_heatmaps.py single_bee cfa

Con almeno una delle due impostata checkpoint e heatmap vanno in
checkpoints|heatmaps/ablation/<modello>/<tag>/ (tag es. "n5000",
"resnet18"), senza toccare quelli di single_bee_val/. I sottoinsiemi sono
annidati (prime N api di una permutazione fissa con SUBSET_SEED), così
n500 ⊂ n5000 ⊂ n20000. Il resto del recipe (step, lr, layer allo stesso
stride) non viene ri-ottimizzato.
"""

import os

import numpy as np

TRAIN_N = int(os.environ.get("ABL_TRAIN_N") or 0)
BACKBONE = os.environ.get("ABL_BACKBONE") or None
SUBSET_SEED = 0
ACTIVE = bool(TRAIN_N or BACKBONE)


def tag():
    parts = ([BACKBONE] if BACKBONE else []) + ([f"n{TRAIN_N}"] if TRAIN_N else [])
    return "_".join(parts)


def backbone(default):
    return BACKBONE or default


def dirs(repo_root, model_name, checkpoint_dir, output_dir):
    """(CHECKPOINT_DIR, OUTPUT_DIR): quelli ufficiali o quelli dell'ablation."""
    if not ACTIVE:
        return checkpoint_dir, output_dir
    return (repo_root / "checkpoints" / "ablation" / model_name / tag(),
            repo_root / "heatmaps" / "ablation" / model_name / tag())


def subset_train(dataset):
    """Riduce in place un VarroaDataset TRAIN alle prime TRAIN_N api della permutazione fissa."""
    if not TRAIN_N:
        return dataset
    if TRAIN_N > len(dataset):
        raise ValueError(f"ABL_TRAIN_N={TRAIN_N} > {len(dataset)} api di training")
    idx = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(dataset))[:TRAIN_N])
    dataset.samples = dataset.samples.iloc[idx].reset_index(drop=True)
    dataset.image_paths = [dataset.image_paths[i] for i in idx]
    dataset.labels = [dataset.labels[i] for i in idx]
    dataset.mask_paths = [dataset.mask_paths[i] for i in idx]
    print(f"[ablation] training ridotto a {TRAIN_N} api (seed {SUBSET_SEED})")
    return dataset
