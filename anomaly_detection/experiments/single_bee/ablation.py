"""
Ablation of the single-bee models: number of training bees and backbone.

The main_<model>.py scripts read two environment variables (unset = official
configuration, unchanged):

    ABL_TRAIN_N=5000            random subset of N bees from the training set
    ABL_BACKBONE=resnet18       backbone replacing the recipe one

    ABL_TRAIN_N=5000 python experiments/single_bee/main_cfa.py
    ABL_TRAIN_N=5000 python experiments/plot_heatmaps.py single_bee cfa

With at least one of the two set, checkpoints and heatmaps go to
checkpoints|heatmaps/ablation/<model>/<tag>/ (tag e.g. "n5000",
"resnet18"), without touching those in single_bee_val/. The subsets are
nested (first N bees of a fixed permutation with SUBSET_SEED), so
n500 ⊂ n5000 ⊂ n20000. The rest of the recipe (steps, lr, layers at the same
stride) is not re-optimised.
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
    """(CHECKPOINT_DIR, OUTPUT_DIR): the official ones or the ablation ones."""
    if not ACTIVE:
        return checkpoint_dir, output_dir
    return (repo_root / "checkpoints" / "ablation" / model_name / tag(),
            repo_root / "heatmaps" / "ablation" / model_name / tag())


def subset_train(dataset):
    """Reduces a TRAIN VarroaDataset in place to the first TRAIN_N bees of the fixed permutation."""
    if not TRAIN_N:
        return dataset
    if TRAIN_N > len(dataset):
        raise ValueError(f"ABL_TRAIN_N={TRAIN_N} > {len(dataset)} training bees")
    idx = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(dataset))[:TRAIN_N])
    dataset.samples = dataset.samples.iloc[idx].reset_index(drop=True)
    dataset.image_paths = [dataset.image_paths[i] for i in idx]
    dataset.labels = [dataset.labels[i] for i in idx]
    dataset.mask_paths = [dataset.mask_paths[i] for i in idx]
    print(f"[ablation] training reduced to {TRAIN_N} bees (seed {SUBSET_SEED})")
    return dataset
