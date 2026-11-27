"""
Final metrics and heatmaps from the best configuration saved by training.

    python experiments/plot_heatmaps.py single_bee dinomaly
    python experiments/plot_heatmaps.py single_bee dinomaly patchcore rd4ad
    python experiments/plot_heatmaps.py single_bee all
    python experiments/plot_heatmaps.py full_image rd4ad

Training (experiments/<setting>/main_<model>*.py) saves the best checkpoint in
checkpoints/<setting>/<model>/ (best.pth + config.json); this script
reloads it and writes to heatmaps/<setting>/<model>/.

single_bee: same pipeline for all models
            (moviad.utilities.evaluation.report.evaluate_and_report):
            metrics with AU-PRO at 512x512, heatmaps normalised globally and
            anchored to the pixel-F1-optimal threshold, score_distribution.png and
            results.json (with epoch and selection metrics of the best checkpoint).
full_image: sliding window and figures from the model's script (the module's
            plot_heatmaps function).
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXPERIMENTS = Path(__file__).resolve().parent
MODULE_PATTERN = {"single_bee": "main_{}", "full_image": "main_{}_full_image"}
# scripts in the single_bee folder that are not models with a best checkpoint to plot
NOT_MODELS = {"stfpm_clad"}


def available_models(setting):
    prefix, suffix = MODULE_PATTERN[setting].split("{}")
    names = []
    for path in sorted((EXPERIMENTS / setting).glob("main_*.py")):
        stem = path.stem
        if stem.startswith(prefix) and stem.endswith(suffix):
            name = stem[len(prefix):len(stem) - len(suffix)]
            if name not in NOT_MODELS:
                names.append(name)
    return names


def load_best(mod, device):
    """Model with the best configuration and the information saved in config.json."""
    from moviad.utilities.evaluation.checkpoints import load_checkpoint

    if hasattr(mod, "load_best"):          # e.g. full-image PaDiM: redoes the fitting
        return mod.load_best(device)
    model = mod.build_model(device)
    state, info = load_checkpoint(mod.CHECKPOINT_DIR, map_location="cpu")
    mod.load_model_state(model, state)
    return model, info


def plot_single_bee(mod, model, info, device, num_heatmaps):
    from moviad.datasets.varroa.varroa_dataset import VarroaDataset
    from moviad.utilities.configurations import Split
    from moviad.utilities.evaluation.report import evaluate_and_report

    batch_size = getattr(mod, "TEST_BATCH_SIZE", mod.BATCH_SIZE)
    test_dataset = VarroaDataset(mod.dataset_args(), "varroa", Split.TEST)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=mod.NUM_WORKERS, pin_memory=True,
    )
    # F1 thresholds and heatmap normalisation from the validation set, if the script has one
    val_loader = None
    if hasattr(mod, "VAL_DIR"):
        val_loader = DataLoader(
            VarroaDataset(mod.dataset_args(mod.VAL_DIR), "varroa", Split.TEST), batch_size=batch_size,
            shuffle=False, num_workers=mod.NUM_WORKERS, pin_memory=True,
        )
    epoch = info.get("best_epoch")
    title = f"{mod.MODEL_NAME} single-bee" + (f" (best: epoch {epoch})" if epoch else "")
    evaluate_and_report(
        model, test_dataset, test_loader, device, mod.OUTPUT_DIR, title=title,
        num_heatmaps=num_heatmaps or mod.NUM_HEATMAPS, seed=mod.SEED,
        map_postprocess=getattr(mod, "map_postprocess", None),
        raw_label=getattr(mod, "RAW_LABEL", "Raw anomaly map"),
        extra_info={k: v for k, v in info.items() if k != "config"},
        val_loader=val_loader,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", choices=list(MODULE_PATTERN))
    parser.add_argument("models", nargs="+", help="model names (e.g. dinomaly) or 'all'")
    parser.add_argument("--num-heatmaps", type=int, default=None,
                        help="single_bee only: number of heatmaps (half healthy, half with varroa)")
    args = parser.parse_args()

    sys.path.insert(0, str(EXPERIMENTS / args.setting))
    models = available_models(args.setting) if args.models == ["all"] else args.models
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    for name in models:
        print(f"\n{'#'*60}\n# {args.setting} / {name}\n{'#'*60}")
        mod = importlib.import_module(MODULE_PATTERN[args.setting].format(name))
        mod.set_seed(mod.SEED)
        try:
            model, info = load_best(mod, device)
        except FileNotFoundError as e:
            print(f"  skipping {name}: {e}")
            continue
        if info.get("best_epoch"):
            print(f"Best: epoch {info['best_epoch']} — selection metrics: {info.get('metrics')}")

        if args.setting == "single_bee":
            plot_single_bee(mod, model, info, device, args.num_heatmaps)
        else:
            mod.plot_heatmaps(model, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
