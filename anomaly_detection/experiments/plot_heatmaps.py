"""
Metriche finali e heatmap dalla configurazione migliore salvata dal training.

    python experiments/plot_heatmaps.py single_bee dinomaly
    python experiments/plot_heatmaps.py single_bee dinomaly patchcore rd4ad
    python experiments/plot_heatmaps.py single_bee all
    python experiments/plot_heatmaps.py full_image rd4ad

Il training (experiments/<setting>/main_<modello>*.py) salva la best in
checkpoints/<setting>/<modello>/ (best.pth + config.json); questo script la
ricarica e scrive in heatmaps/<setting>/<modello>/.

single_bee: stessa pipeline per tutti i modelli
            (moviad.utilities.evaluation.report.evaluate_and_report):
            metriche con AU-PRO a 512x512, heatmap normalizzate globalmente e
            ancorate alla soglia pixel F1-ottimale, score_distribution.png e
            results.json (con epoca e metriche di selezione della best).
full_image: sliding window e figure dello script del modello (funzione
            plot_heatmaps del modulo).
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXPERIMENTS = Path(__file__).resolve().parent
MODULE_PATTERN = {"single_bee": "main_{}", "full_image": "main_{}_full_image"}
# script della cartella single_bee che non sono modelli con una best da plottare
NOT_MODELS = {"stfpm_clad", "patchcore_outliners"}


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
    """Modello con la configurazione migliore e le informazioni salvate in config.json."""
    from moviad.utilities.evaluation.checkpoints import load_checkpoint

    if hasattr(mod, "load_best"):          # es. PaDiM full-image: rifà il fitting
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
    # soglie F1 e normalizzazione delle heatmap dal validation set, se lo script ne ha uno
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
    parser.add_argument("models", nargs="+", help="nomi dei modelli (es. dinomaly) oppure 'all'")
    parser.add_argument("--num-heatmaps", type=int, default=None,
                        help="solo single_bee: numero di heatmap (metà sane, metà con varroa)")
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
            print(f"  salto {name}: {e}")
            continue
        if info.get("best_epoch"):
            print(f"Best: epoca {info['best_epoch']} — metriche di selezione: {info.get('metrics')}")

        if args.setting == "single_bee":
            plot_single_bee(mod, model, info, device, args.num_heatmaps)
        else:
            mod.plot_heatmaps(model, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
