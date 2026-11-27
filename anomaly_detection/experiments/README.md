# Varroa experiments

```
experiments/
├── plot_heatmaps.py   final metrics + heatmaps from the best checkpoint (all models)
├── single_bee/        single-bee crops (512x512), VarroaDataset
│   ├── main_<model>.py              training + saving the best checkpoint
│   ├── sweep_<model>.py             hyperparameter sweep (logs in logs/<model>_sweep/)
│   ├── make_val_split.py              val / test_heldout split of the final dataset
│   ├── ablation.py, summarize_ablation.py  training-set size + backbone ablation
│   ├── measure_efficiency.py          inference time, memory, parameters
│   ├── score_distributions.py, hard_cases.py  score analysis and hard cases
│   ├── diagnose_fastflow_varroa.py    FastFlow diagnostics
│   └── main_stfpm_clad.py             continual STFPM (FT / multi-task / replay)
└── full_image/        whole images with many bees, sliding window
    ├── main_<model>_full_image.py   training + saving the best checkpoint
    └── dinomaly_from_single_bee.py    full-image heatmaps with the single-bee Dinomaly weights
```

Single-bee models: cfa, dinomaly, fastflow, ganomaly, padim, paste, patchcore,
rd4ad, stfpm, structcore, supersimplenet.
Full-image models: cfa, dinomaly, ganomaly, padim, paste, patchcore, rd4ad, stfpm.

## Workflow

From the repository root:

```
# 1. training: saves the best configuration
python experiments/single_bee/main_dinomaly.py
python experiments/full_image/main_rd4ad_full_image.py

# 2. final metrics and heatmaps from the best checkpoint (same command for every model)
python experiments/plot_heatmaps.py single_bee dinomaly
python experiments/plot_heatmaps.py single_bee all
python experiments/plot_heatmaps.py full_image rd4ad
```

Choosing the best checkpoint (models trained for epochs): every `EVAL_INTERVAL` epochs
and at the last one, the metrics are computed on the test set (full-image: on the first
`EVAL_IMAGES_PER_CLASS` clean and varroa images) and the epoch with the highest
`img_roc_auc` is kept, with ties broken by the highest `pxl_avg_prec`. Since the choice
uses the test set, the metrics of the best checkpoint are optimistic.
PaDiM, PatchCore and StructCore have no epochs: they save the state estimated in a
single pass. Full-image PaDiM does not save the Gaussians (~20 GB at 512 px):
both the script and `plot_heatmaps.py` redo the fitting.

## Output

Paths are relative to the repository root, regardless of the folder the
script is launched from:

```
checkpoints/
├── single_bee/<model>/   best.pth, config.json (hyperparameters, epoch and metrics of the best checkpoint)
└── full_image/<model>/
heatmaps/
├── single_bee/<model>/   sample_*.png, score_distribution.png, results.json
└── full_image/<model>/
```

In the single-bee scripts, GT masks, pixel metrics and heatmaps are always at
512x512 (the crop resolution), whatever the model's input resolution:
the map is resized back to 512x512 before computing the metrics.
Single-bee heatmaps are normalised globally and anchored to the
pixel-F1-optimal threshold (`moviad.utilities.evaluation.report`), the same for all
models. The full-image scripts use the sliding window and each model's own
figures.
