# Esperimenti varroa

```
experiments/
├── plot_heatmaps.py   metriche finali + heatmap dalla best (tutti i modelli)
├── single_bee/        crop della singola ape (512x512), VarroaDataset
│   ├── main_<modello>.py              training + salvataggio della best
│   ├── diagnose_fastflow_varroa.py    diagnostica FastFlow
│   ├── find_top_anomalies_fastflow.py top-N anomalie con FastFlow
│   ├── main_patchcore_outliners.py    outlier nel training set con PatchCore
│   └── main_stfpm_clad.py             STFPM continual (FT / multi-task / replay)
└── full_image/        immagini intere con molte api, sliding window
    └── main_<modello>_full_image.py   training + salvataggio della best
```

Modelli single-bee: cfa, dinomaly, fastflow, ganomaly, padim, paste, patchcore,
rd4ad, stfpm, structcore, supersimplenet.
Modelli full-image: cfa, dinomaly, ganomaly, padim, paste, patchcore, rd4ad, stfpm.

## Flusso

Dalla root della repo:

```
# 1. training: salva la configurazione migliore
python experiments/single_bee/main_dinomaly.py
python experiments/full_image/main_rd4ad_full_image.py

# 2. metriche finali e heatmap dalla best (stesso comando per tutti i modelli)
python experiments/plot_heatmaps.py single_bee dinomaly
python experiments/plot_heatmaps.py single_bee all
python experiments/plot_heatmaps.py full_image rd4ad
```

Scelta della best (modelli con training a epoche): ogni `EVAL_INTERVAL` epoche
e all'ultima si calcolano le metriche sul test set (full-image: sulle prime
`EVAL_IMAGES_PER_CLASS` immagini pulite e con varroa) e si tiene l'epoca con
`img_roc_auc` più alto, a parità con `pxl_avg_prec` più alto. Poiché la scelta
usa il test set, le metriche della best sono ottimistiche.
PaDiM, PatchCore e StructCore non hanno epoche: salvano lo stato stimato in un
solo passaggio. PaDiM full-image non salva le gaussiane (~20 GB a 512 px):
sia lo script sia `plot_heatmaps.py` rifanno il fitting.

## Output

Percorsi relativi alla root della repo, indipendentemente dalla cartella da
cui si lancia lo script:

```
checkpoints/
├── single_bee/<modello>/   best.pth, config.json (iperparametri, epoca e metriche della best)
└── full_image/<modello>/
heatmaps/
├── single_bee/<modello>/   sample_*.png, score_distribution.png, results.json
└── full_image/<modello>/
```

Negli script single-bee maschere GT, metriche pixel e heatmap sono sempre a
512x512 (la risoluzione dei crop), qualunque sia la risoluzione di input del
modello: la mappa viene riportata a 512x512 prima di calcolare le metriche.
Le heatmap single-bee sono normalizzate globalmente e ancorate alla soglia
pixel F1-ottimale (`moviad.utilities.evaluation.report`), uguale per tutti i
modelli. Gli script full-image usano la sliding window e le figure di ciascun
modello.
