# Esperimenti varroa

```
experiments/
├── single_bee/    crop della singola ape (512x512), VarroaDataset
│   ├── main_<modello>.py              training + valutazione + heatmap
│   ├── diagnose_fastflow_varroa.py    diagnostica FastFlow
│   ├── find_top_anomalies_fastflow.py top-N anomalie con FastFlow
│   ├── main_patchcore_outliners.py    outlier nel training set con PatchCore
│   └── main_stfpm_clad.py             STFPM continual (FT / multi-task / replay)
└── full_image/    immagini intere con molte api, sliding window
    └── main_<modello>_full_image.py
```

Modelli single-bee: cfa, dinomaly, fastflow, ganomaly, padim, paste, patchcore,
rd4ad, stfpm, structcore, supersimplenet.
Modelli full-image: cfa, dinomaly, ganomaly, padim, paste, patchcore, rd4ad, stfpm.

Esecuzione dalla root della repo, per esempio:

```
python experiments/single_bee/main_dinomaly.py
python experiments/full_image/main_rd4ad_full_image.py
```

## Output

I percorsi sono relativi alla root della repo, indipendentemente dalla cartella
da cui si lancia lo script:

```
heatmaps/
├── single_bee/<modello>/      sample_*.png, score_distribution.png, results.json
└── full_image/<modello>/
checkpoints/
├── single_bee/<modello>.pth
└── full_image/<modello>.pth
```

Negli script single-bee maschere GT, metriche pixel e heatmap sono sempre a
512x512 (la risoluzione dei crop), qualunque sia la risoluzione di input del
modello: la mappa viene riportata a 512x512 prima di calcolare le metriche.
Le heatmap sono normalizzate globalmente e ancorate alla soglia pixel
F1-ottimale (`moviad.utilities.evaluation.report`), uguale per tutti i modelli.
