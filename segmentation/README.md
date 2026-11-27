# Segmentation, crop extraction and active learning

From photographs of hive frames to clean single-bee crops, their classes (usable / young / adult / drone /
sick) and per-batch colony counts. This folder also builds the synthetic-varroa dataset used by
[`../anomaly_detection`](../anomaly_detection).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

`requirements.txt` covers the `bee_al` package, the dataset scripts and the tests. The segmentation models
need some extra packages:

| Step | Extra packages |
|---|---|
| YOLO training / inference | `ultralytics` |
| SAM 3 pseudo-labels | `transformers` with access to the gated `facebook/sam3` model (`hf auth login`) |
| LangSAM pseudo-labels (first ablation) | `lang-sam` |
| UMAP embedding maps | `umap-learn`, `pandas` |
| Faster k-NN in the hub (optional) | `faiss-cpu` |

Dataset folders are resolved from `BEES_DATA_ROOT` and `BEE_AL_WORKDIR` (see the
[top-level README](../README.md#data)); scripts are run from this folder.

## Pipeline

| # | Step | Script(s) | Output under `$BEES_DATA_ROOT/processed/` |
|---|---|---|---|
| 1 | Segmentation pseudo-labels with SAM 3 (text prompt "bee", adaptive scale, multi-scale tiling, mask NMS) | `generate_labels_sam3.py` | `bee_batch3+4+5+6_SAM3/` |
| 1b | Earlier pseudo-labels with LangSAM (batches 3+4) | `generate_labels.py` | `train_batch*_labelled/` |
| 2 | Train / val / test split by photo | `split_dataset_sam3.py` | `bee_batch3+4+5+6_SAM3_split/` |
| 3 | Fine-tune YOLO26s-seg and evaluate | `train_yolo.py`, `test_yolo.py`, `predict_random_samples.py`, `data_sam3.yaml` | `runs/` |
| 4 | Extract single bees with the chosen model (per-photo brightness/contrast normalisation) | `extract_anomaly_dataset.py` | `…/raw_bees/` |
| 5 | Clean the crops (blur, area, fill, aspect filters), 512×512, long axis vertical | `dataset_cleaning.py` | `…/cleaned_bees/` |
| 5b | Head-up orientation with a linear classifier on DINOv2 features | `bee_orientation.py` (labels in `config/head_orientation_labels.json`) | |
| 6 | Active learning, phase 1 (usable vs. badly segmented) and phase 2 (demographic classes and sick bees) | `bee_al/`, `scripts/`, `active_learning_tool/app.py`, `config/al_*.yaml` | `$BEE_AL_WORKDIR/` |
| 7 | Snapshot, blind audit, corrected estimates and per-batch counts | `scripts/export_snapshot.py`, `scripts/audit_report.py`, `scripts/batch_counts.py` | |
| 8 | Synthetic varroa: on full frames, then extracted per bee | `varroa_generator.py`, `extract_synthetic_anomaly_dataset.py` | `…/synthetic_varroa_output/` |
| 8b | Synthetic varroa pasted directly on clean crops, train/test split for anomaly detection | `varroa_on_crops.py`, `build_varroa_split.py` | `single_bee3+4+5+6_SAM3_varroa_split/` |

`dataset_clustering.py` is an earlier baseline that clusters the crops on DINOv2 features (PCA, k-means and an
SVM trained on the labels of the first annotation tool).

## Active-learning hub (`bee_al`)

A web app to label bee crops efficiently: frozen DINOv2 embeddings → memory bank → weighted k-NN and label
propagation → query batches (uncertainty, diversity, novelty) → labelling in the browser → final clusters.
Labelling runs in phases, each with its own config, labels and model versions:

```bash
# phase 1
cd active_learning_tool && python -u app.py
# phase 2, prepared from a phase-1 snapshot
python scripts/new_stage.py --snapshot fase1_finale --keep "Segmentata Bene" --name fase2
cd active_learning_tool && BEE_AL_CONFIG=config/al_fase2.yaml python -u app.py
```

The full workflow — stopping criteria, snapshots, blind audit, bias-corrected estimates and colony
indicators — is documented in [`bee_al/README.md`](bee_al/README.md).

## Folder layout

```
segmentation/
├── bee_al/                 active-learning package (engine, hub, audit, estimates, export)
├── active_learning_tool/   entry point of the hub web app
├── scripts/                command-line tools around bee_al (embeddings, stages, snapshots, reports)
├── config/                 hub configurations per phase, head-orientation labels
├── tests/                  pytest suite
├── docs/                   method and result summaries
├── logs/                   logs and evaluation scripts of the runs reported in the thesis
├── visualizations/         qualitative figures
├── *.py                    pipeline scripts (table above)
└── data*.yaml              YOLO dataset definitions
```

## Documentation

| Document | Content |
|---|---|
| [`docs/pipeline_results.md`](docs/pipeline_results.md) | Whole pipeline on batches 3-6: data, SAM 3 labels, YOLO, extraction, cleaning, active learning, counts |
| [`docs/ablation_sam3.md`](docs/ablation_sam3.md) | YOLO26s-seg ablation on SAM 3 labels (resolution, augmentation, freeze, seed) |
| [`docs/yolo_finetuning_langsam.md`](docs/yolo_finetuning_langsam.md) | First ablation on LangSAM labels (batches 3+4) |
| [`docs/phone_photos.md`](docs/phone_photos.md) | Generalisation to frame photos taken with a phone |
| [`docs/synthetic_varroa_on_crops.md`](docs/synthetic_varroa_on_crops.md) | Synthetic varroa generator on single-bee crops |
