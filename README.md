# Unsupervised Learning for Honeybee Health Monitoring

Code for the master's thesis on monitoring honeybee colonies from photographs of hive frames.
The repository goes from raw frame photos to per-bee crops and colony-level indicators, and uses
those crops to detect the *Varroa destructor* mite with unsupervised anomaly detection.

<!-- TODO: thesis title, university, supervisors, year and a link to the PDF once it is published. -->

```
 frame photos
      │
      ▼
 ┌────────────────────────── segmentation/ ──────────────────────────┐
 │ 1. SAM 3 pseudo-labels ("bee" text prompt, multi-scale tiling)     │
 │ 2. YOLO26s-seg fine-tuned on the pseudo-labels                     │
 │ 3. single-bee extraction, cleaning, 512×512 crops, head-up         │
 │ 4. bee_al: active learning + label propagation on DINOv2 features  │
 │    phase 1: usable vs. badly segmented crops                       │
 │    phase 2: young / adult / drone / sick                           │
 │ 5. per-batch colony counts with bootstrap intervals                │
 │ 6. synthetic varroa pasted onto healthy crops (with pixel masks)   │
 └───────────────────────────────────────────────────────────────────┘
      │  healthy crops (train) + synthetic-varroa crops (val/test)
      ▼
 ┌────────────────────── anomaly_detection/ ─────────────────────────┐
 │ 11 unsupervised anomaly-detection models (MoViAD library fork):   │
 │ PatchCore, CFA, Dinomaly, PaDiM, STFPM, FastFlow, PaSTe, RD4AD,   │
 │ GANomaly, SuperSimpleNet, StructCore                              │
 │ hyper-parameter sweeps, train-size/backbone ablation, inference   │
 │ cost, score analysis, full-image (sliding window) experiments     │
 └───────────────────────────────────────────────────────────────────┘
```

## Repository layout

| Folder | Content | Details |
|---|---|---|
| [`segmentation/`](segmentation/) | Bee segmentation, crop extraction and cleaning, the `bee_al` active-learning hub, synthetic varroa generation | [`segmentation/README.md`](segmentation/README.md) |
| [`anomaly_detection/`](anomaly_detection/) | Fork of the MoViAD library and the varroa anomaly-detection experiments | [`anomaly_detection/README.md`](anomaly_detection/README.md) |
| `segmentation/docs/` | Written summaries of the methods and results (segmentation ablation, colony counts, phone photos, synthetic varroa) | |
| `*/logs/` | Logs, metrics (JSON/CSV) and summary reports of the runs reported in the thesis | |
| `segmentation/visualizations/` | Qualitative figures | |

The two folders are independent Python projects with their own dependencies. They communicate only
through the dataset on disk: `segmentation/` writes the single-bee crops and the synthetic-varroa
split, and `anomaly_detection/` reads them.

## Data

The bee photographs are **not** included in this repository. Every script resolves its default input and
output folders from two environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `BEES_DATA_ROOT` | Root of the dataset (`batch_3/ … batch_7/` with the raw photos, `processed/` with everything the pipeline produces) | `data/DatasetApi_Ceschi` |
| `BEE_AL_WORKDIR` | Labels, embeddings and model versions of the active-learning hub | `data/bee_al` |

```bash
export BEES_DATA_ROOT=/path/to/DatasetApi_Ceschi
export BEE_AL_WORKDIR=/path/to/bee_al_workdir
```

Most scripts also accept the folders as command-line arguments or have them as constants at the top of
the file. Model weights (`*.pt`, `*.pth`) and the `data/` folder are ignored by git.

<!-- TODO: say whether and where the dataset / trained weights are available (e.g. on request, Zenodo). -->

## Quick start

Segmentation and active learning (Python ≥ 3.10):

```bash
cd segmentation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # unit tests of bee_al and the dataset scripts
```

Anomaly detection (Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/)):

```bash
cd anomaly_detection
uv sync
uv run python experiments/single_bee/main_patchcore.py
uv run python experiments/plot_heatmaps.py single_bee patchcore
```

See the README of each folder for the full workflow.

## Main results

**Segmentation** (YOLO26s-seg trained at 1376 px on SAM 3 pseudo-labels, test set of 119 photos, inference
at 2048 px): mask mAP50-95 0.692, mAP50 0.927, precision 0.931, recall 0.881. The metrics are measured
against the SAM 3 pseudo-labels, not a manual ground truth. Full ablation:
[`segmentation/docs/ablation_sam3.md`](segmentation/docs/ablation_sam3.md).

**Varroa anomaly detection** (512×512 single-bee crops; trained on 102,303 healthy bees, checkpoint chosen on
the validation split, evaluated once on a held-out test of 957 bees with synthetic varroa and 5,000 healthy
bees):

| Model | Image AUROC | Pixel AP |
|---|---:|---:|
| PatchCore | 0.999 | 0.777 |
| Dinomaly | 0.996 | 0.768 |
| CFA | 0.996 | 0.755 |
| PaDiM | 0.989 | 0.521 |
| STFPM | 0.983 | 0.534 |
| FastFlow | 0.982 | 0.469 |
| PaSTe | 0.965 | 0.393 |
| RD4AD | 0.953 | 0.349 |

The anomalies are synthetic, so these numbers measure how well each model finds a varroa-like object on a
bee, not field performance. Training-set size, backbone and inference-cost ablations:
[`anomaly_detection/logs/ablation/summary/report.md`](anomaly_detection/logs/ablation/summary/report.md).
Experimental protocol: [`anomaly_detection/single_bee_protocol.md`](anomaly_detection/single_bee_protocol.md).

## Credits

- `anomaly_detection/src/moviad` is a fork of **MoViAD**, a modular visual anomaly detection library by
  Francesco Borsatti, Manuel Barusco, Davide Petrovic, Ruslan Nuriev, Davide Frizzo and colleagues. The full
  history of the library is kept in this repository. The varroa dataset classes (`datasets/varroa/`), the
  checkpoint, report and full-image helpers in `utilities/evaluation/` and the changes to the models for
  single-bee crops were added for this thesis.
- Segmentation uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
  [SAM 3](https://huggingface.co/facebook/sam3) and [LangSAM](https://github.com/luca-medeiros/lang-segment-anything);
  the active-learning hub uses frozen [DINOv2](https://github.com/facebookresearch/dinov2) features.

## License

Released under the Apache License 2.0 (see [`LICENSE`](LICENSE)), the same license as MoViAD.

## Citation

<!-- TODO: fill in once the thesis is published. -->

```bibtex
@mastersthesis{ballarin_honeybees,
  author = {Ballarin, Tommaso},
  title  = {Deep Learning for Apicultural Health Assessement: Honeybee Segmentation and Visual Anomaly Detection},
  school = {University of Padova},
  year   = {2026}
}
```
