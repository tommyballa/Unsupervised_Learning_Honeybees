# Varroa detection with unsupervised visual anomaly detection

The thesis experiments on detecting the *Varroa destructor* mite on single-bee crops, built on a fork of the
**MoViAD** library (`src/moviad`, described [below](#about-moviad)). The models learn only from healthy bees and
are evaluated on bees with synthetic varroa produced by [`../segmentation`](../segmentation)
(`build_varroa_split.py`).

## Setup

Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

The dataset is read from `$BEES_DATA_ROOT` (default `./data/DatasetApi_Ceschi`, see the
[top-level README](../README.md#data)). The single-bee experiments use
`$BEES_DATA_ROOT/processed/single_bee3+4+5+6_SAM3_varroa_split`:

```
single_bee3+4+5+6_SAM3_varroa_split/
├── train/normal/        healthy bees (512×512 crops)
├── test/normal/         healthy test bees
├── test/anomalous/      test bees with synthetic varroa
├── test/masks/          pixel masks of the varroa
└── test/labels.csv      bee_id, cluster, label, num_varroa, image_path, mask_path
```

`experiments/single_bee/make_val_split.py` splits the test part into a validation set (used to choose
checkpoints and thresholds) and a held-out test set (used once for the final metrics).

## Experiments

```bash
# train one model and keep the best checkpoint (chosen on the validation split)
uv run python experiments/single_bee/main_patchcore.py
# final metrics and heatmaps on the held-out test
uv run python experiments/plot_heatmaps.py single_bee patchcore
uv run python experiments/plot_heatmaps.py single_bee all
```

| Folder / file | Content |
|---|---|
| [`experiments/README.md`](experiments/README.md) | All scripts, the training/evaluation flow and the output folders |
| `experiments/single_bee/` | 11 models on 512×512 crops: `main_<model>.py`, `sweep_<model>.py`, ablation, efficiency, score analyses |
| `experiments/full_image/` | Whole frame photos with many bees, sliding window |
| [`recap_protocollo_single_bee.md`](recap_protocollo_single_bee.md) | Experimental protocol: splits, model selection, thresholds |
| [`logs/ablation/summary/report.md`](logs/ablation/summary/report.md) | Training-set size, backbone and inference-cost ablation |
| `logs/<model>_sweep/` | Hyper-parameter sweeps (one JSON per configuration) |
| `logs/val/`, `logs/score_dist/` | Final runs on the validation/held-out protocol and score distributions |

Checkpoints (`checkpoints/`) and heatmaps (`heatmaps/`) are written next to this file and are not versioned.

## Changes to MoViAD made for the thesis

- `datasets/varroa/`: `VarroaDataset` (single-bee crops with pixel masks) and a domain-shift variant.
- `utilities/evaluation/`: best-checkpoint tracking, reports and full-image (sliding window) evaluation.
- Models adapted to single-bee crops (bee-masked losses and maps, map blur, max-of-map image score), see the
  `main_<model>.py` scripts for the per-model recipe.

---

# About MoViAD

<p align="center">
<img src="https://github.com/user-attachments/assets/93cf614e-2d18-4aa3-b670-80206c03bb7c" alt="MoViAD logo" style="width:300px;"/>
</p>

The text below is the original README of the MoViAD library. Paths in it are relative to this folder.

## Objective

MoViAD is an Open Source library for easy and modular Visual Anomaly Detection, built for industrial and research purposes. 
The library contains some State-of-the-Art models with their trainers, standard datasets, evaluator for calculating standard metrics, and a feature extractor.
The library structure is totally modular, allowing an easy isolation of the components needed for your project. 
The library will support different scenarios (see also below):
* Unsupervised anomaly detection ✅
* Contaminated anomaly detection ✅
* Semi-supervised anomaly detection 🚧 (Work in progress)
* Few shot anomaly detection 🚧 (Work in progress)
* Continual anomaly detection 🚧 (Work in progress)


## How to use the library

The library follows this structure:
- inside the <code>/models/model_name</code> directory is present the code of anomaly detection models
- inside the <code>/trainers</code> directory is present the code for training the anomaly detection models
- inside the <code>/datasets</code> directory is present the code for the anomaly detection datasets that must be used
- inside the <code>/utilies</code> directory is present the code for anomaly detection utilities

## Execution example

Inside the <code>/main_scripts</code> directory are present some execution scripts for training and testing the AD models. 

For example, for training patchcore: 

```bash
uv run main_scripts/main_patchcore.py --mode train --dataset_path /home/datasets/mvtec --category pill --backbone mobilenet_v2 --ad_layers features.4 features.7 features.10 --device cuda:0 --save_path ./patch.pt 
```

For every main script all its parameters are documented. 

**AD Models**

- PatchCore: [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html), [code](https://github.com/amazon-science/patchcore-inspection)
- CFA [paper](https://ieeexplore.ieee.org/abstract/document/9839549), [code](https://github.com/sungwool/CFA_for_anomaly_localization)
- Student-Teacher Feature Pyramid: [paper](https://arxiv.org/abs/2103.04257), [code](https://github.com/gdwang08/STFPM)
- PaSTe: [paper](https://arxiv.org/abs/2103.04257)
- PaDiM: [paper with code](https://paperswithcode.com/paper/padim-a-patch-distribution-modeling-framework)
- FastFlow: [paper](https://arxiv.org/abs/2111.07677)
- GANomaly: [paper](https://arxiv.org/abs/1805.06725)
- SuperSimpleNet: [paper](https://arxiv.org/abs/2408.03143)
- RD4AD: [paper](https://arxiv.org/abs/2201.10703)

**Feature Extraction Backbones**

- MobileNet: [link](https://paperswithcode.com/paper/mobilenets-efficient-convolutional-neural)
- PhiNet: [link](https://paperswithcode.com/paper/phinets-a-scalable-backbone-for-low-power-ai)
- MicroNet: [link](https://paperswithcode.com/paper/micronet-improving-image-recognition-with)
- MCUNet: [link](https://paperswithcode.com/paper/mcunet-tiny-deep-learning-on-iot-devices)
- WideResnet

**Datasets**

- MVTecAD: [link](https://paperswithcode.com/dataset/mvtecad)
- VisA: [link](https://paperswithcode.com/dataset/visa)
- RealIAD: [paper](https://arxiv.org/abs/2403.12580)
- MIIC Dataset: [link](https://github.com/wenbihan/MIIC-IAD) 

**Scenarios**

- Unsupervised: standard VAD scenario where the model is trained on a dataset of normal samples and evaluated on a dataset of normal and abnormal samples.
- Contaminated: model trained on a dataset of both normal and abnormal samples, a certain percentage of abnormal samples (train set contamination).
- Semi-supervised: model trained on a dataset of mostly normal samples with a small percentage of labeled abnormal samples, leveraging both types of data to improve anomaly detection performance.
- Few-shot: model trained on a dataset of normal samples and a few abnormal samples which are labeled.
- Continual: the model is trained on a specific subset, called task, and then updated with new tasks. The model should be able to perform well on all seen tasks.

## How to Install

Inside the main repository directory, after installing [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

## Contribute

If you want to contribute to the repository, follow the present code structure: 
- inside the <code>/models/model_name</code> directory put the code for possible new anomaly detection models
- inside the <code>/trainers</code> directory put the code for training an anomaly detection model
- inside the <code>/datasets</code> directory put the code for possible new anomaly detection datasets that must be used

Every contribution must be open with a pull request. 

**Execute tests**

Please, before opening a pull request, execute the tests with the command, and add new tests if you added new functionalities:

```bash
uv run pytest -v -s
```

**Build Documentation**
To build the documentation locally, use the following commands:

```bash
cd code-files/moviad/docs

uv run sphinx-build -b html . _build/html
```


## Citations

If you use MoViAD in your work, please cite us! 🤗

Works that use MoViAD: 
- "PaSTe: Improving the efficiency of visual anomaly detection at the edge", Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto. [paper](https://arxiv.org/abs/2103.04257)
- "From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithm", Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto. [paper](https://arxiv.org/pdf/2502.18328?)
