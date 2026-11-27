# Single-bee varroa anomaly detection: experimental protocol

This document summarises the dataset split, the model-selection protocol and the final results of the anomaly detection models on single-bee crops with synthetic varroa. The way the synthetic varroa and the dataset were generated is described in [synthetic_varroa_on_crops.md](../segmentation/docs/synthetic_varroa_on_crops.md).

## 1. Dataset and split

**Final dataset:** `single_bee3+4+5+6_SAM3_varroa_split` (512×512 crops segmented with SAM3).

- **Train:** 102,303 healthy bees (Young/Adult/Drone/Sick).
- **Original test:** 24,297 healthy + 1,279 with synthetic varroa (1 or 2 per bee), with pixel-level masks.

**Reduced test (used until 18 October):** 5,000 healthy bees sampled with seed 0 plus all 1,279 anomalous ones, so about 20% anomalous. Without subsampling the full test would have only 5% anomalous, and AP and F1 would change a lot. For comparability all models use the reduced test.

**Val / held-out test split** (`experiments/single_bee/make_val_split.py`, seed 0):

- **Val:** 25% of the anomalous bees (322: 210 with one varroa, 112 with two), stratified by cluster × number of varroa. To these are added 1,251 healthy bees, stratified by cluster, taken from those **excluded** from the reduced test. Here too the anomalous ones are about 20%.
- **Held-out test:** the other 957 anomalous bees (639 / 318) and the **same 5,000 healthy** bees as the reduced test.
- The two splits share no images (there is an assert in the script). The remaining 18,046 healthy bees are not used.

## 2. How the best configuration was chosen

The process had two levels.

### a) Recipe (hyperparameters): per-model sweep on the validation set

Script `sweep_<model>.py`, logs in `logs/<model>_sweep/`.

- Variants of backbone, layers, resolution, lr, number of steps, map blur, image score, bee masking, etc.
- For speed the metrics were computed at 128 px, with the post-processing variants evaluated on the same maps without retraining.
- The criterion was image-level AUROC, also looking at pixel AP.
- Findings that hold for almost all models:
  - Resolution 512 px (native) or higher.
  - Processing only the bee: masked loss and map restricted to the eroded mask. The black background is about 86% of the crop.
  - Map blur with σ around 8–16 px (a varroa measures about 25–43 px).
  - Image score = maximum of the map.
  - Short training, because the trained models peak early and then get worse.

### b) Choice of step/epoch and thresholds, on the validation set

- During training, `BestTracker` evaluates the model on **val** at regular intervals. It keeps the best checkpoint with a **lexicographic** criterion: first `img_roc_auc`, then `pxl_avg_prec` to break ties.
- Models without training (PatchCore, PaDiM) have no step to choose: a single fit is done.
- The following are also fixed on val:
  - the **image-level threshold** that maximises F1;
  - the **pixel-level threshold** and the heatmap normalisation.
- The **held-out test is evaluated only once**, on the chosen checkpoint and with the val thresholds. `img_f1` and `pxl_f1` are therefore at the val threshold. For comparison, `*_f1_oracle` is also reported, i.e. the F1 at the optimal threshold on the test.

## 3. Final recipes and results on the test set

| Model | Recipe (summary) | Best | img AUROC | img AP | img F1 | pxl AP | AU-PRO |
|---|---|---|---|---|---|---|---|
| PatchCore | WRN50 layer2-3, bank 30k, σ 8, score = max | fit | **0.9986** | 0.993 | 0.954 | **0.777** | 0.998 |
| CFA | WRN50 layer2-3, 768 px, 32k centroids on bee patches, lr 1e-5 | step 348 | 0.9959 | 0.976 | 0.926 | 0.755 | 0.998 |
| Dinomaly | official ViT-B (layers 2-9), 10k iterations | iter 10000 | 0.9958 | 0.974 | 0.930 | 0.768 | 0.999 |
| PaDiM | WRN50 layer3, pool 3×3, shared Gaussian, eps 1e-4, σ 4 | fit | 0.9886 | 0.933 | 0.881 | 0.521 | 0.994 |
| STFPM | R18, map from layer3 only, 600 steps, SGD lr 0.4 | step 540 | 0.9832 | 0.920 | 0.835 | 0.534 | 0.996 |
| FastFlow | WRN50, 512 px, NLL loss on bee pixels, z² on all scales, erosion 24, σ 16 | step 125 | 0.9818 | 0.910 | 0.834 | 0.469 | 0.987 |
| PaSTe | R18, lr 0.4, 400 steps, map layer2+3, σ 16 | step 350 | 0.9652 | 0.810 | 0.763 | 0.393 | 0.993 |
| RD4AD | WRN50, map layer2, 400 steps, σ 8 | step 400 | 0.9531 | 0.787 | 0.732 | 0.349 | 0.989 |

PatchCore with MobileNetV2 (0.9987) and ResNet18 (0.9983) has an equivalent AUROC but a lower pixel AP (0.754 / 0.735). GANomaly is excluded from the final comparison: it was not rerun on the final dataset with the val protocol (on batch3+4 its AUROC was 0.897).

## 4. Limitations

2. **The synthetic varroa come from only 2 source textures**, shared between val and test. Val and test therefore measure the same anomaly distribution, and generalisation to real varroa is not demonstrated.
3. **Run-to-run noise is about ±0.01 AUROC** for the trained models (RD4AD fluctuates by ±0.02 from one evaluation to the next). Differences below this threshold, for example between CFA and Dinomaly, are not significant. There is only one seed.
4. **Anomalous bees are about 20%** of val and of the reduced test, a proportion chosen for this evaluation, not the natural one. AP and F1 should be read in this context.
