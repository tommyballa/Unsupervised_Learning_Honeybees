# YOLO26s-seg fine-tuning and ablation study on LangSAM labels (batches 3+4)

This document describes how YOLO26s-seg was fine-tuned to segment honey bees on hive-frame photographs using LangSAM pseudo-labels, and the ablation study that compares the fine-tuning choices that were tried. It is the first of two ablations: the later one, on SAM 3 labels and batches 3-6, is described in [ablation_sam3.md](ablation_sam3.md).

## Notes on the metrics

- Primary metric: mask mAP50-95 on the test set. Box metrics are secondary.
- Each configuration was trained once (seed 0), except the 1440 px reference, which was trained with two seeds. Those two runs differ by 0.010 mask mAP50-95 on the test set at 1440 px and by 0.008 at 2048 px. Differences of about 0.01 mAP or less should therefore be read as within noise, not as improvements or regressions.
- Runs are referred to by neutral names (for example "baseline", "1440 px", "1440 px + rotation"); the corresponding folder names are listed in the Runs table so the results can be traced.
- Bracketed "author to confirm" notes mark facts that have not been verified yet.

## Task and dataset

- Task: instance segmentation of honey bees on photographs of hive frames, single class (`bee`).
- Images: 6016 x 4000 px JPEG photographs. `[AUTHOR TO CONFIRM: camera, acquisition conditions, apiary]`
- Images are very dense: on average more than 300 bees per image, up to about 1,400.
- Labels: polygon masks generated with LangSAM (text-prompted segmentation, prompt `"bee."`), exported in YOLO segmentation format. `[AUTHOR TO CONFIRM: whether labels were manually reviewed or corrected, and how]`. If they were not manually corrected, the reported metrics measure agreement with LangSAM pseudo-labels.
- Data was collected and labelled in batches. `[AUTHOR TO CONFIRM: what each batch corresponds to (dates, hives, domains)]`

Splits used for the main experiments ("batch 3+4" dataset):

| Split | Images | Bee instances | Mean per image | Max per image |
|---|---|---|---|---|
| Train | 298 | 92,075 | 309 | 1,371 |
| Validation | 55 | 17,572 | 319 | 1,194 |
| Test | 56 | 19,314 | 345 | 1,360 |

- The training set is the union of an earlier batch (115 images) and the newly labelled batch 4 (183 images).
- Validation and test come from 111 held-out images (53 from the earlier batch, 58 from batch 4), shuffled with a fixed seed (42) and split in half.
- No image appears in more than one split (checked by filename).
- The split is random per image. It is not stratified by batch, domain, or number of bees per image.

Earlier training set ("205-image" dataset), used by the September models:

- 205 images: an older batch of 90 images plus the same 115-image batch above.
- None of these 205 images appears in the current validation or test sets (checked by filename).
- The exact training set of the September runs is inferred from folder dates, because the run configuration only records the name of the dataset config file. `[AUTHOR TO CONFIRM]`

Object size, which motivates the resolution experiments. When the long side is resized to 1280 px, bee size (square root of bounding-box area) is:

| Percentile | 5th | 25th | 50th | 75th | 95th |
|---|---|---|---|---|---|
| Size in px | 23 | 34 | 49 | 69 | 91 |

Polygons have a median of 24 vertices.

## Model and training setup

- Model: YOLO26s-seg (Ultralytics), 11.4 M parameters, 37.3 GFLOPs, initialised from the official pretrained weights and fine-tuned end to end.
- Software: Ultralytics 8.4.138, PyTorch 2.14.0 (CUDA), Python 3.10.
- Hardware: one NVIDIA RTX A6000 (48 GB) per run.
- Seed 0, deterministic mode, automatic mixed precision.

Baseline hyperparameters (shared by all "batch 3+4" runs unless stated otherwise):

| Group | Setting |
|---|---|
| Schedule | up to 200 epochs, early stopping with patience 40 |
| Batch | batch size 2 (nominal batch size 64 through gradient accumulation) |
| Input size | 1280 px (long side) |
| Optimizer | AdamW, initial LR 0.001, cosine decay to 1% of initial LR, 5 warm-up epochs, weight decay 0.0005 |
| Augmentation | HSV jitter (h 0.015, s 0.7, v 0.4), rotation +/-10 degrees, translation 0.1, scale +/-50%, horizontal flip 0.5, vertical flip 0.2, random erasing 0.4 |
| Disabled | mosaic (off, it quarters the effective resolution of small objects), mixup, copy-paste |
| Masks | mask downsampling ratio 4, overlapping masks enabled |
| Loss weights | box 7.5, cls 0.5, dfl 1.5 (library defaults) |
| Detections | max 1,500 per image during training-time validation |

The batch size is limited to 2 by GPU memory: with hundreds of instances per image, the mask loss dominates memory use.

The September runs on the 205-image dataset used the same settings except: up to 150 epochs, patience 20, max 1,200 detections.

## Evaluation protocol

- Model selection: best checkpoint on the validation set (Ultralytics fitness, combining box and mask mAP).
- Final comparison: all models evaluated on the same 56-image test set with identical settings: confidence threshold 0.10, max 2,000 detections per image.
- Each model is tested at two input sizes: its own training resolution, and 2048 px.
- Metrics: COCO-style mAP50-95, mAP50, mAP75 for masks and boxes; precision, recall and F1 for masks.

## Runs

| Label for the text | Folder name | Train set | Train size (px) | Change vs. baseline | Epochs run | Best epoch | Training time |
|---|---|---|---|---|---|---|---|
| Sept. baseline A | `bee_model_finetuned_yolo26s_dominiuniti` | 205 img | 1280 | — | 104 | n/a | n/a |
| Sept. frozen | `bee_model_finetuned_yolo26s_dominiuniti_freeze5` | 205 img | 1280 | first 5 layers frozen | 101 | n/a | n/a |
| Sept. baseline B | `bee_model_finetuned_yolo26s_dominiuniti_splittestval` | 205 img | 1280 | — (repeat of A with a separate val/test split) | 142 | 122 | 1.7 h |
| Baseline | `bee_model_finetuned_yolo26s_batch3+4` | 298 img | 1280 | — | 146 | 106 | 1.8 h |
| 1440 px | `bee_model_finetuned_yolo26s_batch3+4_img1440` | 298 img | 1440 | input size 1440 | 119 | 79 | 1.8 h |
| 1440 px + rotation | `bee_model_finetuned_yolo26s_batch3+4_img1440_rot` | 298 img | 1440 | rotation +/-180 degrees, vertical flip 0.5 | 152 | 112 | 2.3 h |
| 1440 px + rotation + copy-paste | `bee_model_finetuned_yolo26s_batch3+4_img1440_rot_cp` | 298 img | 1440 | as above, plus copy-paste 0.3 | 184 | 144 | 2.9 h |

All the "batch 3+4" runs in this table stopped through early stopping, none reached 200 epochs. The freeze-depth runs are listed separately in their own section below.

Configurations that could not be trained (out of GPU memory on 48 GB):

- 1600 px, batch size 2: crashed at epoch 9 inside the mask loss, on a batch with many instances.
- 1600 px with mask ratio 2 (higher-resolution mask supervision): out of memory immediately, even at batch size 1.

Peak GPU memory during training: about 33 GB at 1280 px, 43 to 45 GB at 1440 px. Time per epoch: about 45 s at 1280 px, 55 s at 1440 px.

One further old model exists (`best_model_tuned`: YOLO26n-seg, 1080 px, library-default settings with mosaic, max 300 detections). It scores 0.079 mask mAP50-95 on this test set. It differs from the others in many settings at once, so it is not part of the ablation and is listed only as an early attempt.

## Validation results (best checkpoint, at training resolution)

| Run | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| Baseline (1280) | 0.891 | 0.855 | 0.921 | 0.659 | 0.885 | 0.850 | 0.908 | 0.608 |
| 1440 px | 0.888 | 0.858 | 0.923 | 0.671 | 0.883 | 0.853 | 0.911 | 0.634 |
| 1440 px + rotation | 0.890 | 0.852 | 0.920 | 0.664 | 0.886 | 0.844 | 0.907 | 0.623 |
| 1440 px + rotation + copy-paste | 0.886 | 0.857 | 0.919 | 0.665 | 0.883 | 0.851 | 0.904 | 0.622 |

The September models were validated on a different validation set, so their validation numbers are not comparable and are omitted; see the test tables for them.

## Test results (56 images, 19,314 instances)

Tested at each model's training resolution:

| Run | Test size | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | Mask mAP75 | Mask P | Mask R | Mask F1 |
|---|---|---|---|---|---|---|---|---|---|
| Sept. baseline A | 1280 | 0.6056 | 0.8864 | 0.5500 | 0.8734 | 0.6115 | 0.8690 | 0.8356 | 0.8520 |
| Sept. frozen | 1280 | 0.5858 | 0.8727 | 0.5285 | 0.8590 | 0.5799 | 0.8543 | 0.8247 | 0.8392 |
| Sept. baseline B | 1280 | 0.6120 | 0.8818 | 0.5536 | 0.8689 | 0.6205 | 0.8756 | 0.8405 | 0.8577 |
| Baseline | 1280 | 0.6235 | 0.8918 | 0.5675 | 0.8782 | 0.6443 | 0.8821 | 0.8410 | 0.8611 |
| 1440 px | 1440 | 0.6408 | 0.8930 | 0.5971 | 0.8870 | 0.6898 | 0.8767 | 0.8460 | 0.8610 |
| 1440 px + rotation | 1440 | 0.6223 | 0.8876 | 0.5746 | 0.8750 | 0.6533 | 0.8737 | 0.8294 | 0.8510 |
| 1440 px + rotation + copy-paste | 1440 | 0.6309 | 0.8900 | 0.5815 | 0.8767 | 0.6635 | 0.8730 | 0.8417 | 0.8571 |

Tested at 2048 px:

| Run | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | Mask mAP75 | Mask P | Mask R | Mask F1 |
|---|---|---|---|---|---|---|---|---|
| Sept. baseline A | 0.6370 | 0.8900 | 0.5959 | 0.8761 | 0.6916 | 0.8761 | 0.8396 | 0.8574 |
| Sept. frozen | 0.6259 | 0.8917 | 0.5898 | 0.8795 | 0.6885 | 0.8799 | 0.8503 | 0.8648 |
| Sept. baseline B | 0.6363 | 0.8905 | 0.5980 | 0.8781 | 0.6959 | 0.8787 | 0.8421 | 0.8600 |
| Baseline | 0.6504 | 0.8967 | 0.6093 | 0.8838 | 0.7124 | 0.8987 | 0.8473 | 0.8723 |
| **1440 px** | **0.6624** | 0.8979 | **0.6280** | 0.8926 | **0.7376** | 0.8949 | **0.8553** | **0.8747** |
| 1440 px + rotation | 0.6542 | 0.8972 | 0.6166 | 0.8842 | 0.7233 | 0.8934 | 0.8533 | 0.8728 |
| 1440 px + rotation + copy-paste | 0.6598 | 0.9049 | 0.6198 | 0.8923 | 0.7306 | 0.8932 | 0.8527 | 0.8725 |

## Ablation: effect of each factor on test mask mAP50-95

| Factor | Comparison | Tested at training resolution | Tested at 2048 px |
|---|---|---|---|
| Inference at 2048 px instead of training resolution | each model against itself | — | +0.031 to +0.061 |
| Training at 1440 px instead of 1280 px | Baseline -> 1440 px | +0.030 | +0.019 |
| Larger training set (298 vs 205 images) | Sept. baseline B -> Baseline | +0.014 | +0.011 |
| Freezing the first 5 layers | Sept. baseline A -> Sept. frozen | -0.022 | -0.006 |
| Full rotation augmentation | 1440 px -> 1440 px + rotation | -0.023 | -0.011 |
| Copy-paste on top of rotation | 1440 px + rotation -> + copy-paste | +0.007 | +0.003 |

Per-model gain from testing at 2048 px (mask mAP50-95): Sept. baseline A +0.046, Sept. frozen +0.061, Sept. baseline B +0.044, Baseline +0.042, 1440 px +0.031, 1440 px + rotation +0.042, 1440 px + rotation + copy-paste +0.038.

## Freeze-depth ablation on the 1440 px model

This is a second, more controlled experiment on layer freezing. All runs use exactly the 1440 px configuration (298-image training set, same hyperparameters, seed 0) and differ only in the `freeze` setting. `freeze = N` means that layers 0 to N-1 are frozen. In Ultralytics, freezing a layer fixes its weights and also keeps its batch-normalisation statistics fixed.

YOLO26s-seg has 24 layers (indices 0 to 23):

| Layers | Role | Share of parameters |
|---|---|---|
| 0-10 | backbone | 47.3% |
| 11-22 | neck (feature pyramid) | 31.1% |
| 23 | segmentation head | 21.6% |

What each setting freezes:

| Setting | Frozen layers | What stays trainable | Trainable parameters |
|---|---|---|---|
| freeze = 0 | none | everything | 100% |
| freeze = 5 | 0-4 | all but the first backbone stages | 97.4% |
| freeze = 10 | 0-9 | last backbone block, neck, head | 61.3% |
| freeze = 20 | 0-19 | last neck stage and head | 42.8% |
| freeze = 23 | 0-22 | segmentation head only | 21.6% |

The freeze = 0 run with seed 0 is the "1440 px" model of the previous sections. A second freeze = 0 run with seed 1 was added to measure run-to-run variation. freeze = 23 was added as the extreme case (head-only training).

Training outcome:

| Setting | Folder suffix | Epochs run | Stopped by | Training time | Time per epoch | Final train mask loss | Final val mask loss |
|---|---|---|---|---|---|---|---|
| freeze = 0, seed 0 | `_img1440` | 119 | early stopping | 1.8 h | 55 s | 1.345 | 1.569 |
| freeze = 0, seed 1 | `_img1440_seed1` | 149 | early stopping | 2.3 h | 56 s | 1.311 | 1.594 |
| freeze = 5 | `_img1440_freeze5` | 126 | early stopping | 1.9 h | 55 s | 1.388 | 1.626 |
| freeze = 10 | `_img1440_freeze10` | 133 | early stopping | 2.0 h | 54 s | 1.409 | 1.613 |
| freeze = 20 | `_img1440_freeze20` | 200 | epoch limit | 2.8 h | 51 s | 1.776 | 1.795 |
| freeze = 23 | `_img1440_freeze23` | 200 | epoch limit | 3.3 h | 59 s | 1.967 | 1.883 |

Peak GPU memory was about 42.5 GB for freeze = 5 and freeze = 10, the same as without freezing, and 41 GB for freeze = 20 and freeze = 23.

Validation results (best checkpoint, 1440 px):

| Setting | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| freeze = 0, seed 0 | 0.888 | 0.858 | 0.923 | 0.671 | 0.883 | 0.853 | 0.911 | 0.634 |
| freeze = 0, seed 1 | 0.906 | 0.868 | 0.928 | 0.680 | 0.899 | 0.860 | 0.912 | 0.632 |
| freeze = 5 | 0.891 | 0.852 | 0.920 | 0.659 | 0.882 | 0.848 | 0.904 | 0.617 |
| freeze = 10 | 0.902 | 0.856 | 0.924 | 0.663 | 0.896 | 0.850 | 0.909 | 0.621 |
| freeze = 20 | 0.764 | 0.726 | 0.802 | 0.489 | 0.760 | 0.723 | 0.790 | 0.499 |
| freeze = 23 | 0.773 | 0.721 | 0.799 | 0.479 | 0.774 | 0.717 | 0.793 | 0.491 |

Test results at 1440 px (56 images, same protocol as above):

| Setting | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | Mask mAP75 | Mask P | Mask R | Mask F1 |
|---|---|---|---|---|---|---|---|---|
| freeze = 0, seed 0 | 0.6408 | 0.8930 | 0.5971 | 0.8870 | 0.6898 | 0.8767 | 0.8460 | 0.8610 |
| freeze = 0, seed 1 | 0.6382 | 0.8948 | 0.5869 | 0.8809 | 0.6721 | 0.8859 | 0.8477 | 0.8664 |
| freeze = 5 | 0.6214 | 0.8877 | 0.5731 | 0.8754 | 0.6470 | 0.8723 | 0.8363 | 0.8539 |
| freeze = 10 | 0.6246 | 0.8903 | 0.5758 | 0.8773 | 0.6578 | 0.8741 | 0.8372 | 0.8553 |
| freeze = 20 | 0.4202 | 0.7012 | 0.4279 | 0.6974 | 0.4769 | 0.7329 | 0.6834 | 0.7073 |
| freeze = 23 | 0.4109 | 0.7043 | 0.4189 | 0.6956 | 0.4581 | 0.7463 | 0.6725 | 0.7075 |

Test results at 2048 px:

| Setting | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | Mask mAP75 | Mask P | Mask R | Mask F1 |
|---|---|---|---|---|---|---|---|---|
| freeze = 0, seed 0 | 0.6624 | 0.8979 | 0.6280 | 0.8926 | 0.7376 | 0.8949 | 0.8553 | 0.8747 |
| freeze = 0, seed 1 | 0.6620 | 0.9003 | 0.6205 | 0.8875 | 0.7264 | 0.9049 | 0.8611 | 0.8824 |
| freeze = 5 | 0.6525 | 0.9035 | 0.6192 | 0.8912 | 0.7238 | 0.8935 | 0.8514 | 0.8719 |
| freeze = 10 | 0.6581 | 0.9073 | 0.6187 | 0.8944 | 0.7240 | 0.8963 | 0.8644 | 0.8801 |
| freeze = 20 | 0.5026 | 0.7931 | 0.5112 | 0.7896 | 0.5832 | 0.8062 | 0.7436 | 0.7736 |
| freeze = 23 | 0.4595 | 0.7613 | 0.4699 | 0.7611 | 0.5279 | 0.7707 | 0.7138 | 0.7411 |

Mean of the two freeze = 0 runs, test mask mAP50-95: 0.592 at 1440 px, 0.624 at 2048 px.

What this experiment shows:

- Run-to-run variation: the two unfrozen runs differ by 0.010 (1440 px) and 0.008 (2048 px) in test mask mAP50-95. This is the noise level against which the other differences should be read.
- freeze = 5 and freeze = 10 behave alike. At 1440 px they are 0.016 to 0.019 below the mean of the unfrozen runs, slightly more than the seed-to-seed gap. At 2048 px they are within noise of the unfrozen runs (0.619 against 0.621 and 0.628).
- Freezing 39% of the parameters (freeze = 10) costs almost nothing in accuracy, which indicates that the pretrained backbone features transfer well to this domain.
- Freezing brings no practical benefit here: time per epoch and peak memory are unchanged for freeze = 5 and freeze = 10, and the runs did not converge in fewer epochs.
- freeze = 20 and freeze = 23 collapse: against the mean of the unfrozen runs, mask mAP50-95 is 0.11 to 0.15 lower at 2048 px and 0.16 to 0.17 lower at 1440 px.
- This is underfitting, not overfitting: the training mask loss stays at 1.78 and 1.97 against 1.35 without freezing, and both runs used all 200 epochs without triggering early stopping.
- The drop happens between freeze = 10 and freeze = 20, that is, when the neck is frozen. The feature-pyramid layers need to adapt to the target domain (small, densely packed objects); retraining the head alone, or the head with the last neck stage, is not enough.
- The result agrees with the earlier September comparison at 1280 px on the 205-image set (freezing 5 layers: -0.022 at native resolution, -0.006 at 2048 px).

## What the results show

Inference resolution:
- Testing at 2048 px improves every model, by 0.03 to 0.06 mask mAP50-95, with no retraining.
- The gain is concentrated in localisation quality: mask mAP75 rises much more than mAP50 (for the 1440 px model, mAP75 goes from 0.690 to 0.738 while mAP50 goes from 0.887 to 0.893).
- This is consistent with the small object size: a quarter of the bees are under 34 px at 1280 px input.

Training resolution:
- Raising the training size from 1280 to 1440 px is the only training-side change that gave a clear improvement (+0.030 at native resolution, +0.019 at 2048 px).
- It also converged faster: best checkpoint at epoch 79 instead of 106.
- 1440 px is the largest size that fits in 48 GB at batch size 2 with this instance density; 1600 px ran out of memory.

Training data:
- Going from 205 to 298 training images improved mask mAP50-95 by about 0.01.
- This comparison is confounded: the two runs also differ in maximum epochs (150 vs 200) and patience (20 vs 40), and the 298-image set does not contain the 90 oldest images. It is indicative only.

Layer freezing:
- Freezing the first 5 layers lowered mask mAP50-95 by 0.022 at native resolution and 0.006 at 2048 px.
- At 2048 px the frozen model has slightly higher recall and F1 (0.8503 and 0.8648 against 0.8396 and 0.8574), so the result is best summarised as "no benefit", not "clearly harmful".
- The controlled freeze-depth experiment on the 1440 px model (previous section) confirms this and extends it: freezing up to the first 10 layers is roughly neutral, freezing the neck as well is clearly harmful.

Stronger augmentation:
- Full rotation and more vertical flips reduced overfitting during training: the validation mask loss stayed flat until the end (about 1.55), whereas in the 1440 px run without them it rose after the best epoch (1.51 at epoch 80 to 1.57 at epoch 119) while the training loss kept falling.
- This did not translate into better accuracy: both augmented models score about 0.01 lower than the plain 1440 px model on test, which is at the edge of the noise level.
- Copy-paste on top of rotation changed results by less than 0.01.
- The augmented runs trained longer before stopping (152 and 184 epochs against 119).

Overfitting in the baseline:
- In the 1280 px baseline, validation mask mAP50-95 plateaued around epoch 64 (0.60 to 0.61) while the training mask loss kept decreasing (1.55 at epoch 60, 1.37 at epoch 140) and the validation mask loss rose (1.60 to 1.64).

## Configuration adopted

YOLO26s-seg fine-tuned on the 298-image set at 1440 px with the baseline augmentation, used at 2048 px at inference. On the test set at 2048 px: mask mAP50-95 0.628, mask mAP50 0.893, mask mAP75 0.738, mask precision 0.895, recall 0.855, F1 0.875; box mAP50-95 0.662.

## Limitations

- One run per configuration, except the unfrozen 1440 px model (two seeds); no confidence intervals. The noise estimate rests on that single pair of runs.
- freeze = 20 and freeze = 23 were still improving slowly when they reached the 200-epoch limit, so their scores are a lower bound for those settings; the gap to the unfrozen model is far larger than what the remaining trend could close.
- Small evaluation sets (55 validation and 56 test images), although each contains more than 17,000 instances.
- Ground truth comes from LangSAM. `[AUTHOR TO CONFIRM: extent of manual correction]`
- The data-size comparison is confounded with schedule changes and a partly different image pool.
- The training set of the September models is inferred, not recorded.
- Precision, recall and F1 are reported at the confidence threshold of 0.10 and would change with a different threshold.
- Resolutions above 1440 px in training, and finer mask supervision (mask ratio 2), could not be tested because of GPU memory. Training on image tiles instead of downscaled full images is the natural next step and was not attempted.
