# Brief: thesis section on YOLO fine-tuning and ablation study

This file is a briefing for writing one section of a master's thesis. It contains everything that was actually done and measured. Please write the section in English, in academic style, using only the facts and numbers given here.

## What I need from you

Write the thesis section that describes how an instance-segmentation model was fine-tuned to segment honey bees on hive-frame photographs, and the ablation study that compares the fine-tuning choices we tried.

A structure that would work (adapt it if a different order reads better):

1. Task and dataset
2. Model and training setup
3. Evaluation protocol
4. Ablation study: one subsection or paragraph per factor, with tables
5. Discussion, limitations, and the configuration finally adopted

Things to keep in mind while writing:

- Every number must come from this file. If something you want to say is not supported here, leave it out or mark it as `[AUTHOR TO CONFIRM]`.
- Items already marked `[AUTHOR TO CONFIRM]` below are facts I could not verify. Keep the marker visible in the text so the author can fill them in.
- Each configuration was trained once (seed 0) and the test set has 56 images. Differences below roughly 0.01 mAP should be described as within noise, not as improvements or regressions.
- Use neutral names for the runs in the text (for example "baseline", "1440 px", "1440 px + rotation"). The folder names are listed only so the author can trace results.
- Tables can be given in Markdown or LaTeX. `[AUTHOR TO CONFIRM: preferred format]`
- Primary metric: mask mAP50-95 on the test set. Box metrics are secondary.

## Task and dataset

- Task: instance segmentation of honey bees on photographs of hive frames, single class (`bee`).
- Images: 6016 x 4000 px JPEG photographs. `[AUTHOR TO CONFIRM: camera, acquisition conditions, apiary]`
- Images are very dense: on average more than 300 bees per image, up to about 1,400.
- Labels: polygon masks generated with LangSAM (text-prompted segmentation, prompt `"bee."`), exported in YOLO segmentation format. `[AUTHOR TO CONFIRM: whether labels were manually reviewed or corrected, and how]`. If they were not manually corrected, the reported metrics measure agreement with LangSAM pseudo-labels, and the text should say so.
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

All "batch 3+4" runs stopped through early stopping, none reached 200 epochs.

Configurations that could not be trained (out of GPU memory on 48 GB):

- 1600 px, batch size 2: crashed at epoch 9 inside the mask loss, on a batch with many instances.
- 1600 px with mask ratio 2 (higher-resolution mask supervision): out of memory immediately, even at batch size 1.

Peak GPU memory during training: about 33 GB at 1280 px, 43 to 45 GB at 1440 px. Time per epoch: about 45 s at 1280 px, 55 s at 1440 px.

One further old model exists (`best_model_tuned`: YOLO26n-seg, 1080 px, library-default settings with mosaic, max 300 detections). It scores 0.079 mask mAP50-95 on this test set. It differs from the others in many settings at once, so it is not part of the ablation; mention it at most as an early attempt.

## Validation results (best checkpoint, at training resolution)

| Run | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| Baseline (1280) | 0.891 | 0.855 | 0.921 | 0.659 | 0.885 | 0.850 | 0.908 | 0.608 |
| 1440 px | 0.888 | 0.858 | 0.923 | 0.671 | 0.883 | 0.853 | 0.911 | 0.634 |
| 1440 px + rotation | 0.890 | 0.852 | 0.920 | 0.664 | 0.886 | 0.844 | 0.907 | 0.623 |
| 1440 px + rotation + copy-paste | 0.886 | 0.857 | 0.919 | 0.665 | 0.883 | 0.851 | 0.904 | 0.622 |

The September models were validated on a different validation set, so their validation numbers are not comparable and are omitted. Use the test table for them.

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
- This comparison is confounded: the two runs also differ in maximum epochs (150 vs 200) and patience (20 vs 40), and the 298-image set does not contain the 90 oldest images. It should be presented as indicative only.

Layer freezing:
- Freezing the first 5 layers lowered mask mAP50-95 by 0.022 at native resolution and 0.006 at 2048 px.
- At 2048 px the frozen model has slightly higher recall and F1 (0.8503 and 0.8648 against 0.8396 and 0.8574), so the honest summary is "no benefit", not "clearly harmful".

Stronger augmentation:
- Full rotation and more vertical flips reduced overfitting during training: the validation mask loss stayed flat until the end (about 1.55), whereas in the 1440 px run without them it rose after the best epoch (1.51 at epoch 80 to 1.57 at epoch 119) while the training loss kept falling.
- This did not translate into better accuracy: both augmented models score about 0.01 lower than the plain 1440 px model on test, which is at the edge of the noise level.
- Copy-paste on top of rotation changed results by less than 0.01.
- The augmented runs trained longer before stopping (152 and 184 epochs against 119).

Overfitting in the baseline:
- In the 1280 px baseline, validation mask mAP50-95 plateaued around epoch 64 (0.60 to 0.61) while the training mask loss kept decreasing (1.55 at epoch 60, 1.37 at epoch 140) and the validation mask loss rose (1.60 to 1.64).

## Configuration adopted

YOLO26s-seg fine-tuned on the 298-image set at 1440 px with the baseline augmentation, used at 2048 px at inference. On the test set at 2048 px: mask mAP50-95 0.628, mask mAP50 0.893, mask mAP75 0.738, mask precision 0.895, recall 0.855, F1 0.875; box mAP50-95 0.662.

## Limitations to state in the text

- One run per configuration, single seed; no confidence intervals.
- Small evaluation sets (55 validation and 56 test images), although each contains more than 17,000 instances.
- Ground truth comes from LangSAM. `[AUTHOR TO CONFIRM: extent of manual correction]`
- The data-size comparison is confounded with schedule changes and a partly different image pool.
- The training set of the September models is inferred, not recorded.
- Precision, recall and F1 are reported at the confidence threshold of 0.10 and would change with a different threshold.
- Resolutions above 1440 px in training, and finer mask supervision (mask ratio 2), could not be tested because of GPU memory. Training on image tiles instead of downscaled full images is the natural next step and was not attempted.
