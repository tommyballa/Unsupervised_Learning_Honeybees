# YOLO26s-seg ablation on SAM 3 labels (batches 3-6)

Fine-tuning of YOLO26s-seg for bee segmentation, redone on the SAM 3 pseudo-labels of batches 3-6. It repeats the earlier ablation on the LangSAM labels (batches 3+4, see `thesis_brief_yolo_finetuning.md`), with the same variants and the same test protocol.

Runs carried out from 12 to 16 October 2026.

## Dataset

- Labels: polygon masks generated with SAM 3 (`generate_labels_sam3.py`), folder `bee_batch3+4+5+6_SAM3`. 791 photos, after removing the 22 taken with the phone.
- Split: `split_dataset_sam3.py`, **random by photo**, in `processed/bee_batch3+4+5+6_SAM3_split`. Config: `data_sam3.yaml`.

| Split | Photos | Bees | Mean per photo |
|---|---|---|---|
| Train | 554 | 182 318 | 329 |
| Validation | 118 | 39 076 | 331 |
| Test | 119 | 39 192 | 329 |

With SAM 3 a single photo reaches about 1 850 bees (with LangSAM the maximum was about 1 370), which is why `max_det` is 2 500.

## Common setup

- Model: pre-trained YOLO26s-seg (`yolo26s-seg.pt`), full fine-tuning; script `train_yolo.py`.
- Ultralytics 8.4.138, PyTorch 2.14.0, one 48 GB NVIDIA RTX A6000 per run (shared server).
- Hyperparameters identical to the LangSAM ablation: up to 200 epochs, early stopping with patience 40, batch 2, AdamW (lr0 0.001, cosine decay down to 1%, 5 warm-up epochs, weight decay 0.0005), mosaic disabled, `cache=True`, seed 0.
- Base augmentation: HSV (h 0.015, s 0.7, v 0.4), rotation ±10°, translation 0.1, scale ±50%, horizontal flip 0.5, vertical flip 0.2.
- `max_det` 2 500.
- No run had its batch size reduced because of out-of-memory errors: all of them use batch 2.

**High resolution: 1376 px instead of 1440.** In the LangSAM ablation the high resolution was 1440 px. With SAM 3 the photos are more crowded and 1440 px runs out of memory at batch 2. Memory test with the two most crowded photos in the same batch (3 579 instances):

| Resolution | Peak memory |
|---|---|
| 1280 px | 36.7 GB |
| 1344 px | 42.3 GB |
| 1376 px | 41.4 GB |
| 1440 px | out of memory |

## Test protocol

- Selected model: `best.pt`, i.e. the best checkpoint on the validation set (Ultralytics fitness).
- Test: 119 photos, `split=test`, conf 0.10, `max_det` 2 500, script `logs/ablation_sam3/abl_eval.py`.
- Each model is tested at two resolutions: the training one and 2048 px.
- Segmentation (mask) metrics: mAP50-95, mAP50, mAP75, precision, recall, F1. In addition, box mAP50-95 and mAP50.
- Criterion for the "best model": seg mAP50-95 on the test set at 2048 px.

## Runs

| Run | Folder (`runs/segment/runs/segment/`) | Training | Change from base | Epochs (best) | Val seg mAP50-95 | Notes |
|---|---|---|---|---|---|---|
| base | `bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3` | 1280 px | — | 147 (85) | 0.647 | early stop, 3.9 h |
| img1376 | `..._SAM3_img1376` | 1376 px | resolution 1376 | 144 (104) | 0.664 | early stop, 5.0 h |
| img1376_rot | `..._SAM3_img1376_rot` | 1376 px | rotation ±180°, vertical flip 0.5 | 176 (136) | 0.656 | early stop, 4.8 h |
| img1376_rot_cp | `..._SAM3_img1376_rot_cp` | 1376 px | as above + copy-paste 0.3 | 198 (158) | 0.659 | early stop, 5.6 h |
| freeze5 | `..._SAM3_img1376_freeze5` | 1376 px | first 5 layers frozen | 137 (97) | 0.649 | early stop, 3.7 h |
| freeze10 | `..._SAM3_img1376_freeze10` | 1376 px | first 10 layers frozen | 200 (154) | 0.644 | resumed 2 times ¹ |
| freeze20 | `..._SAM3_img1376_freeze20` | 1376 px | first 20 layers frozen | 200 (164) | 0.530 | |
| freeze23 | `..._SAM3_img1376_freeze23` | 1376 px | first 23 layers frozen | 200 (164) | 0.476 | |
| seed1 | `..._SAM3_img1376_seed1` | 1376 px | img1376 with `seed=1` | 200 (82) | 0.662 | resumed 7 times ² |

¹ freeze10 was resumed from `last.pt` at epochs 161 and 184, after out-of-memory errors caused by other processes on the GPU. The first attempt had been interrupted on 13/10 by a full system disk: it is kept as `..._freeze10__interrotto_disco_pieno_1013`.

² seed1 was resumed at epochs 99, 136, 145, 154, 163, 172 and 181, each time because of out-of-memory errors caused by other processes. The tested `best.pt` is from epoch 82, saved before the first resume, so it is identical to that of an uninterrupted run. However, every resume resets the early-stopping counter: this is why the run reached 200 epochs instead of stopping around epoch 122.

freeze10, freeze20, freeze23 and seed1 were trained on a shared server: their training times are not comparable and I do not report them. Failed attempts are kept with the suffix `__interrotto_*`, both in `runs/` and in the logs.

## 1. Resolution and augmentation

### Test at 2048 px

| Run | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|
| base (1280) | 0.688 | 0.926 | 0.798 | 0.924 | 0.882 | 0.902 | 0.699 | 0.914 |
| **img1376** | **0.692** | **0.927** | **0.803** | **0.931** | 0.881 | **0.905** | 0.703 | **0.915** |
| img1376_rot | 0.688 | 0.925 | 0.800 | 0.929 | 0.871 | 0.899 | 0.703 | 0.914 |
| img1376_rot_cp | 0.690 | 0.919 | 0.802 | **0.931** | 0.873 | 0.901 | **0.709** | **0.915** |

### Test at the training resolution

| Run | Test | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| base | 1280 | 0.631 | 0.912 | 0.724 | 0.913 | 0.864 | 0.888 | 0.669 | 0.900 |
| **img1376** | 1376 | **0.646** | **0.914** | **0.743** | 0.920 | **0.865** | **0.891** | **0.675** | 0.903 |
| img1376_rot | 1376 | 0.640 | 0.912 | 0.735 | 0.915 | 0.863 | 0.888 | 0.674 | **0.909** |
| img1376_rot_cp | 1376 | 0.641 | **0.914** | 0.737 | **0.921** | 0.860 | 0.889 | **0.675** | 0.903 |

## 2. Freeze (img1376 configuration)

`freeze=N` freezes the first N layers of the model: the backbone up to layer 10, up to the head with 20-23.

### Test at 2048 px

| Run | freeze | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| **img1376** | 0 | **0.692** | **0.927** | **0.803** | **0.931** | **0.881** | **0.905** | **0.703** | **0.915** |
| freeze5 | 5 | 0.685 | 0.921 | 0.794 | **0.931** | **0.881** | **0.905** | 0.696 | **0.915** |
| freeze10 | 10 | 0.682 | 0.918 | 0.792 | 0.926 | 0.874 | 0.899 | 0.699 | 0.913 |
| freeze20 | 20 | 0.580 | 0.835 | 0.673 | 0.881 | 0.778 | 0.826 | 0.546 | 0.822 |
| freeze23 | 23 | 0.484 | 0.760 | 0.554 | 0.777 | 0.701 | 0.737 | 0.451 | 0.738 |

### Test at 1376 px

| Run | freeze | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| **img1376** | 0 | **0.646** | **0.914** | **0.743** | **0.920** | **0.865** | **0.891** | **0.675** | **0.903** |
| freeze5 | 5 | 0.631 | 0.904 | 0.719 | 0.916 | 0.854 | 0.884 | 0.658 | 0.899 |
| freeze10 | 10 | 0.628 | 0.903 | 0.718 | 0.915 | 0.853 | 0.883 | 0.661 | 0.899 |
| freeze20 | 20 | 0.496 | 0.756 | 0.558 | 0.841 | 0.706 | 0.768 | 0.466 | 0.744 |
| freeze23 | 23 | 0.446 | 0.715 | 0.500 | 0.775 | 0.666 | 0.716 | 0.418 | 0.700 |

## 3. Variability between trainings (seed)

Same training as img1376 with `seed=1`.

| Run | Test | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| img1376 (seed 0) | 2048 | 0.692 | 0.927 | 0.803 | 0.931 | 0.881 | 0.905 | 0.703 | 0.915 |
| img1376 seed1 | 2048 | 0.695 | 0.931 | 0.805 | 0.941 | 0.884 | 0.912 | 0.705 | 0.918 |
| img1376 (seed 0) | 1376 | 0.646 | 0.914 | 0.743 | 0.920 | 0.865 | 0.891 | 0.675 | 0.903 |
| img1376 seed1 | 1376 | 0.645 | 0.915 | 0.741 | 0.925 | 0.863 | 0.892 | 0.670 | 0.903 |

The difference between the two seeds is 0.003 in seg mAP50-95, both at 2048 and at 1376 px. With LangSAM it was 0.008: 0.628 versus 0.620 at 2048 px.

## 4. Comparison with the LangSAM ablation

Seg mAP50-95 on the test set at 2048 px. For LangSAM: batches 3+4, test set of 56 photos and 19 314 bees, high resolution at 1440 px. For SAM 3: batches 3-6, test set of 119 photos, high resolution at 1376 px.

| Variant | LangSAM | SAM 3 |
|---|---|---|
| base 1280 | 0.609 | 0.688 |
| high resolution (1440 / 1376) | **0.628** | 0.692 |
| + rotation | 0.617 | 0.688 |
| + rotation + copy-paste | 0.620 | 0.690 |
| freeze5 | 0.619 | 0.685 |
| freeze10 | 0.619 | 0.682 |
| freeze20 | 0.511 | 0.580 |
| freeze23 | 0.470 | 0.484 |
| seed1 | 0.620 | **0.695** |

Precision, recall and F1 at 2048 px, best configuration:

| | P | R | F1 | mAP75 |
|---|---|---|---|---|
| LangSAM img1440 | 0.895 | 0.855 | 0.875 | 0.738 |
| SAM 3 img1376 | 0.931 | 0.881 | 0.905 | 0.803 |

**The two ablations are not directly comparable in absolute terms.** The test photos differ (56 versus 119), as does the number of training photos (298 versus 554) and, above all, the ground truth. Each model is evaluated against the pseudo-labels of the same tool that generated its training labels. The +6/8 points of SAM 3 may therefore come from more consistent labels, from more data or from an easier test set, and do not prove that the masks are closer to the real bees. A fair comparison would need both models evaluated on the same photos, seen by neither of them in training, ideally with hand-corrected labels. The splits overlap: judging by the file names, 40 photos of the LangSAM test set are in the SAM 3 train set and 45 photos of the SAM 3 test set are in the LangSAM train set.

The **trends**, on the other hand, are comparable, and they agree.

## What the results show

1. **Inference resolution.** Testing at 2048 px is worth +4/6 points of seg mAP50-95 for every variant, with both precision and recall higher. The bees are small: the median is about 49 px on the long side at 1280.
2. **Training resolution.** 1376 px versus 1280 gives +1.5 points at the training resolution. At 2048 px the difference (+0.4) is of the same order as the seed-to-seed variability.
3. **Geometric augmentation and copy-paste.** No measurable effect: with SAM 3 the variants are within 0.004 at 2048 px, below the seed-to-seed difference, while with LangSAM they cost about one point. The bees already appear in every orientation in the photos. Copy-paste finds little free space to paste into, since the photos are packed with bees and Ultralytics only pastes where the overlap is below 30%.
4. **Freeze.** Freezing 5-10 layers costs about one point, small but systematic in both ablations. From 20 layers upwards the model collapses: −11 points with freeze20 and −21 with freeze23. The collapse mostly affects recall, from 0.88 to 0.78 and then 0.70. The features pre-trained on COCO need to be adapted to bees: full fine-tuning is necessary.
5. **Mask contours.** The metric that best separates the good configurations is mAP75, i.e. contour precision. Detection is similar across the good variants: box mAP50 is always about 0.91-0.92.

## Adopted configuration

**`img1376`**: full fine-tuning (no freeze), training at 1376 px, base augmentation, inference at 2048 px.

Weights: `runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt`

The seed1 `best.pt` is equivalent: it is slightly ahead on almost every metric, but within the seed-to-seed variability.

## Limitations

- **Ground truth:** these are SAM 3 pseudo-labels, not corrected by hand. The metrics measure agreement with SAM 3, not with the real bees.
- **Split:** it is random by photo, not by shooting sequence or by comb. Photos of the same comb taken a few seconds apart can end up one in train and one in test, so the absolute metrics may be optimistic. The comparison between variants remains valid, because they all use the same split.
- **Seed:** only one repeated seed. The variability between trainings is estimated from a single pair of runs (0.003 here, 0.008 with LangSAM).
- **Resumes:** freeze10 and seed1 were resumed from checkpoints after out-of-memory errors caused by other GPU users. For seed1 the tested model predates all resumes. For freeze10 the `best.pt` is from epoch 154, before the resumes, so in this case too the tested model is not affected by them.

## Files

- Test results: `logs/ablation_sam3/abl_<run>.json` and `.log`.
- Training logs:
  - `logs/train_batch3+4+5+6_SAM3_*.log` for base, img1376, rot, rot_cp and freeze5;
  - `logs/ablation_sam3/freeze_disk1/` for freeze10, freeze20, freeze23 and seed1, together with the interrupted attempts, the queue status (`queue_gpu0.out`) and the scripts.
- Launch queues: `logs/ablation_sam3/queue.sh` (resolution and augmentation) and `logs/ablation_sam3/freeze_disk1/queue_freeze_gated.sh` (freeze and seed).
- Original copy of the freeze10/20/23 and seed1 runs: `/mnt/disk1/tommaso_ballarin/bee_ablation_sam3_freeze/`.
