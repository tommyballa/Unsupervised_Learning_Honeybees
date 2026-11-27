# Thesis brief: all results (full pipeline, batches 3-6)

This file is a briefing to give to Claude for writing the master's thesis. It contains what was done and measured up to 17 October 2026. If the code or the data changed after this date, the code and the data are authoritative.

## Instructions for the writer (Claude)

- Use **only** facts and numbers present in this file and in the two linked briefs (`ablation_sam3.md`, `thesis_brief_yolo_finetuning.md`). If a sentence is not supported by them, remove it or flag it with `[AUTHOR TO CONFIRM]`.
- Keep the `[AUTHOR TO CONFIRM]` markers already present visible in the text: they are facts I could not verify.
- Language: `[AUTHOR TO CONFIRM]`. The previous brief on segmentation asked for academic English.
- In the text, use neutral names for the runs and phases ("base model", "1376 px", "phase 1"). The folder names are only there for the author to find the results.
- Differences of about 0,003-0,008 mAP are below the seed-to-seed variability: they must be described as "within the noise", not as improvements.
- Always distinguish:
  - the segmentation metrics, measured against **SAM 3 pseudo-labels** and not against a manual ground truth;
  - the per-batch classification counts, which are **model output without validation** (see §8 and §9).

## 1. Goal and pipeline

The goal is to estimate hive health indicators from photos of frames: demographic profile (young bees, adults, drones) and presence of sick bees. The photos were taken in four sessions (batches 3-6).

The pipeline has six steps:

1. **Segmentation pseudo-labels** on the full photos with SAM 3, prompted with the text "bee".
2. **Fine-tuning of YOLO26s-seg** on these pseudo-labels, with an ablation on resolution, augmentation, freeze and seed.
3. **Extraction of the single bees** with the selected YOLO model, after normalising the brightness and contrast of each photo.
4. **Crop cleaning and normalisation:** quality filters, 512×512 crops with the long axis vertical, head at the top thanks to a linear classifier on DINOv2.
5. **Semi-supervised active learning** (`bee_al` package, with a web interface for annotation), in two phases:
   - phase 1: usable crops or crops to discard;
   - phase 2: demographic and disease classes, only on the usable crops.
6. **Per-batch counts** from the final classes, with bootstrap intervals over the photos.

A synthetic varroa generator on the crops was also developed for anomaly detection: see `brief_varroa_sintetica_su_crop.md`. I have no quantitative results on that work: `[AUTHOR TO CONFIRM: whether and how it goes into the thesis]`.

## 2. Data

- Photos of frames, folder `DatasetApi_Ceschi`, batches 3, 4, 5 and 6. `[AUTHOR TO CONFIRM: apiary, hives, session dates, camera]`.
- The 22 photos in `batch_6/favi_phone_camera`, taken with the phone, were removed from the pipeline. **791 photos** remain.

| Batch | Photos | Notes |
|---|---:|---|
| batch_3 | 168 | |
| batch_4 | 241 | |
| batch_5 | 199 | |
| batch_6 | 183 | 180 in `favi/` and 3 in `sick/` |

On the 3 photos in the `sick` folder the final model finds only 59 valid bees in phase 2, of which 2 are classified as Malata (sick). They therefore do not explain the increase of Malata in batch_6 (§9).

## 3. SAM 3 pseudo-labels

Script: `generate_labels_sam3.py`. Model: `facebook/sam3` (Hugging Face transformers), text prompt "bee". Output in `processed/bee_batch3+4+5+6_SAM3`, in the YOLO segmentation format.

SAM 3 works on tiles, with the scale adapted to each photo:

1. **Scale estimate.** The photo is divided into a 3×3 grid. Each cell, downscaled to 1008 px, goes through SAM 3 with threshold 0,5, and the median length of the bees found is taken. The whole thumbnail is not used because, in wide and crowded photos, SAM 3 saw no bees: 59 photos out of 791 had 0-5 boxes.
2. **Multi-scale tiling.** The photo is rescaled so that a bee is 120 px long and then 150 px long, with a scale factor between 0,05 and 5. Each version is divided into 500 px tiles with 40% overlap, and the results of the two scales are merged. With a single scale about 1% of the bees were left unlabelled, with two about 0,25%.
3. **Score thresholds.**
   - Above 0,5 all masks are kept: raising the threshold to 0,6 loses about 8% of the bees.
   - Between 0,3 and 0,5 a mask is kept only if its area is at least 60% of that of the median bee in the photo. Below 0,5 SAM 3 returns almost only heads of bees stuck in cells, cells with pollen and empty spaces.
   - Minimum area 600 px² in the tile and maximum area 25% of the tile.
4. **NMS on masks instead of boxes.** A mask is discarded when more than 60% of the smaller mask lies inside one with a higher score. NMS on boxes left about 5% duplicates and fragments.

The parameters were tuned by eye for recall, on 18 photos chosen to cover different zoom levels (bees from 150 to 680 px long) and different focus. **The labels were not corrected by hand** `[AUTHOR TO CONFIRM]`.

Total: 260.586 bees on 791 photos, about 329 per photo; the maximum is about 1.850 bees in one photo.

## 4. Segmentation: fine-tuning of YOLO26s-seg (ablation)

The full detail, with all tables and metrics, is in **`ablation_sam3.md`**: it must be attached together with this file. Here I report its summary.

### Split and setup
- **Random split by photo**:

  | Split | Photos | Bees |
  |---|---:|---:|
  | Train | 554 | 182.318 |
  | Validation | 118 | 39.076 |
  | Test | 119 | 39.192 |

- Pre-trained YOLO26s-seg, full fine-tuning.
- Hyperparameters:
  - up to 200 epochs, early stopping with patience 40;
  - batch 2;
  - AdamW, lr 0,001 with cosine decay down to 1%;
  - mosaic disabled, `max_det` 2.500.
- Hardware: one 48 GB RTX A6000 per run.
- The high training resolution is 1376 px, because 1440 px runs out of memory at batch 2.
- Test: conf 0,10, both at the training resolution and at 2048 px.
- Selection criterion: seg mAP50-95 on the test set at 2048 px.

### Results (test, 119 photos, at 2048 px)

| Variant | seg mAP50-95 | seg mAP50 | seg mAP75 | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| base, 1280 px | 0,688 | 0,926 | 0,798 | 0,924 | 0,882 | 0,902 |
| **1376 px (adopted)** | **0,692** | 0,927 | 0,803 | 0,931 | 0,881 | 0,905 |
| 1376 + rotation ±180° | 0,688 | 0,925 | 0,800 | 0,929 | 0,871 | 0,899 |
| 1376 + rotation + copy-paste | 0,690 | 0,919 | 0,802 | 0,931 | 0,873 | 0,901 |
| 1376, freeze 5 | 0,685 | 0,921 | 0,794 | 0,931 | 0,881 | 0,905 |
| 1376, freeze 10 | 0,682 | 0,918 | 0,792 | 0,926 | 0,874 | 0,899 |
| 1376, freeze 20 | 0,580 | 0,835 | 0,673 | 0,881 | 0,778 | 0,826 |
| 1376, freeze 23 | 0,484 | 0,760 | 0,554 | 0,777 | 0,701 | 0,737 |
| 1376, seed 1 | 0,695 | 0,931 | 0,805 | 0,941 | 0,884 | 0,912 |

### Main conclusions
1. **Inference at 2048 px:** worth +4/6 points of mAP50-95 for every variant, without retraining.
2. **Training at 1376 px instead of 1280:** +1,5 points at the training resolution, but at 2048 px only +0,4, which is within the seed-to-seed noise (0,003).
3. **Rotation and copy-paste:** no measurable effect.
4. **Freeze:** 5-10 layers cost about one point. From 20 layers upwards the model collapses (−11 and −21 points), especially in recall: full fine-tuning is necessary.
5. **mAP75** is the metric that best separates the good variants: what matters is contour precision, while detection is similar across variants.

### Comparison with the first ablation (LangSAM, batches 3+4)
It is described in `thesis_brief_yolo_finetuning.md`.
- **LangSAM:** 298 train photos, 56 test photos, high resolution 1440 px. Best 0,628 at 2048 px.
- **SAM 3:** best 0,692.

**The absolute values are not comparable.** Each model is evaluated against the pseudo-labels of the tool that generated its training labels, and the photos and number of images differ. The splits overlap: 40 photos of the LangSAM test set are in the SAM 3 train set, and 45 photos of the SAM 3 test set are in the LangSAM train set. Only the trends are comparable, and they agree.

**Adopted model:** `runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt`, used at 2048 px.

## 5. Extraction of the single bees

- Script: `extract_anomaly_dataset.py`, run with the adopted model, `--imgsz 2048` and `--conf 0.35` (default).
- Source: the 791 photos in `bee_batch3+4+5+6_SAM3/images`.
- **The 554 training photos of the YOLO model are included:** on those photos the segmentation is more optimistic than on new photos.
- **Per-photo normalisation** before cropping: linear transformation of the L channel in LAB, with a single gain and a single offset for the whole photo. This way the bees of the same photo remain comparable with each other.
  - The statistics are measured on the bee pixels (if there are at least 5.000), because the background changes from photo to photo.
  - Target: mean L 110 and standard deviation 42, i.e. the medians of 90 photos of batch3+4. Before the correction the mean ranged between 72 and 139 and the standard deviation between 33 and 51.
  - Gain limited between 0,6 and 1,8.
- Result: **249.982 raw crops** (RGBA PNG) in `processed/single_bee3+4+5+6_SAM3/raw_bees`.

## 6. Crop cleaning and orientation

Script: `dataset_cleaning.py`, pipeline version 4. Output in `processed/single_bee3+4+5+6_SAM3/cleaned_bees`.

### Filters (rejection reasons)

| Reason | Rule | Bees rejected |
|---|---|---:|
| border | bee cut off by the crop border: more than 15 px of mask in an 8 px band | 31.917 |
| blur | sharpness score < 50, measured at a common scale | 25.015 |
| short | long side of the rotated rectangle < 105 px: below about 100 px they were abdomens in cells, heads or wings | 9.388 |
| aspect | length/width ratio > 3 | 205 |
| fragments | keeping only the main body loses more than 20% of the mask | 60 |
| **Total rejected** | | **66.585** (26,6%) |
| **Accepted** | | **183.397** |

Other filter thresholds: minimum area 1.500 px², fill ratio ≥ 0,30, short side ≥ 25 px.

| Batch | Photos | Crops extracted | Accepted | % accepted |
|---|---:|---:|---:|---:|
| batch_3 | 168 | 45.182 | 34.131 | 75,5% |
| batch_4 | 241 | 99.639 | 73.611 | 73,9% |
| batch_5 | 199 | 58.914 | 41.805 | 71,0% |
| batch_6 | 183 | 46.247 | 33.850 | 73,2% |

### Normalisation
- Each crop is a 512×512 PNG on a black background, with the long axis of the bee vertical and the long side equal to about 85% of the frame.
- The mask goes through closing, dilation, opening (which separates adjacent bees), recovery of legs and antennae and smoothing; the edges are feathered over 12 px.
- The kernels are proportional to the bee length, with a reference of 190 px.

### Head at the top (`bee_orientation.py`)
- Method: logistic regression (C = 0,1) on DINOv2 ViT-S/14 embeddings. The input is the difference between the embedding of the bee and that of the same bee rotated by 180°: the score is therefore antisymmetric by construction and the step is idempotent.
- Labels: 214 bees from batches 3-6 oriented by hand (`config/head_orientation_labels.json`), ambiguous ones excluded.
- **10-fold cross-validation: 0 errors out of 214.** Ambiguous bees (side views, partial) are not among the labelled ones, so some errors are to be expected there.
- On the 183.397 accepted bees:
  - **57.501 were rotated** by 180° (31,4%);
  - **9.164 have |score| < 1**, i.e. uncertain orientation (5,0%). With |score| < 2 they become 18.572, with |score| < 3 28.836.
  - The uncertain bees were not re-checked by hand.

## 7. Active learning method (`bee_al`)

Full description in `bee_al/README.md`.

### Representation
- **Embeddings:** DINOv2 ViT-S/14 at 224 px. No classification head: classes are assigned from the neighbours in the embedding space.

### Class assignment
- **Label propagation:** label spreading (Zhou et al.) on a kNN graph with 15 neighbours, α = 0,8, at most 30 iterations.
- **Fallback kNN** (k = 10, temperature 0,07).
- **Unassigned:** the 5% of bees furthest from any labelled example stays "unassigned" and does not enter the clusters.

### Selection of the bees to label
- Batches of 30 bees.
- 70% by uncertainty: margin between the top two classes of the propagation.
- 20% by novelty and 10% random.
- Cold start below 50 labels.

### Interface
The web interface offers three ways of labelling:
- active learning queue, single or grid;
- Clusters page, where groups of similar bees are labelled by hand;
- Review page, with labels that are suspicious because they disagree with their neighbours.

There is also a "Scarta ape" (discard bee) button for bees that one does not want to assign.

### Contrastive re-training (on demand, via a button)
- Fine-tuning of the last 2 blocks of DINOv2, 300 steps, batch 96.
- 50% of each batch is labelled; pseudo-labels enter with confidence ≥ 0,95.
- Loss: SupCon on labelled and pseudo-labelled bees, SimCLR on the others. Temperature 0,1, projection to 128 dimensions.
- Learning rate: 1e-5 for the backbone, 1e-3 for the projection head.
- After re-training, the embeddings and the graph are regenerated.

### Validation set
100 random bees, fixed (seed 123), excluded from the queue. **It was not labelled in either phase**, so accuracy and macro-F1 are not available and the models were accepted without a metric comparison.

### Stopping criteria
- Prediction stability: less than 1% of bees change class for 3 rounds.
- No new class for 3 rounds.
- F1 plateau on the validation set.

### Snapshots
The state is frozen in immutable snapshots (`predictions.csv`, `frames.csv`, `meta.json`, labels) and the clusters are exported into per-class folders.

## 8. Phase 1: usable crops or crops to discard

- **Data:** all 183.397 cleaned bees. Snapshot `fase1_finale_okvsscarta` of 15/10/2026, model v003, 14 rounds.
- **Classes:** Ok (usable crop) and Scarta (discard: segmentation or crop not usable).
- **Annotation:** a single annotator (the author), 1.735 label events and 1.561 labelled bees.
  - By query type: manual 1.505, uncertainty 112, diversity 58, novelty 39, random 21.
  - **Almost all labels were chosen by the annotator from the Clusters page, not sampled at random.**
- **Contrastive re-trainings:**

  | Model | Labels | Pseudo-labels | Initial loss | Final loss | Duration |
  |---|---:|---:|---:|---:|---:|
  | v001 | 386 | 35.899 | 4,20 | 3,01 | 8 min |
  | v002 | 1.070 | 9.037 | 4,20 | 3,07 | 8 min |
  | v003 | 1.537 | 20.237 | 4,18 | 3,06 | 8 min |

- **Result:**
  - Ok 138.788 (824 by hand, 137.964 predicted);
  - Scarta 35.422 (737 by hand, 34.685 predicted);
  - 9.087 unassigned and 100 in the validation set.
- **Stopping criteria:**
  - no new class: met;
  - stability: **not met**, in the last 3 rounds 3,15%, 1,85% and 1,96% of the bees changed class;
  - F1: cannot be evaluated, because the validation set is missing.

Share of Ok per batch (`scripts/batch_counts.py`). The shares are over the assigned bees, with a 95% bootstrap interval over the photos:

| Batch | Photos | Bees | Ok | Scarta | Unassigned |
|---|---:|---:|---|---|---:|
| batch_3 | 168 | 34.131 | 26.566 (81,4% [78,4-84,8]) | 6.054 (18,6%) | 1.511 |
| batch_4 | 241 | 73.611 | 55.879 (79,1% [77,2-80,7]) | 14.783 (20,9%) | 2.949 |
| batch_5 | 199 | 41.805 | 29.777 (76,8% [75,0-78,9]) | 8.980 (23,2%) | 3.048 |
| batch_6 | 183 | 33.850 | 26.640 (82,6% [80,5-84,9]) | 5.631 (17,4%) | 1.579 |

The bootstrap measures the photo-to-photo variability, **not the model errors**.

**Previous experience (pilot on batches 3+4, archived).** There had been a first run with LangSAM labels, 68.559 bees and the classes Crop Utilizzabile / Ape Incompleta / Crop Non Utilizzabile (usable crop / incomplete bee / unusable crop). There the validation set had been labelled, and it showed that the model let through about 2/3 of the unusable crops. This is why phase 1 exists as a separate step. `[AUTHOR TO CONFIRM: whether to mention the pilot in the thesis]`

## 9. Phase 2: demographic profile and disease

- **Data:** only the 138.788 Ok bees from phase 1, with the original DINOv2 embeddings (v000) as the starting point.
- **Snapshot** `fase2_finale` of 16/10/2026: model v004, 9 rounds. Export in `single_bee3+4+5+6_SAM3_clusters/fase2_finale`.
- **Classes**, as defined by the author:
  - **Giovane, Adulte, Fuco** (young, adult, drone): demographic profile.
  - **Malata** (sick): sick bees in general.
  - **Non Classificabile** (unclassifiable): bees that cannot be classified because of segmentation or pose. It is a reject class: the model learns it, but it is left out of the shares.
  - Visual criteria used for age: `[AUTHOR TO CONFIRM]`. One indication given during the work was thorax hair and wing condition.
- **Annotation:** one annotator, 1.954 events and 1.837 labelled bees.
  - By query type: manual 1.786, diversity 65, uncertainty 61, novelty 25, random 17.
- **Contrastive re-trainings:**

  | Model | Labels | Pseudo-labels | Initial loss | Final loss |
  |---|---:|---:|---:|---:|
  | v001 | 607 | 16.711 | 4,13 | 2,31 |
  | v002 | 1.019 | 3.184 | 4,04 | 2,37 |
  | v003 | 1.338 | 2.684 | 4,07 | 2,37 |
  | v004 | 1.685 | 1.745 | 4,02 | 2,39 |

  The pseudo-labels above 0,95 decrease from one model to the next.
- **Final classes:**

  | Class | By hand | Predicted | Total | % of the 131.829 bees in the clusters |
  |---|---:|---:|---:|---:|
  | Adulte | 1.250 | 122.432 | 123.682 | 93,8% |
  | Non Classificabile | 129 | 3.821 | 3.950 | 3,0% |
  | Giovane | 99 | 1.994 | 2.093 | 1,6% |
  | Fuco | 130 | 1.271 | 1.401 | 1,1% |
  | Malata | 229 | 474 | 703 | 0,5% |

  In addition, 6.842 unassigned, 100 in the validation set and 17 "Scarta ape".
- **Stopping criteria:**
  - no new class: met;
  - stability: **far from the threshold**, in the last 3 rounds 9,49%, 13,92% and 9,14% of the bees changed class;
  - F1: cannot be evaluated.

**Per-batch counts** (`batch_counts.py`, bootstrap over the photos, 1.000 replicates).
- The profile shares are computed over the total Giovane + Adulte + Fuco.
- Malata is a % of all classified bees, i.e. excluding Non Classificabile and the unassigned ones.

| Batch | Photos | Bees | Giovane (% profile) | Adulte (% profile) | Fuco (% profile) | Malata (% classified) | Non Classificabile | Unassigned |
|---|---:|---:|---|---|---|---|---:|---:|
| batch_3 | 168 | 26.548 | 1,0% [0,8-1,2] | 98,7% [98,5-99,0] | 0,3% [0,2-0,3] | 0,2% [0,1-0,2] (42) | 685 | 1.392 |
| batch_4 | 241 | 55.850 | 2,1% [1,8-2,4] | 96,3% [95,9-96,7] | 1,6% [1,4-1,7] | 0,4% [0,3-0,4] (198) | 1.713 | 2.303 |
| batch_5 | 199 | 29.761 | 1,5% [1,3-1,8] | 97,2% [96,7-97,6] | 1,3% [1,0-1,6] | 0,4% [0,3-0,5] (109) | 898 | 1.921 |
| batch_6 | 183 | 26.629 | 1,5% [1,3-1,7] | 97,8% [97,5-98,0] | 0,8% [0,6-0,9] | 1,4% [1,1-1,8] (354) | 658 | 1.226 |

batch_6 has a Malata share about 3-7 times higher than the others. The photos with the most Malata bees are all in batch_6 (DSC_6797: 19, DSC_6796: 18, DSC_6785: 16). They do not come from the `sick` folder (§2).

### How to read these numbers (to be written in the thesis)
1. **They are not validated.** The validation set of 100 random bees was not labelled, and the blind audit was not done (`bee_al/audit.py`, `scripts/audit_report.py`, available but not used). The intervals only show the variability between photos.
2. **Likely underestimation of the rare classes.**
   - Graph propagation favours the class with the most examples: Adulte has 1.250 examples, the others 99-229.
   - For Malata, one third of the total (229 out of 703) are the bees labelled by hand. The model therefore found only 474 more out of about 130.000 bees.
   - The shares of Giovane, Fuco and Malata should be read as lower bounds or as a relative comparison between batches, not as absolute prevalences.
3. **Non-random label sample.** 91% of the labels were chosen by the annotator from the Clusters page. They cannot be used to estimate accuracy.
4. **Model not stabilised.** Between one round and the next, 9-14% of the bees change class.
5. **A single annotator**, with no measure of inter-annotator agreement.
6. **The phase 2 classes were defined after the snapshot.** The `fase2_finale` snapshot was taken with an empty class list in the config. Reject, demographic and disease classes were applied afterwards, at counting time, by reading `config/al_fase2_sam3.yaml`. The labels and predictions do not change.
7. **Cascade effect of phase 1.** Bees wrongly predicted as "Scarta" do not reach phase 2, and unusable "Ok" bees partly ended up in Non Classificabile.

Steps that would make the numbers defensible, not yet done: `[AUTHOR TO CONFIRM whether to do them before submission]`
- label the 100 bees of the phase 2 validation set;
- run the confidence-stratified audit and obtain corrected prevalence estimates;
- possibly balance the class masses in the propagation.

## 10. General limitations (limitations section of the thesis)

- **Segmentation**
  - Ground truth = SAM 3 pseudo-labels not corrected by hand.
  - Random split by photo and not by shooting sequence or comb: nearly identical photos can be one in train and one in test. The absolute metrics may therefore be optimistic, but the comparison between variants remains valid.
  - Only one repeated seed. Some runs (freeze10, seed1) were resumed from checkpoints after out-of-memory errors caused by other GPU users; the tested `best.pt` files predate the resumes.
- **Extraction:** also done on the training photos of the YOLO model.
- **Cleaning:**
  - thresholds chosen by eye;
  - there is no measure of how many good bees are rejected;
  - 5% of uncertain orientations not re-checked.
- **Classification:** see §9, points 1-7.
- **Metadata:** there is no photo → hive mapping. The "session" coincides with the batch, so the effect of time cannot be separated from that of the hive `[AUTHOR TO CONFIRM: hives photographed in each batch]`.

## 11. Files and reproducibility

| What | Where |
|---|---|
| SAM 3 ablation, full detail | `ablation_sam3.md`, `logs/ablation_sam3/` |
| LangSAM ablation (previous) | `thesis_brief_yolo_finetuning.md` |
| SAM 3 pseudo-labels | `generate_labels_sam3.py`, `processed/bee_batch3+4+5+6_SAM3`, `logs/generate_labels_sam3.log` |
| Split | `split_dataset_sam3.py`, `data_sam3.yaml` |
| Extraction | `extract_anomaly_dataset.py`, `logs/extract_single_bee3+4+5+6_SAM3.log` |
| Cleaning and orientation | `dataset_cleaning.py`, `bee_orientation.py`, `logs/cleaning_single_bee3+4+5+6_SAM3.log`, manifest `cleaned_bees/_cleaning_manifest.json` |
| Active learning | `bee_al/` (README), config `config/al_config.yaml` (phase 1) and `config/al_fase2_sam3.yaml` (phase 2) |
| Phase 1 snapshot | `/mnt/disk1/tommaso_ballarin/bee_al/results_sam3_b3456/snapshots/fase1_finale_okvsscarta/` (with `batch_counts/`) |
| Phase 2 snapshot | `/mnt/disk1/tommaso_ballarin/bee_al/results_fase2_sam3/snapshots/fase2_finale/` (with `batch_counts/`: CSVs per batch and per photo, plot `batch_counts.png`) |
| Exported clusters | `processed/single_bee3+4+5+6_SAM3_clusters/{fase1_finale_okvsscarta, fase2_finale}` |
| Contrastive models | `/mnt/disk1/tommaso_ballarin/bee_al/models_{sam3_b3456,fase2_sam3}/v00N/` (`meta.json`, `train_log.json`) |
| Synthetic varroa | `brief_varroa_sintetica_su_crop.md`, `varroa_on_crops.py`, `build_varroa_split.py` |

Software: Ultralytics 8.4.138, PyTorch 2.14.0, Python 3.10, 48 GB NVIDIA RTX A6000 GPU (shared server).

The ready-made plots are `batch_counts/batch_counts.png` of the two snapshots and the Ultralytics ones in the run folders (`results.png`, PR curves).
