# Thesis brief: photos taken with the phone (domain shift)

This is a briefing to give to Claude for writing the master's thesis. It complements `pipeline_results.md` (full pipeline on batches 3-6) and contains what was done and measured on 23 October 2026 on the 22 phone photos. If the code or the data changed after this date, the code and the data are authoritative.

## Instructions for the writer (Claude)

- The same rules as in `pipeline_results.md` apply:
  - only facts and numbers present here;
  - the `[AUTHOR TO CONFIRM]` markers stay visible;
  - small differences are described as "within the noise".
- The segmentation metrics are measured against the **SAM 3 pseudo-labels**, not against a manual ground truth.
- The class shares are **model output, not validated**: the validation set of the dedicated phase was not labelled (§6).
- All numbers are in `visualizations/thesis_telefono/metrics.json`. The figures are in the same folder.

## 1. Why this experiment

The YOLO model and phase 2 of the active learning were built on photos from a camera, in batches 3-6. Knowing how the pipeline behaves with a phone matters for practical use in the apiary. Questions:

1. Does the segmentation hold up when the device changes?
2. Does the DINOv2 space of the active learning recognise the phone bees as similar to those already labelled?
3. Is a dedicated annotation phase, with few examples, enough to organise them into classes?

## 2. Data

- **22 photos** in `batch_6/favi_phone_camera`, taken on 11/10/2026 between 17:59 and 18:22 (from the file names).
  - 21 photos are 4000×3000 and one is 4608×3456.
  - 4 photos have EXIF orientation 3, i.e. they are rotated by 180°.
  - `[AUTHOR TO CONFIRM: phone model, hive, whether they are the same combs photographed in batch_6]`.
- **They were never used** to train YOLO: they had been removed from the 791 photos of the pipeline (`pipeline_results.md` §2).
- **Labels:** SAM 3 pseudo-labels in `processed/phone_captured`, produced with the same procedure as batches 3-6. In total 6.959 bees, on average 316 per photo.
- **Bee size.**
  - Median bee length: 229 px on 4000 px photos from the phone, 264 px on 6016 px photos in batches 3-6.
  - Rescaled to 2048 px for YOLO, the phone bees are about 1,5 times larger: 117 px versus 90 px.
  - The photo `IMG_20261011_180019` is a close-up with much larger bees.

**Problem found in the pseudo-labels.**
- In the 4 photos with EXIF 3, the SAM 3 labels were rotated by 180° with respect to the image that YOLO reads. YOLO, through OpenCV, applies the EXIF orientation.
- The original labels were not modified. The metrics below use a corrected copy (x→1−x, y→1−y), in `/mnt/disk1/tommaso_ballarin/phone_captured_labels_ruotate`.
- The metrics with the wrong labels are reported only for completeness. They should at most be cited as a methodological note: the EXIF orientation must always be checked when generating labels.

## 3. Segmentation: YOLO on the phone photos

- **Model:** the best one from the ablation, YOLO26s-seg trained at 1376 px (`pipeline_results.md` §4).
- **Protocol:** the same as for the test set, i.e. `conf=0.10` and `max_det=2500` (`logs/phone/phone_eval_data.py`).

| Data | Inference | seg mAP50-95 | seg mAP50 | seg mAP75 | P | R |
|---|---|---:|---:|---:|---:|---:|
| Test batches 3-6 (119 photos, reference) | 2048 px | 0,692 | 0,927 | 0,803 | 0,931 | 0,881 |
| Phone (22 photos) | 2048 px | 0,527 | 0,804 | 0,589 | 0,884 | 0,765 |
| Phone | 1376 px | 0,529 | 0,828 | 0,584 | 0,908 | 0,789 |
| Phone, base model 1280 px | 1280 px | 0,528 | 0,827 | 0,581 | 0,903 | 0,785 |
| *Phone, uncorrected labels (do not use)* | 2048 px | 0,425 | 0,651 | 0,472 | 0,793 | 0,677 |

- **The drop due to the domain shift is about 0,16 mAP50-95**, from 0,692 to 0,527.
- On the phone, the three models or resolutions are **within the noise**: between 0,527 and 0,529.
- On the batch 3-6 test set, inference at 2048 px gave +0,046 compared with 1376 px. On the phone this advantage disappears. A compatible explanation, not verified, is that at 2048 px the phone bees are larger than those seen in training (§2).

**Mask-by-mask matching with SAM 3** (IoU ≥ 0,5, greedy by confidence; `logs/phone/phone_compare.py`):
- **At the threshold used for extraction (conf ≥ 0,35):** YOLO finds 5.768 bees, of which 5.184 are matched. **Precision is 0,90**, **recall is 0,74**.
- **At the evaluation threshold (conf ≥ 0,10):** precision 0,68, recall 0,83.
- The per-photo median of the mean IoU of the matched bees is 0,83. When YOLO finds a bee, the mask is good: the drop comes mostly from missed bees.
- **Figure 1** (`fig1_seg_per_foto.png`): precision and recall per photo. 19 photos out of 22 have recall between 0,63 and 0,87. The worst are:
  - the close-up `180019`: recall 0,22, YOLO finds 34 bees out of 120;
  - `175932` and `180633`: recall 0,54 and 0,59.
- Visual examples in `visualizations/phone_yolo_vs_sam3_ruotate/`: SAM 3 on the left, YOLO on the right, in red the bees the other one does not have.
  - By eye, the bees missed by YOLO are mostly bees with open wings, overlapping or on the edge of the photo.
  - The extra bees found by YOLO are partly omissions by SAM 3: the pseudo-labels are not a perfect truth.

## 4. Extraction and cleaning

Pipeline identical to batches 3-6: extraction at 2048 px with conf 0,35, per-photo normalisation, cleaning v4 with head at the top. Output in `processed/single_bee_phone_SAM3/`.

- **Bees:** 5.768 raw crops, 4.689 accepted (81,3%).
- **Rejection reasons:**
  - border 915;
  - too short 159;
  - aspect ratio 4;
  - blur 1.
- **The blur filter almost never triggers, even though by eye many phone crops are blurred or "smeared"** (see the figures with the crops). The threshold was tuned on the camera.
- Head at the top: 1.547 bees out of 4.689 rotated by 180°.

## 5. Domain distance in the DINOv2 space

The phone bees were passed to the **phase 1 model** (Ok / Scarta, embedding v003, trained on batches 3-6) with the same criterion used for batch 7 (`scripts/add_dataset_to_stage.py`).

- A bee is "unassigned" when its cosine similarity with the nearest labelled bee does not exceed **0,674**. This is the threshold that separated the unassigned bees in phase 1.

| Data | Bees | Ok | Scarta | Unassigned | Median similarity to nearest labelled |
|---|---:|---:|---:|---:|---:|
| batches 3-6 (phase 1 data) | 183.397 | | | 5,0% | 0,764 |
| batch 7 (camera, new session) | 44.810 | 34.645 | 8.108 | **4,6%** | 0,766 |
| phone | 4.689 | 895 | 167 | **77,4%** | 0,626 |

- **Figure 2** (`fig2_similarita_dominio.png`): similarity distribution for batches 3-6 and for the phone. The phone distribution is shifted to the left, largely below the threshold. Batch 7 is not in the figure, but it has a distribution almost identical to batches 3-6 (median 0,766 versus 0,764, table above).
- **Interpretation:** a new session with the same camera (batch 7) is not a domain shift for DINOv2. The phone is.
- This shows that the share of "unassigned" bees is a simple, label-free indicator for spotting a domain shift.

The 895 Ok bees were also projected into the phase 2 of batches 3-6, as a held-out dataset (`embedding_map/fase2_v004_b3456_b7_phone/solo_phone`). There:
- 33% remain unassigned, versus 4,5% for batch 7 and 4,9% for batches 3-6;
- 65% go to Adulte.

Even the phone bees that look good remain far from the camera examples.

## 6. Dedicated phase: active learning on the phone only

To see whether a few labels are enough, a separate annotation phase was run (`config/al_fase2_phone.yaml`, separate hub):
- on **all 4.689 cleaned bees**, without the phase 1 filter;
- **from scratch**, with the original DINOv2 embeddings and no labels;
- with the same classes as phase 2: Giovane, Adulte, Fuco, Malata, Non Classificabile (reject class).

**Snapshot** `foto_telefono` of 23/10/2026 14:21: model v002, 4 rounds, about 20 minutes of annotation (first label 13:58, last 14:17).

**Annotation:** one annotator (the author), 200 labelled bees.
- By class: Non Classificabile 94, Adulte 77, Giovane 18, Malata 11, Fuco 0.
- By query type: manual 151 (chosen from the Clusters page), diversity 44, random 5.
- Review of suspicious labels: 13 checked, all confirmed. The model always suggested Adulte; the annotator kept Non Classificabile (6), Giovane (4) and Malata (3).

**Contrastive re-trainings** (supervised contrastive + SimCLR, 300 steps):

| Model | Labels | Pseudo-labels | Initial loss | Final loss | Duration |
|---|---:|---:|---:|---:|---:|
| v001 | 178 | 90 | 3,94 | 2,20 | 72 s |
| v002 | 200 | 110 | 3,89 | 2,27 | 69 s |

**Stability:** 18,7% of the bees change class from v000 to v001 and 6,0% from v001 to v002. It has not stabilised: the stopping-criterion threshold is 1%.

**Final classes** (`batch_counts.py`; validation-set bees are counted with their predicted class):

| Class | Bees | % of the 4.689 | % of the classified (3.215) |
|---|---:|---:|---:|
| Adulte | 3.115 | 66,4% | 96,9% [96,2-97,7] |
| Giovane | 87 | 1,9% | 2,7% [2,1-3,3] |
| Malata | 13 | 0,3% | 0,4% [0,2-0,7] |
| Fuco | 0 | 0 | 0 |
| Non Classificabile | 1.255 | 26,8% | (reject) |
| Unassigned | 219 | 4,7% | |

The intervals come from the bootstrap over the 22 photos and **do not measure the model errors**.

**Comparison with batches 3-6** (phase 2, `pipeline_results.md` §9):

| | batches 3-6 | phone |
|---|---|---|
| Non Classificabile, % of all bees | 3,0% | **26,8%** |
| Unassigned | 4,9% | 4,7% |
| Giovane, % profile | 1,0-2,1% | 2,7% |
| Fuco, % profile | 0,3-1,6% | 0% (no labels) |
| Malata, % classified | 0,2-1,4% | 0,4% |

- **After the dedicated phase the unassigned bees return to the batch 3-6 level**, from 77% to 4,7%. With about 200 labels from the same domain, the DINOv2 space covers the phone bees again.
- **About one bee in four is Non Classificabile**, versus 3% for the camera. This is consistent with the visual quality of the crops (blurred, smeared) and with the more crowded photos. It should be read together with point 2 of the limitations (§7).
- The numbers for Giovane, Malata and Fuco **are not comparable** with batches 3-6. The labels are few (18 and 11) and there are none for Fuco, so the model cannot find any.

**Figures:**
- **Figure 3** (`fig3_fase1_vs_fase_telefono.png`) compares the judgement of the camera model (phase 1 filter) with the class assigned in the dedicated phase, on the same bees:
  - of the **895 Ok** according to the camera model, 94% are Adulte in the dedicated phase;
  - of the **167 Scarta**, 57% are Non Classificabile;
  - of the **3.627 unassigned**, 61% are Adulte and 31% Non Classificabile.
  
  So the phase 1 filter, when it decides, is consistent with the dedicated annotation. But it also discarded as "unassigned" about 2.200 bees that, with phone labels, turn out to be Adulte. **Using the camera model on the phone would have lost almost three quarters of the data.**
- **Figure 4** (`fig4_composizione_per_foto.png`): classes per photo. Non Classificabile ranges from 11% to 46%. It is highest in the close-up `180019` and in the last photos of the sequence (`182238`, `182242`: 44-45%).
- **UMAP map** (`map_overview.png`, `map_by_cluster.png`; embedding v002, 4.689 bees):
  - Non Classificabile occupies its own region, on the right;
  - Adulte occupies the rest;
  - Giovane is scattered, with a small group at the bottom;
  - the unassigned bees lie mostly at the margins of the Non Classificabile region.
- **Example crops:** `crops_<classe>.png`, 24 random ones per class.

**Internal consistency of the labels** (leave-one-out k-NN on the 200 labelled bees, k=10, like the hub's kNN):

| Embedding | Accuracy | macro-F1 | F1 Adulte | F1 Giovane | F1 Malata | F1 Non Class. |
|---|---:|---:|---:|---:|---:|---:|
| original DINOv2 (v000) | 0,69 | 0,47 | 0,71 | 0,10 | 0,31 | 0,76 |
| after re-training (v002) | 0,72 | 0,51 | 0,75 | 0,09 | 0,40 | 0,80 |

- Giovane is almost always confused with Adulte: 17 out of 18 with v000. Malata is taken for Adulte 7 times out of 11.
- **This is not an accuracy estimate**:
  - the labels were almost all chosen by hand from the Clusters page;
  - the v002 row is optimistic, because the re-training saw exactly these labels.
- It only serves to say that, with these few labels, DINOv2 separates Adulte from Non Classificabile well but **does not distinguish age or disease**.

## 7. Limitations (to be reported)

1. **Segmentation against pseudo-labels.** The SAM 3 pseudo-labels are not corrected by hand, and for 4 photos they were also rotated (§2). The metrics are "agreement with SAM 3", not accuracy.
2. **Classes not validated.**
   - The validation set (100 random bees) was not labelled and there is no audit.
   - 76% of the labels were chosen by the annotator, so they are not a random sample.
   - 47% of the labels are Non Classificabile. This may push the propagation towards that class.
3. **Model not stabilised:** 6% of the bees change class in the last round.
4. **Few photos:** 22, from a single session and a single phone `[AUTHOR TO CONFIRM]`. The effect of the device cannot be separated from that of light, distance and comb.
5. **A single annotator**, no measure of agreement.
6. **Cleaning tuned on the camera.** The blur filter does not reject the blurred phone crops: they end up in Non Classificabile instead of being rejected by the cleaning.

## 8. Messages for the thesis

- The **segmentation transfers partially**: −0,16 mAP50-95. The masks found are good (median IoU 0,83), but about one bee in four is missed. The problem gets worse with close-ups.
- In DINOv2 **the phone is a different domain; the same camera in another session is not** (batch 7). The share of unassigned bees (77% versus 5%) signals this without labels.
- **A small dedicated annotation phase** (200 labels, about 20 minutes) is enough to cover the domain again: the unassigned bees drop from 77% to 5%. It is not enough, however, for the fine-grained classes (Giovane, Malata, Fuco).
- With the phone **about a quarter of the crops cannot be classified**, versus 3% for the camera. For field use, image quality matters more than the model.

## 9. Files and reproducibility

| What | Where |
|---|---|
| YOLO evaluation | `logs/phone/phone_eval_data.py`, `logs/phone/phone_eval_ruotate.json` (corrected labels), `logs/phone/phone_eval.json` (original labels), `logs/phone/data_phone*.yaml` |
| Per-photo comparison with SAM 3 | `logs/phone/phone_compare.py`, `logs/phone/phone_compare_ruotate.csv`, images in `visualizations/phone_yolo_vs_sam3_ruotate/` |
| Corrected SAM 3 labels | `/mnt/disk1/tommaso_ballarin/phone_captured_labels_ruotate/` (with `LEGGIMI.txt`) |
| Extraction and cleaning | `logs/phone/run_extract_clean_phone.sh`, `logs/phone/extract_clean_phone.log`, `processed/single_bee_phone_SAM3/` |
| Phase 1 filter | `/mnt/disk1/tommaso_ballarin/bee_al/results_sam3_b3456/added_phone_predictions.csv` (and `added_b7_predictions.csv`) |
| Phone in the phase 2 of batches 3-6 | dataset `phone` in `config/al_fase2_sam3.yaml` (held out), map in `results_fase2_sam3/embedding_map/fase2_v004_b3456_b7_phone/` |
| Dedicated phase | `config/al_fase2_phone.yaml`, snapshot `/mnt/disk1/tommaso_ballarin/bee_al/results_fase2_phone/snapshots/foto_telefono/` (with `batch_counts/` and `report_tesi/`), models in `models_fase2_phone/` |
| Figures and metrics of this brief | `logs/phone/thesis_report_phone.py` → `visualizations/thesis_telefono/` (copy of `report_tesi/`) |
