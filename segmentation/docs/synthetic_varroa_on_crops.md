# Brief: how synthetic varroa is added (crop-based version)

Context document for Claude. It describes the current method used to generate
"anomalous" bees with synthetic varroa for anomaly detection. Written on 8 October 2026:
if the code changed afterwards, the code is authoritative.

## In short

The varroa is pasted **directly onto the crop of the single, already segmented bee**
(512×512 PNG on a black background), no longer onto the full comb image. For each
anomalous bee a binary varroa mask is also saved, to be used as ground truth
for the pixel-level metrics.

## What changed compared to before

| | Previous method | Current method |
|---|---|---|
| Where it is pasted | full comb image | 512×512 crop of the single bee |
| Scripts | `varroa_generator.py` + `extract_synthetic_anomaly_dataset.py` | `varroa_on_crops.py` + `build_varroa_split.py` |
| Segmentation | YOLO runs *after* insertion, then `dataset_cleaning.clean_bee` | none: the crops are already segmented and cleaned |
| GT mask | mask on the comb, transformed with the `clean_bee` affine and intersected with the bee | generated directly in the crop frame |
| Position | inside the bee polygon, most saturated/darkest area | on the body only (see below) |

Consequences to keep in mind:

- The varroa no longer goes through YOLO or through the cleaning, so it cannot alter the
  segmentation and is never cut off by the bee mask.
- The varroa no longer undergoes the same resizing as the bee (the crops are small
  bees upscaled to 512). This is why sharpness and grain matching is
  essential: without it, the varroa is visibly sharper than the bee.

The two old scripts were not modified and remain usable.
`varroa_on_crops.py` imports the shared functions from `varroa_generator.py`
(source loading, colour adaptation, shadows, compositing).

## Files

- `varroa_on_crops.py` — insertion on a single crop (`add_varroa_to_crop`) and
  CLI for visual checks with a comparison sheet.
- `build_varroa_split.py` — construction of the train/test dataset.
- `tests/test_varroa_split.py` — tests on counts and splitting.
- Varroa sources: `varroa_alpha.png` and `varroa_alpha2.png` (the third one, `varroa_alpha3.png`, was removed on 9 October 2026) in
  `/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/` (RGBA PNG).
- Input crops (default): `.../processed/train_single_bee_batch3+4_clusters/Prova1/Crop_Utilizzabile`
  (54.182 PNGs, 512×512, long axis of the bee vertical, head at the top **or** at the bottom).

## Insertion on a crop, step by step

Function: `varroa_on_crops.add_varroa_to_crop(crop, varroa_sources, ratio_range, ..., num_varroa=None)`.
Returns `(synthetic_crop, gt_mask, zone)`; the first two are `None` on failure.

### 1. Varroa sources (`vg.load_varroa_sources`)
For each PNG: alpha binarised at 128, eroded to remove the light fringe left by the
cut-out, morphological opening to remove legs and hairs, only the largest contour
kept and filled. Cropped to the body, holes filled with inpainting, body
normalised to a width of 128 px.

### 2. Bee mask (`bee_mask_from_crop`)
Pixels with maximum channel > 10, largest connected component, holes filled.

### 3. Allowed zone (`body_zone`)
The varroa may only sit on the thorax and abdomen, up to just below the eyes.

1. **Thick part**: morphological opening of the mask with radius = 60% of the
   maximum thickness (distance transform). Removes open wings, legs, antennae.
2. **Body axis**: principal direction (PCA) of the thick part.
3. **Excluded ends**: 22% of the length at each end. The head cannot be
   told apart from the abdomen, so both ends are excluded: this removes the
   eyes, but also the tip of the abdomen.
4. **Central band**: distance from the axis ≤ 70% of the body half-width.
5. **Light wings**: pixels with saturation < 75 and value > 125 (HSV, after
   blurring) are excluded, i.e. translucent membrane and reflections.

### 4. Size and rotation
- Varroa width = bee width × U(0.20, 0.24). The bee width is the
  short side of the minimum oriented rectangle of the **whole** mask (wings included).
- Uniform rotation 0–360°.
- The binary mask (threshold 128, largest component), tightly cropped, is derived
  from the rotated varroa: this is what becomes the GT.

### 5. Position (`find_position_in_mask`)
Up to 200 candidate centres sampled at random among the zone pixels. A candidate is
discarded if it goes outside the frame, if it overlaps an already inserted varroa or if the
varroa does not fall **100%** inside the zone. Among the valid ones, the one with the highest
score wins: `mean saturation + 0.5 × (255 − mean value)` under the varroa, i.e.
saturated, dark areas.

### 6. Colour adaptation (`vg.adapt_color_to_region`, LAB space)
- mean lightness shifted towards that of the bee (weight 0.60);
- contrast brought closer to that of the bee (0.45);
- chrominance shifted towards that of the bee (0.45);
- the varroa inherits the local highlights and shadows of the underlying bee (0.60).

The statistics are computed on a copy of the crop in which the black background is
replaced by the mean colour of the bee, so the black does not skew them.

### 7. Sharpness and grain (`match_sharpness_and_noise` in `varroa_on_crops.py`)
- Sharpness is the 90th percentile of the Laplacian. The reference sharpness is measured
  around the varroa (3× its rectangle), on bee pixels only.
- The varroa is blurred with increasing sigma (from 0.6, step 0.25) until it drops to the
  reference sharpness, capped at `0.12 × sqrt(varroa area)`. Alpha is blurred with
  the same sigma.
- Grain: standard deviation estimated from the crop (MAD of the high-frequency residual, cap 6).
  The added noise is blurred with the same sigma and rescaled, because the grain
  of the crops is "coarse" (they are upscaled).

### 8. Shadows and compositing
- Feathered edge with radius ≈ 8% of the varroa.
- Contact shadow: ring ≈ 10% of the varroa, opacity 0.25.
- Drop shadow: offset ≈ 8%, blur ≈ 20%, opacity 0.20.
- Direct alpha compositing onto the crop (no Poisson blending).

### 9. GT mask
OR of the tight binary masks of each varroa, 512×512, values 0/255. It is the
sharp silhouette of the varroa: it includes neither the edge feathering nor the shadows, which
do however modify the bee pixels within a few pixels of the contour.

### Number of varroa
- `num_varroa=None`: 1 or 2 at random, keeping those that fit.
- `num_varroa=N`: exactly N, with up to 5 attempts per varroa (new size and
  rotation at every attempt); if it does not succeed, the insertion fails.

## Dataset construction (`build_varroa_split.py`)

1. Random split of the crops: 80% train, 20% test.
2. In the test set 3% of the bees receive varroa: 2/3 with one, 1/3 with two.
3. The test set is scanned in random order until the quotas are filled. If the varroa
   does not fit on a bee, that bee stays normal and the next one is tried.
4. Normal bees are identical copies of the source crops (or symlinks with `--mode symlink`).

Counts on the 54.182 crops: train 43.346, test 10.836, of which 325 with varroa
(217 with one, 108 with two).

Output (default `.../processed/single_bee_batch3+4_varroa_split/`):

```
train/normal/        normal train bees
test/normal/         normal test bees
test/anomalous/      test bees with varroa
test/masks/          <bee_id>_mask.png, 0/255
test/labels.csv      bee_id, label (0/1), num_varroa, image_path, mask_path
split.csv            file, split, label, num_varroa
config.json          parameters and counts
```

Reproducibility: seed 42. Each bee has its own seed derived from the file name,
so the result does not depend on the processing order.

Commands:

```
venv/bin/python build_varroa_split.py --dry-run          # counts only
venv/bin/python build_varroa_split.py                    # full dataset
venv/bin/python build_varroa_split.py --crops-dir <dir> --output <dir> --limit 2000
venv/bin/python varroa_on_crops.py --name <prova> --num-bees 24   # visual check
```

The script does not write into a non-empty output folder.

## Main parameters

At the top of `varroa_on_crops.py`: `VARROA_BEE_RATIO_MIN/MAX` (0.20–0.24),
`BODY_CORE_RATIO` (0.60), `BODY_END_MARGIN` (0.22), `BODY_AXIS_BAND` (0.70),
`WING_MAX_SATURATION` (75), `WING_MIN_VALUE` (125), `MIN_ZONE_COVERAGE` (1.0),
`MAX_BLUR_RATIO` (0.12), `ATTEMPTS_PER_VARROA` (5), `CROPS_DIR`.

At the top of `build_varroa_split.py`: `TEST_FRACTION` (0.20), `ANOMALOUS_FRACTION`
(0.03), `TWO_VARROA_FRACTION` (1/3), `SEED` (42), `OUTPUT_DIR`.

In `varroa_generator.py` (shared with the old method): `BLEND_LUMINANCE`,
`BLEND_CONTRAST`, `BLEND_CHROMA`, `BLEND_SHADING`, `MIN_BLUR_SIGMA`,
`MAX_NOISE_SIGMA`, `CONTACT_SHADOW_OPACITY`, `DROP_SHADOW_OPACITY`.

## Known limitations

- **Brown wings folded over the abdomen** are not recognised as wings: there the
  varroa can end up on top of the wing.
- **Non-constant size**: the bee width includes the wings, so on
  bees with open wings the varroa comes out larger.
- **Thin light rim** around the varroa in some cases, more visible on dark areas.
- **Tip of the abdomen excluded** as a side effect of excluding the eyes.
- **Only two source varroa**: the variety of appearance comes from rotation, scale and
  colour adaptation.
- **Realism assessed only by eye** on a few dozen bees. The effect on the
  anomaly detection model has not been verified, nor has a comparison been made with
  the previous method.

## Status

- Pipeline tested end-to-end on 3.000 crops (counts, disjoint groups, binary
  masks with the right number of varroa, normal bees identical to the sources,
  reproducibility).
- At the time of writing the full dataset **has not been generated yet**.

## Update 17 October 2026: AD dataset from the phase 2 clusters

`build_varroa_split.py` now accepts multiple source folders (`--crops-dir A B ...`),
the fraction of anomalous bees from the command line (`--anomalous-fraction`) and `--mode hardlink`.
The source folder of each bee goes into the `cluster` column of
`test/labels.csv` and `split.csv` and into the per-cluster counts of `config.json`. After
writing, the script re-checks the number of files in each folder (the NTFS disk
can lose them without errors).

Dataset generated from the clusters Giovane (young), Adulte (adult), Malata (sick), Fuco (drone) of the export
`single_bee3+4+5+6_SAM3_clusters/fase2_finale` (Non_Classificabile excluded):
127.879 bees, train 102.303, test 25.576, of which 1.279 with varroa (5%: 853 with one,
426 with two). Output in `processed/single_bee3+4+5+6_SAM3_varroa_split/`, normal bees
as hardlinks of the export files.

```
venv/bin/python build_varroa_split.py \
  --crops-dir <fase2_finale>/Giovane <fase2_finale>/Adulte <fase2_finale>/Malata <fase2_finale>/Fuco \
  --anomalous-fraction 0.05 --mode hardlink \
  --output .../processed/single_bee3+4+5+6_SAM3_varroa_split
```
