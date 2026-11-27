# bee_al — from clusters to colony estimates

`bee_al` is the active-learning hub for crops of individual bees: frozen DINOv2 → memory bank →
weighted k-NN + label propagation → query batches → labelling in the hub → final clusters.
This document describes the workflow that takes the clusters to reliable estimates of
colony composition.

```
labelling ──► stopping criteria ──► snapshot ──► blind audit ──► corrected estimates ──► colony indicators
```

Why it is needed: counting the bees the model assigns to each class ("classify and count") gives
biased numbers, especially for rare classes. If 5% of healthy bees are mistaken for diseased ones
and the truly diseased bees are 2%, the raw count says 5.6%. The audit measures the model's errors
on a sample it has never seen and uses them to correct the count.

## Phased pipeline

Labelling is done in several phases, each with its own config, labels and models:

1. **Phase 1 – segmentation quality.** All bees are split into well-segmented and poorly segmented
   ones (`labels.classes` suggests `Segmentata Bene` and `Segmentata Male`).
2. **Phase 2 – demographic profile and anomaly detection.** Only the bees kept by phase 1 are
   used. It is prepared from a phase 1 snapshot:

```
python scripts/new_stage.py --snapshot fase1_finale --keep "Segmentata Bene" --name fase2
cd active_learning_tool && BEE_AL_CONFIG=config/al_fase2.yaml ../venv/bin/python -u app.py
```

The script writes the list of kept bees (`results_fase2/subset.json`) and `config/al_fase2.yaml`, with
`paths.subset_file` pointing to the list. Phase 1 is left untouched: to go back to it, just start the hub
without `BEE_AL_CONFIG`. `--min-confidence` leaves out the bees the model assigned with low
confidence.

## Grid labelling

`/al-grid` shows 12 bees at a time (4 × 3, on the left) and the controls on the right. Each colour is a
group with its own class: pick a colour, click the bees of that class, switch to another
colour and save everything together. Keys 1–6 pick the colour, 0 picks "Scarta ape" (discard bee). Bees
left without a colour come back in the next grid. `/al-grid?mode=validation` does the same on the
validation set.

The **Shuffle: propose other bees** button (grid and single page, not in validation or audit mode)
replaces the not-yet-labelled bees of the batch with others, chosen by the same query strategy and
the same model: no new round starts and labels already given are kept. Removed bees are not
proposed again until the next round; those another annotator is looking at stay theirs. Use it
sparingly: deliberately skipping hard bees deprives the model of exactly the most useful examples.

## Cluster inspection

`/al-clusters` shows **one group at a time**, chosen from a drop-down menu: the clusters assigned by the
model, the "unassigned" bees and the hand-labelled classes. Bees are as large as in the labelling
grid, twelve at a time on three rows:

| Row | Bees |
|---|---|
| 1 | the most similar to the group centre |
| 2 | those around the median |
| 3 | the least similar |

Similarity is the cosine between the bee's embedding and the centre (the mean) of the group; for the
"unassigned" bees it is the similarity to the nearest labelled example. *Other bees* moves to the next
twelve: the following ones from the top, from the bottom and around the median, never repeating a bee.
Corrections are made with the colours, as in the grid; the hand-given labels also include *Remove label*.

Two menus choose how many bees to show (12, 24, 48 or 96) and in which order: the three bands (default),
from most or least similar, from most or least confident for the model (predicted clusters only) or a
random sample. The choice persists when switching group.

With several batches, the **Batch** menu shows only the bees of one batch (the session from
`frames.session_dirs`, derived from the source frame; without a frame map it is the dataset folder). The
counts in the group menu and the group centre follow the filter.

## Batch held out for validation

A new batch can be kept out of training to measure the model on never-seen photos:

```
python scripts/add_dataset_to_stage.py --snapshot fase1_finale_okvsscarta --stage-config config/al_fase2_sam3.yaml \
    --name b7 --dir /percorso/single_bee7_SAM3/cleaned_bees --keep Ok      # phase 1 filter
python scripts/extend_stage_embeddings.py --config config/al_fase2_sam3.yaml --name b7   # embeddings from the current model
```

and in the config `validation.holdout_datasets: [b7]`. The bees in that folder are predicted (they are in
the propagation graph as unlabelled bees) but are never proposed, never enter re-training, not even
as unlabelled bees, and do not move the "unassigned" threshold. Corrections made to them on the
clusters page become `holdout` events, which train nothing; in the grid they are marked
*held out*. The thesis measurements come from the blind audit restricted to the batch
(`audit.datasets: [b7]`, section 3).

The hub home page lists only the active-learning pages; the older tools are grouped
under the *Old tools* button.

## 1. Labelling and stopping criteria

Label as usual from `/al-labeling`. For bees you are unsure about, it is worth declaring
*Low* confidence: they are the first to end up in the review of suspicious labels (see below). The
`/al-status` page shows three stopping criteria, evaluated over the last `stopping.window` rounds that
received new labels:

| Criterion | Meaning |
|---|---|
| `f1_plateau` | the validation macro-F1 does not improve by more than `f1_min_delta` |
| `predictions_stable` | less than `max_pred_change` of the bees change class from one round to the next |
| `no_new_classes` | no new class for at least `window` rounds |

They are guidance only: nothing stops on its own. A hub restart or a recomputation without
new labels creates a round but does not count. After a re-train the predictions are not
comparable with the previous round, and that round is excluded from the stability criterion.
Without labels in the validation set, the first criterion stays "not yet known".

```yaml
stopping:
  window: 3
  max_pred_change: 0.01
  f1_min_delta: 0.01
```

## 2. Freezing a snapshot

A snapshot freezes the predictions for every bee. It never changes again: the audit and the estimates
are computed on it, even if labelling continues in the meantime.

- From the hub: `/al-status` → **Freeze snapshot**.
- From the command line, without starting the hub and without adding rounds:

```
python scripts/export_snapshot.py --name 2026-10-05_round42
```

In `<results_dir>/snapshots/<name>/`:

| File | Contents |
|---|---|
| `predictions.csv` | one row per bee: status, final class, predictions (propagation and k-NN), frame, session, hive |
| `frames.csv` | one row per frame: raw bees extracted, accepted and rejected by the cleaning step, rejection reasons |
| `meta.json` | round, model, full config, counts, validation metrics, stopping criteria, reliability of the frame maps |
| `al_labels.csv`, `validation_set.json` | copies of the state at freeze time |

Status of a bee (`status`): `labeled` (human label), `predicted` (assigned by the model),
`rejected` (final class is a reject class), `unassigned` (too far from every example), `unknown`
("Scarta ape"), `validation`, `no_prediction`.

### Which frame each bee comes from

The link is derived from the extraction manifest (`raw_bees/_extraction_manifest.json`):
bee ids increase by 1 frame after frame, so the per-frame counts give the
id ranges. If the sum of the counts does not match `next_bee_id`, the map for that
folder is declared unreliable and not used: no guessing.

Session and hive are derived from the frame name with a regex. Without a regex, or if it does not
match, the session is the name of the dataset folder and the hive is left empty.

```yaml
frames:
  manifests: {}        # {folder name: manifest path}; empty = <cleaned folder>/../raw_bees/
  frame_pattern: ""    # e.g. "(?P<colony>hive\\d+)_(?P<session>\\d{8})"
```

## 3. Creating and labelling the audit

From `/al-status`, in the snapshot row: **Create audit set** (only once per snapshot,
like the validation set). The sample is drawn from the bees assigned by the model and never
hand-labelled, split into strata:

- predicted class × confidence band (`confidence_bins`);
- one stratum for each predicted reject class;
- one stratum for the unassigned bees;
- if a CSV `image_id,score` is loaded, the strata of `anomaly_split_class` are also split
  by anomaly score.

From each stratum `per_stratum` bees are drawn (all of them, if the stratum has fewer; `large_stratum`
if it has at least `large_threshold`). A `double_fraction` share must be labelled by two
different annotators, to measure agreement.

The audit is labelled from the **Audit** menu (`/al-labeling?mode=audit`), blind: no
suggestions, no predicted class, no stratum. The bees are shuffled.

Rules guaranteed by the code:

- audit labels never enter the memory bank, the k-NN, the propagation,
  re-training or the validation metrics;
- bees in the active audit are no longer proposed in normal mode and cannot be
  modified from the clusters page;
- the same annotator never gets the same bee twice.

```yaml
audit:
  per_stratum: 40
  large_stratum: 250
  large_threshold: 5000
  confidence_bins: [0.9]
  double_fraction: 0.1
  anomaly_split_class: ""
  anomaly_quantile: 0.9
  seed: 7
```

## 4. Report and indicators

- From the hub: `/al-status` → **results**, or `/al-audit-results?snapshot=NOME`.
- From the command line:

```
python scripts/audit_report.py --snapshot 2026-10-05_round42
python scripts/audit_report.py --snapshot NOME --anomaly-scores scores.csv
```

In `<snapshot>/report/`: `strata.csv`, `prevalence.csv`, `groups.csv`,
`colony_indicators.json`, `colony_indicators.csv`, `agreement.json`, `report.md`.
The result is cached until new audit answers arrive.

Without an audit, only the model's per-batch counts (shares with a bootstrap over photos, bees per photo):

```
python scripts/batch_counts.py --snapshot NOME --config config/al_fase2_sam3.yaml
```

In `<snapshot>/batch_counts/`: `batch_counts.csv`, `photo_counts.csv`, `batch_counts.png`.

### Estimates

For stratum `h`: `N_h` bees in the population, `n_h` audit answers ("Scarta ape" answers are
counted separately), `r_hk` share of answers of class `k`. `H_k` are the bees hand-labelled
as `k`: they count exactly and are not corrected.

| Quantity | Formula |
|---|---|
| Precision of a stratum | `r_h,c` for the predicted class `c`, with Wilson interval |
| Estimated total | `Ñ_k = Σ_h N_h · r_hk + H_k` |
| Prevalence | `p_k = Ñ_k / Σ_{j not a reject class} Ñ_j` |
| Variance | `Var(Ñ_k) = Σ_h N_h² (1 − n_h/N_h) r_hk (1 − r_hk) / (n_h − 1)` |
| Class never seen in a stratum | upper bound `≈ 3 / n_h` (rule of three) |
| Per group `g` | `Ñ_g,k = Σ_h N_g,h · r_hk + H_g,k` |

The main intervals come from a bootstrap: in each replicate the audit answers are resampled
within each stratum and, for groups, the frames of the group.

Things to keep in mind when reading the numbers:

- **Per-group estimates assume that the model's errors within a stratum are the same in
  all groups.** If a hive has "unusual" bees that the model gets wrong more often, that
  group's estimate does not see it.
- If a stratum has no audit answers yet, the predicted class is used as-is for that
  stratum, and the report flags it.
- The bootstrap resamples from slightly smoothed frequencies (one pseudo-observation per
  stratum). Without it, a rare class never seen in a large stratum would have zero uncertainty and
  the interval would cover the true value about 77% of the time instead of 95%.
- For double-labelled bees the first annotator's answer counts; the second is used only for kappa.
- With few frames per group (two or three), frame resampling underestimates the variability:
  per-group intervals and "significant" trends should be taken with caution.
- Validation bees, "Scarta ape" bees and bees without a prediction are left out of the estimates.

### Colony indicators

Computed per group (`colony.group_by`: `frame`, `session`, `colony` or `dataset`).

| Indicator | Contents |
|---|---|
| I1 – demographic profile | corrected shares, with interval, of the `demographic_classes` |
| I2 – disease signs | for each class in `disease_classes`: `detected`, `uncertain` or `not_detected` (with upper bound) |
| I3 – Jensen–Shannon divergence | distance of the group profile from the reference profile, in [0, 1] |
| I4 – unexplained share | (unassigned + "Scarta ape") / bees in the group |
| I5 – trend | difference between consecutive sessions of I1, I2, I3, with interval |
| Colony strength | raw bees per frame |
| Quality | estimated reject share and rejections by the automatic cleaning |
| Traffic light | green, yellow or red, with the list of rules triggered |

A disease sign is `detected` when the lower bound of the interval is above zero and
at least `min_detected` bees of the group have been confirmed by a human (audit answers plus
hand-given labels). The I3 threshold is the mean + 2 standard deviations of the divergence of the
reference groups: with few groups it is not very informative.

```yaml
colony:
  group_by: session
  demographic_classes: []
  disease_classes: []
  min_detected: 3
  reference_groups: []        # empty = mean of all groups
  time_order: {}              # {hive: [session1, session2, ...]}; "*" = all hives together
  anomaly_threshold: 0.0      # 0 = off
  n_boot: 1000
  seed: 11
  light_rules: [disease_detected, profile_divergent]   # also: unexplained_high
  red_min_rules: 2
  max_unexplained: 0.0
```

## Review of suspicious labels

A wrong label weighs a lot: labels are few, uncertainty queries pick precisely the
ambiguous bees, and in the propagation human labels are clamped, so an error spreads to
similar bees without the graph correcting it. The **Review** page (`/al-review`) shows the labels that
are worth a second look. Nothing is corrected automatically.

**How they are found.** For each hand-labelled bee, the k-NN votes using the *other* labelled
bees (the bee does not vote for itself), with the same weights as the hub. The label is suspicious if
the most-voted class is a different one and collects at least `review.min_disagreement` of the vote
(default 80%). This is the idea behind *confident learning* (Northcutt et al., 2021). The queue is
recomputed whenever the predictions are recomputed; labels given with *Low* confidence come first,
then *Medium*, then the rest.

**Why the validation set and the audit are excluded.** They are used to measure the model. Correcting
them based on the model's opinion would pull them towards its predictions and make it look better than
it is. For them the only check allowed is inter-annotator agreement, already built into the audit. Bees
set aside with "Scarta ape" are also left out.

**The three decisions.** Next to the bee you see the current label, who gave it and with what
confidence, the suggested class and the most similar labelled bees, which are the evidence.

| Decision | Effect |
|---|---|
| Keep (*confirm*) | the label is correct: no new event; the bee does not come back into the queue as long as its label and the suggested class stay the same |
| Change (*correct*) | the correct class is a different one, the suggested one or another of your choice: it is a normal correction in `al_labels.csv`, and the predictions are recomputed |
| Scarta ape (*set aside*) | the bee cannot be judged: it is set aside |

Each decision is a row of `review_log.jsonl`, append-only; on restart the confirmations are
rebuilt from it.

**Rare classes.** A class with fewer than `review.min_class_size` labels (default 5) has neighbours
almost all from other classes even when the label is correct: this is exactly what a newly discovered
rare phenotype looks like. These bees are listed in a separate section at the bottom of the page: they
should be looked at, but they are expected to be mostly correct. The signal becomes useful once the
class has more examples.

**Numbers to report.** In `/al-status`, the *Label review* panel (and in `review_report.csv`, one row per
decision): labels checked, suspects found, how many were confirmed, corrected and set aside, the
corrections per class pair (old → new), and the share of suspects per declared confidence. If
*Low* labels are suspicious much more often than *High* ones, the declared confidence is informative.
Each row of `al_rounds.jsonl` also has `n_suspects`.

```yaml
review:
  enabled: true
  k: 0                    # 0 = knn.k
  min_disagreement: 0.8
  min_class_size: 5
  show_neighbors: 6
```

## Reject classes

`labels.reject_classes` lists the classes of crops that cannot be judged (blurred,
badly cropped, several bees, no bee). Rule for annotators: **if the defect prevents
judging the phenotype it is a reject; if the phenotype can still be recognised, label the phenotype.**

| Where | What happens to reject bees |
|---|---|
| k-NN and propagation | they stay, so the model learns to recognise them |
| Final clusters | excluded |
| Snapshot | status `rejected`, whether rejected by hand or by the model |
| Audit | one stratum per predicted reject class: measures how many valid bees are rejected by mistake |
| Estimates | estimated like the other classes, then excluded from the prevalence denominator |
| Report | estimated reject share per group, next to the rejections by the automatic cleaning |
| `/al-status` | count per class: hand-labelled and assigned by the model |

## Package files

| File | Role |
|---|---|
| `engine.py` | hub state: batches, rounds, stopping criteria, snapshots, audit |
| `frames.py` | bee → frame link from the manifests |
| `snapshot.py` | writing and reading snapshots |
| `audit.py` | stratified sampling and corrected estimates |
| `colony.py` | colony indicators |
| `report.py` | report files and cache |
| `review.py` | queue of suspicious labels |
| `export.py` | closing a phase: clusters on disk |
| `hub.py` | Flask pages |

Run the tests with `venv/bin/python -m pytest tests`.

## Current phase and closing a phase

The `stage` section of the config says which phase you are in (`name`, `title`, `description`): the hub
shows it in a blue band at the top of every page, so whoever is labelling knows what they have to judge.

When the phase is finished, the **Save configuration & export clusters** button in `/al-status` freezes a
snapshot and writes the clusters to disk in `export.clusters_dir/<name>/`:

| Contents | Description |
|---|---|
| one folder per class | the images of the bees of that class, hand-labelled or assigned by the model |
| `clusters.csv` | all bees: cluster, status, origin, confidence, frame, session, original path |
| `config.yaml` | the configuration the phase was run with |
| `meta.json` | phase, date, who exported, bees per cluster, source snapshot |

Discarded bees ("Scarta ape"), unassigned bees and validation bees stay only in the CSV.
`export.mode` chooses between copying the files (`copy`), `hardlink` and `symlink`. The export runs in
the background and does not stop labelling. Which clusters to carry into the next phase is decided
later, with `scripts/new_stage.py --keep ...`.

The hub works only on the folders listed in `paths.dataset_dirs`: removing a folder (for example
the test set) excludes its bees even if their embeddings have already been extracted.
