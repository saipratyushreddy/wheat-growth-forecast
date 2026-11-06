# Week 3: where the learned models lose, photo features, prediction ranges

[Back to the plan](01-plan.md) · [Week 2 design and results](03-week2-design.md)

Written and committed before any week-3 code is run. Anything changed after results are seen is marked as a change, with
the reason. All week-1 and week-2 input rules still hold. Photos are inputs only when dated on or before the issue date
or cutoff.

## Questions

1. **Where and why do the week-2 learned models lose to N3?** This is the plan's fallback, which applies in part. It is
   answered from week-2 predictions already saved: no new fit, no test reuse.
2. **Do photos add anything beyond the measurements?** Frozen DINOv2 features, added to G (near future) and D (end of
   season).
3. **Can the forecasts carry prediction ranges that hold in a new year?** Conformal ranges calibrated on held-out years.

## Part A: where the learned models lose

Descriptive analyses fixed now. They use `outputs/week2/loyo_near_pred.parquet`.

**A1. Where the loss is:**
- Per held-out year, E - N3 and G - N3 error by issue day after sowing (14-day bins) and by phase.
- Is 2016's loss spread out, or concentrated on a few dates?

**A2. Shared vs plot error:**
- Each forecast error is split into the mean error over all plots of that year, trait, horizon and issue date (the
  field-wide surprise) and the remainder (plot-specific).
- Question: do the learned models lose on the field-wide part, which no model can know without future weather, or on
  the plot-specific part?
- Reported for N3, G and E, per year.

**A3. Extrapolation:**
- For each held-out year: the share of its samples whose degree days so far, degree-day increment or anomaly fall
  outside the 1st-99th percentile range of the training years.
- Error inside vs outside that range.

These are explanations, not tests. Nothing in parts B and C changes because of them.

## Part B: photo features

### Getting the photos without storing them

There are 153,022 aligned inner-plot photos: 960×640 RGB PNG, about 165 GB, every plot-year with 9-63 photos.
- **Why not store them:** stored, they would use about 310,000 files of the Swan group's remaining ~740,000-file quota.
- **What the job does instead:** it streams them in shards of plot-years. For each shard it:
  1. downloads the photos from `mikeboss/FIP1` at the **pinned revision** `11cdd84bc0ba8d5c6641ea7b53550b1658527e1c`
     (the tables' own URLs point to `main`, so they are rewritten to the revision);
  2. checks each file's SHA-256 against the checksum the Hub publishes, and stops on a mismatch;
  3. computes the features and a thumbnail;
  4. deletes the PNGs.
- **Outputs:** per shard, one features table and one thumbnail table, merged per year at the end: a few dozen files and
  a few GB.
- **Resuming:** a shard whose outputs exist is skipped. Downloads retry with backoff when the Hub rate-limits.
- **Thumbnails:** 240×160 JPEG (quality 85). They are for visual checks and the week-4 demo, not model inputs.

**What streaming gives up:** a different crop, resolution or model would mean downloading again. Fine-tuning on the
photos is out of scope. So everything below is fixed now.

### Features

- **Model:** DINOv2 ViT-B/14 (`facebook/dinov2-base`, Hugging Face), frozen, in evaluation mode, on CPU. The weights
  revision is pinned in the code.
- **Preprocessing:**
  - the whole photo, resized to 420×280 (exactly 3:2, 30×20 patches), bicubic, no crop;
  - ImageNet mean and standard-deviation normalisation.
- **Per photo:** the CLS token (768) and the mean of the patch tokens (768), stored as float32.
- **Same-day photos** of a plot (1,872 cases) are averaged.

**From photos to model inputs.** PCA (32 components) is fit, per fit, on the training plots' photos only, after
standardising the 1,536 features. For each sample:
- the PCs of the latest photo dated on or before the issue date (or cutoff);
- that photo's age in days;
- the change in the PCs since the latest photo on or before 14 days earlier.

The number of components (32) is fixed, not tuned.

**Sanity check, not a result:** with the photos of held-out years, the PCs should predict the same day's canopy cover
well. If they do not, the features or their alignment are broken.

### Models

| Id | Model | Compared with |
|---|---|---|
| G+P | G with the photo inputs added, G's week-2 settings frozen | G |
| E+P | mean of G+P and S | E |
| D+P | D with the photo inputs at the cutoff (centred within year, as D's in-season features), D's settings frozen | D |

No new tuning. S is unchanged: adding photos to the sequence model is left out to keep the comparison clean.

**Expectation, stated before results:** FIP's canopy cover is measured from these same photos. So photos can add little
for canopy-cover forecasts. The real question is height, heading, final height and yield.

**Success rules**, the same form as week 2, on leave-one-year-out:

| Comparison | Photos "add" when |
|---|---|
| G+P vs G, E+P vs E | the paired MAE difference, pooled over six years, has its 95% interval below 0, **and** it is lower in at least 4 of 6 years |
| D+P vs D (ranking) | the paired within-year Spearman difference has its 95% interval above 0, pooled over years **and** in `test_genotype` |

## Part C: prediction ranges

**Forecasts wrapped:**
- near future: E;
- end of season: L0 + D for every target, L1 + D for heading.
- The models are those already chosen; photo models are wrapped only if they pass part B's rule.

**Method: year-held-out split conformal.**
1. For each fit (each leave-one-year-out fold, and the official fit), residuals on the training years come from an inner
   leave-one-year-out: refit on all but one training year, predict that year, repeat for each training year. These
   residuals include the shift of a new year, which in-season residuals would miss.
2. The range is the forecast ± q, where q is the empirical quantile of the absolute residuals at level ⌈(n+1)(1-α)⌉/n.
3. q is computed separately per group:
   - near future: trait × horizon × phase;
   - end of season: target × cutoff.
4. Levels: 80% and 90%.
5. Forecasts clipped to the physical range keep their clipped ranges.

**Two versions:**
- **Pooled (primary):** q from all inner residuals pooled.
- **Worst year (robustness):** q = the largest over the training years of that year's own quantile. It reflects that a
  new year is one draw from about five.

**Reported:** coverage and mean width per held-out year, per group and per phase; also for the official unseen-year
splits.

**Calibration rule**, on leave-one-year-out: a version is "calibrated" at a level when the mean coverage over the six
years is within 3 points of the level **and** no year is more than 10 points below it.

## Protocols and runs

- **Week-2 settings stay frozen.**
- **Order of runs:**
  1. the photo feature job (Swan, CPU array over shards);
  2. tests;
  3. the validation sanity check (official fit, scored on validation; bugs only);
  4. the official test, once;
  5. leave-one-year-out.
- **The inner refits for part C:**
  - five per fit (one per training year), each fitting every wrapped model;
  - for leave-one-year-out that is 6 × 5 = 30 refits, run as a SLURM array.
- **Intervals:** 95%, resampling varieties, 2,000 resamples, as before.
- **Not used**, as in weeks 1-2:
  - field position;
  - `*_blue`, `*_heritability`, `harvest_date`, `spatial_check`;
  - any heading information of the forecast season;
  - future weather;
  - photos after the issue date.

## Leakage checks added to `tests/test_leakage.py`

1. A sample's photo inputs come only from photos dated on or before its issue date or cutoff.
2. The photo PCA is unchanged when test plots' photos are removed.
3. A conformal q for a fold uses only residuals of plots in that fold's training years, from refits that excluded each
   residual's own year.
4. The feature job's output for a test image equals a recomputation from the same file, which guards against
   misordered batches.

## Practical

- **New dependencies:** `transformers` and `pillow`, pinned. CPU only, so the week-2 environment is reused.
- **Hugging Face token:** recommended. Unauthenticated downloads are rate-limited. The user creates a free token and
  runs `hf auth login` on Swan once. The job also works without one, more slowly.
- **Disk:** on Swan, under 10 GB at any time. Under 100 files per shard, deleted after merging.
- **Files:**
  - `src/photos.py`: streaming, checksums, features, thumbnails;
  - `src/photo_feats.py`: PCA, issue-date features;
  - `src/ranges.py`: conformal;
  - `src/analysis_w2.py`: part A;
  - `src/run_week3.py`: the runner;
  - `scripts/swan/*.slurm`.

## Clarifications made while implementing, before any week-3 fit

1. **Same-day photos are averaged after the PCA,** on their PCs, per plot and day after sowing.
2. **Week-3 fits drop what week 3 does not compare:** G+W and the week-1 end-of-season baselines F0-F3. N3 is still
   computed in every fit.
3. **Ranges are computed for every wrapped model, including the photo versions E+P and D+P.** The write-up reports
   photo-model ranges only if part B's rule is passed, as the design says.
4. **The "photos add" rule is applied to the pooled leave-one-year-out set.** Per-year and official-split rows report
   the difference only.
5. **The official fit produces predictions for validation and test plots together.** Scoring is gated:
   - test scoring refuses to run before validation scoring, or a second time;
   - leave-one-year-out scoring refuses to run before the test.
6. **Code check with stand-in features:** the full pipeline (units, ranges, comparisons) was run on the laptop with
   random stand-in photo features and tiny settings. It produced no gaps in any method. No number from it is a result.
7. **Duplicate photo listings (2026-11-06, found when the photo job failed, before any fit).**
   - The tables list 1,843 photo files twice or three times under the same plot: 1,872 extra rows, 153,022 rows for
     151,150 files.
   - In the job, two download threads fetched the same file and one deleted it before the other read it, so 21 of 48
     shards stopped with `FileNotFoundError`.
   - Each file is now kept once. The shard of every file is unchanged. Shards that finished before the fix hold the
     repeated rows, which are dropped at the merge.
   - The "1,872 same-day photos" above were these duplicates. There are no two different photos of a plot on the same
     day, so the same-day averaging never applies.

## Part A results (2026-11-05)

From `outputs/week3/partA*.csv`, built from week-2 leave-one-year-out predictions only.

**A2: the field-wide error dominates, and it is where the learned models lose.**
- For every model, most of the error is shared by all plots of a year on the same date. For example, canopy cover
  +28 d in 2016 for N3: field-wide 8.7 of 10.4 points.
- **Plot-specific error:** the learned models reduce it in almost every year, trait and horizon. Height +28 d: 2016
  5.4 to 4.3 cm (G); 2017 4.0 to 3.1 cm.
- **Field-wide error:** where they lose to N3, their field-wide error grew. Height +28 d, G vs N3: 2016 8.4 to 10.3 cm;
  2019 2.3 to 4.0 cm.
- So the models learn plot-level patterns that carry over to a new year. They also learn season-wide shifts from the
  training years, which misfire in a new year.

**A1: where 2016's loss happens.**
- Canopy cover +28 d: G and E beat N3 up to about day 200, then lose heavily from day 210 to 250 (+13 to +15 points at
  days 224-238), the ripening period.
- Height +28 d: they are worse from day 182 on.

**A3: extrapolation is not the main explanation.**
- 0-18% of a held-out year's samples fall outside the training years' range of degree days so far, degree-day
  increment or anomaly.
- The error on those samples is sometimes much higher (2016 canopy cover: E 21.1 vs 10.8 points) and sometimes lower.

**Exploratory, post hoc, not a result.**
- Prompted by A2: keep only the plot-specific part of the learned correction to N3, by subtracting, on each issue date,
  its mean over all plots of the field. That mean is known at forecast time and uses no target.
- On the same week-2 leave-one-year-out predictions:
  - this version of E beats N3 in 6 of 6 years for 5 of the 6 forecasts (5 of 6 years for canopy cover +14 d);
  - its average gain is smaller than E's for canopy cover at 14-28 days.
- It was found by looking at these errors, so evaluating it on them is circular. It is recorded as a hypothesis for an
  independent test (FIP1SOY, week 4), and is not added to the week-3 models.
