# Week 4: soybean replication, future-photo retrieval, write-up

[Back to the plan](01-plan.md) · [Week 3 design and results](04-week3-design.md)

Written and committed before any week-4 code is run. Anything changed after results are seen is marked as a change,
with the reason. All earlier input rules still hold.

## Questions

1. **Does the "within-field" correction hold on independent data?** Week 3 part A suggested it, post hoc, from wheat
   errors (docs/04). Here it is tested once, on soybean (FIP1SOY), with everything fixed in advance.
2. **Does a better trait forecast give a better picture of the future plot?** This is the plan's future-photo demo,
   made testable: retrieve a past photo matching the forecast, and compare it with the photo actually taken later.
3. **Write-up:** one report covering weeks 1-4, with figures made from saved results only.

## Part A: soybean replication

### Data check (2026-11-06, before this design; no model fit)

FIP1SOY (`mikeboss/FIP1SOY`), pinned at revision `47c1f16a973cc1d12a5164b4cf7f20d337fe2e53`: three Arrow files,
about 1 GB.
- **Rows:** 1,536 rows, but each plot is listed once per weather station ("Eschikon-FIP" and "Lindau", identical
  values). After keeping one row per plot and season: 750 plot-seasons, 75 varieties, 8 seasons with 72-132 plots each.
  Four rows have no season and no data.
- **Season label:** the table's `harvest_year` is the sowing year + 1. Sowing is in April and harvest in September of
  the same year. Here a season is named by its **sowing year**: 2015-2022.
- **Traits:** canopy cover in all 8 seasons (11-39 dates per plot). Values up to 1.19 in three seasons (under 1% of
  values above 1). Height in only 2 seasons. Yield in 7 seasons. No heading, no markers, no official splits.
- **Weather is attached to the wrong dates.** Each season's weather runs from 1 September of the sowing year + 1 to
  31 August of the year after. So it never overlaps that season's growing period. The 2022 season has no weather.
- **The fix:** the soybean and wheat weather come from the same site station. On the 1,820 days both cover, daily mean
  temperatures correlate at 0.999 (largest difference 4.0 °C). Together, by calendar date, they cover 1 September
  2015 to 31 December 2022 without gaps.
- **Photos:** listed for 2015-2017 only, as raw camera files stored outside Hugging Face. Not used.

### Decisions

- **Seasons:** 2016-2022 (7). 2015 is dropped because no weather covers its growing period.
- **Weather:** one site calendar by date. Daily mean temperature from the soybean tables where they cover a date, and
  from the wheat tables otherwise. Weather is matched to a season by calendar date, never by the table's year label.
- **Trait:** canopy cover only, horizons 7, 14 and 28 days. Height has too few seasons.
- **Values above 1 are kept** as measured. Forecasts are clipped to [0, 1] as for wheat.
- **Samples:** the wheat rule (issue at each measurement after the first; target the measurement nearest issue +
  horizon, within 3 days).
- **Degree days:** our definition (base 0 °C, from sowing). Future degree days = degree days so far + the training
  seasons' average temperature by day of year (counted from 1 January of the sowing year).

### Models

All settings are frozen from wheat. Nothing is tuned on soybean.

| Id | Model |
|---|---|
| N3 | the training seasons' average degree-day curve, shifted to the plot's current level |
| G | gradient-boosted trees on the residual from N3, wheat's canopy-cover settings |
| S | the GRU on the residual from N3, wheat's canopy-cover epochs (25), 5 seeds |
| E | mean of G and S |
| G-w, E-w | the within-field versions (below) |

**Inputs:** wheat's inputs, minus what soybean lacks.
- No height, so no other-trait inputs.
- No markers, heading or final height. The genetics inputs are:
  - the variety's mean residual in other seasons' training plots (as for wheat);
  - the variety's yield deviation from its season's mean, averaged over other seasons' training plots. This replaces
    the final-height and heading deviations, in G and in S's starting state.
- No photo inputs.

**The within-field correction, as stated in docs/04:**
- On each issue date, take the model's correction to N3 (E - N3) for every plot of the field with a sample for that
  horizon.
- Subtract its mean over those plots: E-w = E - mean(E - N3).
- This uses no target values and no data after the issue date. The same is done for G (G-w).

### Hypothesis and rule

**H1:** E-w beats N3 in a new season.
- It "holds at a horizon" when the paired MAE difference (E-w minus N3), pooled over the 7 held-out seasons, has its
  95% interval below 0 **and** is lower in at least 5 of 7 seasons. This mirrors week 2's 4-of-6 rule.
- H1 "replicates" if it holds at all three horizons, and "partly" if at one or two.

**Also reported, without a rule:**
- E vs N3 and G vs N3, to see whether week 2's result carries over;
- E-w vs E;
- G-w vs N3.

**Expectation, stated before results:** soybean has fewer seasons (6 training seasons per fit) and fewer plots per
season than wheat. The learned corrections may be weak, so E-w may be close to N3.

**Protocol:**
- leave-one-season-out over 2016-2022;
- intervals: 95%, resampling varieties, 2,000 resamples;
- run once.

## Part B: future-photo retrieval (wheat)

The idea: show what a plot will look like in 14 or 28 days by retrieving a real photo from the training years that
matches the forecast. It is tested against the photo actually taken later.

### Queries

- Leave-one-year-out, reusing week-3 predictions (`outputs/week3/units/<year>_none_near.parquet`). Nothing is refit.
- Canopy-cover samples at 14 and 28 days whose plot has a photo within 3 days of the target date (the "true future
  photo").
- One query per plot and horizon: one eligible issue date drawn at random (seed 0).

### Candidates

Photos of plots from the fold's training years (the other five years), each paired with that plot's canopy-cover
measurement within 1 day of the photo. Only candidates within 7 days after sowing of the query's target day count.

### Methods (top-1)

| Id | The retrieved photo |
|---|---|
| R-now | the plot's own latest photo on or before the issue date (no forecast: "it will look like today") |
| R-var | the same variety in a training year, nearest in days after sowing. Falls back to R-N3 if none; the fallback rate is reported |
| R-N3 | the candidate whose measured cover is nearest N3's forecast |
| R-E+P | the candidate whose measured cover is nearest E+P's forecast (week 3's best canopy-cover model) |
| R-oracle | the candidate whose measured cover is nearest the true future cover. A ceiling only: it uses the target |

Ties go to the candidate nearest in days after sowing, then by photo path.

### Metric and rule

- **Distance:** 1 - cosine similarity between the retrieved photo and the true future photo, on DINOv2 features
  standardised with the training years' photos. Lower is better.
- **Comparisons:** paired, pooled over six years, with variety-resampled 95% intervals; per-year counts.
  - **R-E+P vs R-now:** does forecasting beat "it will look like today"?
  - **R-E+P vs R-N3:** does the better forecast give a better picture?
- **Rule:** "better" when the pooled interval is below 0 **and** it is lower in at least 4 of 6 years.
- **Expectation:** R-E+P beats R-now at 28 days. Against R-N3 the difference is small, because their errors differ by only
  0.4-1.4 points of cover on average (docs/04).

### Figure

- Eight queries drawn at random (seed 0) from 2019 at 28 days, not chosen by hand.
- For each: today's photo, the R-E+P photo and the true future photo (thumbnails), with forecast and true cover.

## Part C: write-up

- `docs/06-report.md`: question, data, evaluation design, results of weeks 1-4 (headline: leave-one-year-out),
  negative results, limitations, future directions.
- Figures from saved CSVs only (`src/figures.py`, writing `docs/figures/`). No figure needs a new fit.
- Future directions to cover:
  - G2F maize;
  - the late-season yield photo signal (docs/04), to be tested on new varieties;
  - forecasting the field-wide shift from weather forecasts;
  - honest end-of-season ranges for a new year.

## Leakage checks added to `tests/test_leakage.py`

1. The soybean site weather is matched by calendar date, and a season's inputs use no weather after the issue date.
2. Soybean variety features come only from other seasons' training plots.
3. The within-field mean uses no target values: changing targets does not change E-w.
4. A retrieval candidate never comes from the query's own year. R-now uses no photo after the issue date.

## Practical

- **Files:**
  - `src/soy.py`: loading, de-duplication, site weather, series, samples;
  - `src/run_week4.py`: soybean leave-one-season-out, retrieval;
  - `src/figures.py`;
  - `scripts/swan/w4_*.slurm`.
- **Where:** everything runs on Swan. The soybean data is downloaded there at the pinned revision; the photo features
  are already there.
- **Order:**
  1. tests;
  2. soybean (once);
  3. retrieval (once);
  4. figures;
  5. report.

## Clarifications made while implementing, before any week-4 result

1. **One day without temperature.** 6 May 2020 has no hourly temperature in the soybean tables, and the wheat tables
   have no 2020 season. Gaps of up to 2 days are filled by interpolating between the neighbouring days; this is the
   only one.
2. **Within-field forecasts are clipped to [0, 1],** like every other canopy-cover forecast. The field-day groups are
   held-out season × horizon × issue day; all plots of a season share the sowing date.
3. **R-var's window:** it uses the same candidates as the other methods (training years, within 7 days after sowing of
   the target day).
4. **Shared code.** Wheat's measurement-history inputs were moved into `near2.history_features`, used by wheat and
   soybean. On 500 wheat plots, wheat's features are identical to before the change. The season-day count and the
   sequence model's genetics inputs were made settable; the wheat defaults are unchanged.
5. **Code checks with stand-in data, not results.**
   - Soybean: each season's canopy-cover curves were shuffled between its plots, and S trained for 2 epochs. The
     pipeline ran with no missing predictions.
   - Retrieval: random stand-in photo features on the real photo list and week-3 predictions. 8,113 queries, no
     missing values, no candidate from a query's own year, no "today" photo after the issue date. 28% of R-var queries
     fall back to R-N3 (no photo of the same variety in the window).
6. **The soybean tables built on Swan are checked against the laptop's** (content hashes in
   `scripts/swan/derived_soy_hashes.json`), as for wheat.

## Results (2026-11-06)

Run once on Swan (jobs 18846003-18846007): the soybean tables matched the laptop's and all 26 tests passed. Each
part was then scored once. Outputs are in `outputs/week4/`: `soy_near.csv`, `soy_pred.parquet`, `retrieval.csv`,
`retrieval_queries.parquet` and `retrieval_examples.png`.

### Part A: soybean (leave-one-season-out, 7 seasons, canopy cover)

- **Data:** 662 plot-seasons and 75 varieties, giving 29,128 samples. The weather comes from the soybean tables on
  2,307 days and from the wheat tables on 250 days. One day (6 May 2020) was interpolated.
- **H1: partly.** It holds at 28 days only.

MAE is in points of cover. d is the paired difference against the reference; its interval is 95%, resampling
varieties. "Seasons" counts the seasons where d < 0, out of 7.

| h | Comparison | MAE | Ref. MAE | d | 95% interval | Seasons | Rule |
|---|---|---|---|---|---|---|---|
| 7 | **E-w vs N3** | 9.28 | 9.30 | -0.02 | [-0.29, 0.26] | 6 | no |
| 7 | E vs N3 | 9.97 | 9.30 | +0.66 | [0.28, 1.05] | 4 | |
| 7 | G vs N3 | 10.23 | 9.30 | +0.93 | [0.59, 1.27] | 3 | |
| 7 | G-w vs N3 | 9.28 | 9.30 | -0.02 | [-0.26, 0.24] | 6 | |
| 7 | E-w vs E | 9.28 | 9.97 | -0.69 | [-1.17, -0.19] | 4 | |
| 14 | **E-w vs N3** | 13.07 | 13.32 | -0.26 | [-0.54, 0.06] | 6 | no |
| 14 | E vs N3 | 12.47 | 13.32 | -0.85 | [-1.24, -0.45] | 6 | |
| 14 | G vs N3 | 12.64 | 13.32 | -0.68 | [-1.08, -0.30] | 6 | |
| 14 | G-w vs N3 | 13.09 | 13.32 | -0.24 | [-0.49, 0.05] | 5 | |
| 14 | E-w vs E | 13.07 | 12.47 | +0.59 | [0.07, 1.11] | 1 | |
| 28 | **E-w vs N3** | 16.57 | 16.94 | -0.37 | [-0.59, -0.14] | 6 | **yes** |
| 28 | E vs N3 | 14.17 | 16.94 | -2.77 | [-3.21, -2.36] | 6 | |
| 28 | G vs N3 | 14.85 | 16.94 | -2.09 | [-2.59, -1.65] | 5 | |
| 28 | G-w vs N3 | 16.56 | 16.94 | -0.38 | [-0.60, -0.16] | 7 | |
| 28 | E-w vs E | 16.57 | 14.17 | +2.40 | [1.93, 2.88] | 2 | |

The individual models' MAE (S alone: 10.52, 13.33 and 14.82 at 7, 14 and 28 days) is in `soy_pred.parquet`.

**What happened.** On soybean, unlike wheat, most of the learned models' gain over N3 is the shared part, the
field-wide shift.
- **The plain models beat N3 at 14 and 28 days, in 6 of 7 seasons.** E's gain at 28 days is -2.8 points. This is much
  larger than on wheat. A likely reason: N3's average curve fits soybean less well, because canopy closure depends on
  more than degree days, so there is more shared error to correct.
- **The within-field correction removes most of that gain.** At 14 and 28 days, E-w is worse than E.
- **What it buys is safety.** E's and G's field-wide shift is sometimes badly wrong:
  - In 2017, E moved the whole field about 10 points down, while the truth was 1-10 points above N3. E lost to N3 by
    +6.1 points at 7 days and +4.5 at 14.
  - In 2020 at 28 days, the shift had the wrong sign.
  - E-w is never more than 0.5 points worse than N3 in any season and horizon. E is up to 6.1 worse.
- **So the hypothesis is half right.** The learned field-wide shift is unreliable from one season to the next: wheat
  2016/2017 (docs/04), soybean 2017. But on soybean it is right more often than wrong, so removing it costs accuracy on
  average.

The within-field correction is a safer model, not a better one. The wheat claim "the learned models lose on field-wide
error" did not carry over as a general rule.

The expectation stated before results ("E-w may be close to N3") was right for E-w. It was wrong for E, which did
better than expected.

### Part B: future-photo retrieval (wheat, leave-one-year-out)

- **Queries:** 4,057 at 14 days and 4,056 at 28 days, about 710-790 per year and 250 in 2019, the year with fewest
  photos.
- **R-var** fell back to R-N3 for 28% of queries, as in the stand-in check.

Distance is 1 - cosine on standardised DINOv2 features; lower is better. Intervals are 95%, resampling varieties.
"Years" counts the years where d < 0, out of 6.

| h | Comparison | Distance | Ref. distance | d | 95% interval | Years | Rule |
|---|---|---|---|---|---|---|---|
| 14 | **R-E+P vs R-now** | 0.565 | 0.429 | +0.135 | [0.127, 0.142] | 0 | no |
| 14 | **R-E+P vs R-N3** | 0.565 | 0.570 | -0.006 | [-0.012, 0.001] | 5 | no |
| 14 | R-N3 vs R-now | 0.570 | 0.429 | +0.141 | [0.133, 0.148] | 0 | |
| 14 | R-var vs R-now | 0.591 | 0.429 | +0.161 | [0.152, 0.170] | 0 | |
| 14 | R-oracle vs R-now | 0.555 | 0.429 | +0.125 | [0.118, 0.133] | 0 | |
| 28 | **R-E+P vs R-now** | 0.582 | 0.627 | -0.044 | [-0.054, -0.035] | 5 | **yes** |
| 28 | **R-E+P vs R-N3** | 0.582 | 0.585 | -0.003 | [-0.009, 0.004] | 4 | no |
| 28 | R-N3 vs R-now | 0.585 | 0.627 | -0.042 | [-0.051, -0.033] | 5 | |
| 28 | R-var vs R-now | 0.594 | 0.627 | -0.033 | [-0.044, -0.022] | 5 | |
| 28 | R-oracle vs R-now | 0.561 | 0.627 | -0.066 | [-0.075, -0.056] | 5 | |

**What this means.**
- **At 14 days, today's photo of the same plot is the best picture of the future, by a wide margin.** Even the
  ceiling, which matches the true future cover, is far behind it (0.555 vs 0.429). Over two weeks a plot's own look
  dominates the photo features: its soil, stones, row pattern and lighting change less than its cover does.
  Cover-matched photos from other plots cannot supply that.
- **At 28 days, growth dominates and retrieving by forecast wins.** It wins in 5 of 6 years; the exception is 2016, where
  today's photo is still the best at 28 days.
- **A better forecast does not give a visibly better picture.** R-E+P and R-N3 differ by 0.003-0.006, with intervals
  that include 0. This is as expected: their cover errors differ by about 1 point (7.3 vs 7.9 at 14 days, 9.5 vs 10.6
  at 28 days, on these queries).
- **The headroom from a perfect cover forecast is small.** R-oracle is only 0.02 better than R-E+P. Cover alone
  describes little of what a plot will look like.
- **Matching on variety is worse than matching on forecast cover** at both horizons.

**Figure:** `outputs/week4/retrieval_examples.png`. It shows eight random 2019 queries at 28 days: today's photo,
the R-E+P photo and the true photo. The retrieved photos match the stage of growth. They do not match the plot's own
soil and lighting.

### What week 4 adds

- **The within-field correction replicates only partly on soybean.** It is a safer model, but not a more accurate
  one on average.
- **Soybean's plain learned models beat N3 at 14-28 days,** much more clearly than wheat's did.
- **Forecast-based photo retrieval** is useful only at 28 days, and only because "it will look like today" breaks
  down by then. A better trait forecast does not translate into a better picture.
