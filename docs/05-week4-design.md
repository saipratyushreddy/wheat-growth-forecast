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
