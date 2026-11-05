# Week 1: evaluation design and simple baselines

[Back to the plan](01-plan.md)

Written and committed before any baseline result exists. Anything changed after results are seen is marked as a change,
with the reason.

## Two forecasting tasks

**Near-future growth.**
- **Traits:** canopy cover (reported in percentage points) and height (reported in cm).
- **Issue dates:** every date on which the plot has a measurement of that trait, with at least one earlier measurement.
- **Horizons:** h = 7, 14, 28 days.
- **Target:** the plot's measurement nearest to issue + h, within ±3 days. If there is none, the sample is skipped.
  - The forecaster is told the target's actual date, as a planned measurement date would be known in advance.
  - Results are reported by nominal horizon.
- **Season phase:** results are also reported by the issue day after sowing.
  - Up to 150: autumn and winter.
  - 151-200: spring growth.
  - Over 200: heading to ripening.
- **Measured noise floor:** the median change between two measurements of the same plot one day apart (canopy cover 3.4
  points, height 0.7 cm). No forecast can be reliably better than this, and it is reported beside every table.

**End of season.**
- **Targets:**
  - heading (days after sowing, from the heading date);
  - final height (cm);
  - `yield_adjusted` (t/ha, primary);
  - raw `yield` (secondary).
- **Cutoffs:** days 120, 150, 180 and 210 after sowing (about mid-February to mid-May).
- **Heading cutoffs:** heading is forecast only from cutoffs 120-180. The earliest heading in any year is day 205, so a
  cutoff of 210 could see heading.

## What the forecaster may use

**Allowed inputs:**
- the plot's own measurements dated on or before the issue date or cutoff;
- the variety id;
- markers;
- degree days up to the issue date.

**Not allowed:**
- any value dated after the issue date;
- weather after the issue date;
- the dataset's `*_blue` and `*_heritability` columns, which are computed from all plots of a year, test plots included;
- `harvest_date`, `spatial_check`, any heading information as an input, and field position (week 1).

**Future degree days:**
- When a forecast needs degree days at a future date, they are degree days so far plus the **training years' average
  daily temperature** for the calendar days in between.
- The real future weather is never used in week 1, in any split.

**Data handling:**
- All values are converted from float16 to float32.
- Same-date duplicates are averaged (371,517 -> 366,425 rows).
- Degree days use our definition (base 0 °C, daily mean 2 m air temperature, from sowing); see the plan.

## Splits and protocols

**Protocol 1, official splits:**
- Fit on train.
- Choose any setting on validation.
- Evaluate each test split once.

**Same-season information (caveat):**
- In test_plot and test_genotype, the training set contains other plots from the **same season**, including how they grew
  after the issue date. A baseline built from the training curves therefore "knows" how that season went.
- That is how the dataset defines those splits. In real use you would not know other plots' future in the current season.
- So these two splits measure "a new plot (or variety) in a known season". **The forecasting headline is the unseen-year
  splits** (test_environment, test_genotype_environment) and protocol 2.

**Protocol 2, leave one year out:** run only after protocol 1's test results are recorded.
- For each of the six years, train on all plots of the other five years (all splits) and test on all plots of the
  held-out year.
- Settings stay as chosen in protocol 1.
- 2019 consists entirely of official test plots, so its fold has the same year as protocol 1's unseen-year test but more
  plots.

## Baselines

All baselines are deterministic. Ridge strengths are picked on validation, per trait and horizon or cutoff:
- from {0.1, 1, 10, 100, 1000};
- for the marker model F2, from {10, ..., 10^6} in powers of ten, because it has 18,845 inputs.

**Near-future:**

| Id | Name | Forecast |
|---|---|---|
| N0 | Persistence | last measured value |
| N1 | Average curve (days) | training years' average curve at the target day after sowing |
| N2 | Average curve (degree days) | same, on the degree-day axis |
| N3 | Anomaly persistence | N2 at target + (last value - N2 at issue) |
| N4 | Variety anomaly | as N3 with the variety's own average curve from training plots; N3 if the variety has no training plots |
| N5 | Linear | ridge on: last value, values 7 and 14 days earlier (interpolated), slope over 14 days, N2 at issue and at target, degree days so far, degree-day increment, issue day; target is the change from the last value; one model per trait and horizon |

**Average curve details:**
- Average curves are built per year in fixed bins (7 days, or 50 °C·d), linearly interpolated.
- They are then averaged with **equal weight per year**, so 2021 (59 canopy-cover dates per plot) does not dominate.
- Bin widths are fixed here, not tuned.
- Forecasts are clipped to the physical range: canopy cover 0-1, height at least 0.

**End of season:**

| Id | Name | Forecast |
|---|---|---|
| F0 | Training mean | mean of the training years' means (equal weight per year) |
| F1 | Variety mean | F0 + the variety's average deviation from its year's mean, over training plots; F0 if unseen |
| F2 | Genetics | F0 + ridge on markers predicting the deviation from the year mean; fit and evaluated on the 312 genotyped varieties only, with F0 and F1 reported on the same subset |
| F3 | In-season linear | ridge on features at the cutoff, plus F1's variety deviation (0 and an indicator if unseen); one model per target and cutoff |

**F3 features at the cutoff:**
- canopy cover: last value, maximum so far, slope over the last 14 days;
- height: last value, and an indicator when there is none yet;
- degree days so far.

**Year level:** none of these can predict a year's overall level (for example 2021's low yields). So end-of-season results
are also reported as **within-year Spearman rank correlation**, the breeding-relevant question of which plots do best.

## Metrics and uncertainty

**Near-future:**
- MAE (primary), RMSE;
- skill = 1 - MAE / MAE of persistence.

**End of season:**
- MAE (primary), RMSE, median absolute error;
- within-year Spearman (averaged over the years in the test set).

**Uncertainty:**
- 95% bootstrap intervals, resampling **varieties** with replacement within the test set (2,000 resamples), because plots
  of one variety are correlated.
- Comparisons between baselines are paired: same resamples, difference of MAE.

**Yield:** reported with and without the 12 damaged plot-years (more than 4 robust SD below their year median).

**Breakdowns:** results are reported per split, per horizon or cutoff, per season phase, and per year (protocol 2).

## Checks run before any test number

1. No input row has a date after its issue date or cutoff (unit test on all samples).
2. Average curves, variety means, ridge fits and their standardisation use training plots only (unit test: they are
   unchanged when the test splits are removed from the data).
3. No plot-year appears in two splits.
4. Every test split is scored by one script call, run once.

## Changes after the validation run, before any test number

1. **F3 ridge strength is chosen on within-year error, not plain MAE.**
   - On the 2026-11-05 validation run, F3's gains in plain MAE came from recognising the year (degree days at the cutoff
     differ by year, and validation shares its years with training). Its within-year rank correlation was no better than
     F1.
   - Selecting on plain MAE therefore favours a year lookup, which cannot carry over to an unseen year.
   - Within-year error = each validation year's mean error is subtracted before taking the MAE.
   - F2 was already selected on deviations from the year mean. N5 is unchanged.
