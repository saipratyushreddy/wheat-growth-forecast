# Week 5: how much would knowing the coming weather help? (an upper bound)

[Back to the report](06-report.md) · [Week 4](05-week4-design.md)

Written and committed before any code or result. Anything changed after results are seen is marked as a change, with
the reason.

## Why

Across weeks 1-4, most of a new year's error is one shift shared by all plots of the field on a given day (docs/04
part A; docs/05). Every model fills the days between the issue date and the target with the **training years' average
temperature**. The report's first future direction is to use real weather forecasts instead.

Archived weather forecasts for 2016-2022 at this site are real work to get. This design first asks whether they could
pay off at all. It replaces the average temperature with the **temperature that actually happened**.

**This deliberately breaks the input rule** "no weather after the issue date". Every model here is labelled
**"-true"**. It is an upper bound on what a perfect temperature forecast could give, and never a forecast. No "-true"
model enters the report's results as a forecaster.

## What changes

Only the degree days between the issue date (or cutoff) and the target:
- the forecast's target degree days become the plot's **actual** degree days on the target date, `target_gdd_true`,
  which is already computed for every sample by `near.build_samples`;
- the degree-day increment input `gdd_inc` becomes the actual increment;
- for heading, the degree days reached after the cutoff use the actual temperature.

Nothing else changes: the curves, training plots, settings and the other inputs are the same. All settings are frozen
from weeks 1-3; nothing is tuned.

## Models

| Id | Model | Compared with |
|---|---|---|
| N3-true | N3 with the actual target degree days | N3 |
| G-true | G (week-2 settings) on the residual from N3-true, with the actual `gdd_inc`, in training and held-out rows alike | G |
| L1-true | the degree-day heading level (week 2's L1) with actual temperature after the cutoff | L1, F0 |

**Leaving out S (the GRU), E and the photo models, on purpose:**
- S alone never beat N3 (docs/03).
- The question is whether temperature explains the shared shift. N3-true answers it with no learning at all, and G-true
  checks that a learned model can use it.
- This keeps the run to one cheap job.

**Soybean:** N3-true is also run on soybean (leave-one-season-out, 7 seasons, canopy cover) as an independent check.
It is cheap and needs no fit.

## Protocol

- **Leave-one-year-out only** (wheat: 6 years, the same folds as weeks 2-3; soybean: 7 seasons).
- The official test splits are **not** used. Their single scoring is spent, and an upper bound is not a test of a
  forecaster.
- Traits and horizons as before: canopy cover and height at 7, 14 and 28 days. Heading at cutoffs 120, 150 and 180.
- Intervals: 95%, resampling varieties, 2,000 resamples. Year counts as before.

## Reported

1. Pooled and per-year MAE: N3 vs N3-true and G vs G-true, per trait and horizon, also by season phase.
2. **The shared vs plot-specific split** (docs/04 part A2), for N3 and N3-true. How much of the field-wide error does
   actual temperature remove, per year?
3. Heading level MAE: L1-true vs L1 vs F0, per cutoff and year.
4. Soybean: N3-true vs N3, per season and horizon.

## Rule (decides the next step)

**"Temperature forecasts are worth pursuing"** for a trait and horizon (14 or 28 days) when N3-true's pooled MAE is **at
least 10% below N3's**, its 95% interval for the difference is below 0, **and** it is lower in at least 5 of 6 years.

- **Why 10%:** it matches the best gain learning gave over N3 (docs/03). A real forecast gets only part of the upper
  bound, so a smaller bound is not worth the effort.
- **Why 14 and 28 days only:** at 7 days, canopy cover is near the measurement noise floor, and the temperature
  forecast for one week ahead hardly differs from what happens.

**Decision:**
- **If the rule holds for at least one trait at 28 days:** the next design gets archived real forecasts, for example
  ECMWF re-forecasts or a historical forecast archive. It replaces the average temperature with what was forecast on
  each issue date.
- **Otherwise:** the shared shift is not a temperature effect. Other causes, such as rain, soil water, disease or
  management, are listed as open, and direction 1 of the report is closed.

Reported without a rule: G-true vs G; L1-true; the soybean check.

## Expectation, stated before results

- N3-true helps most at 28 days and in spring, when growth tracks temperature.
- It does not fix 2016's ripening-period loss (docs/04 A1). Ripening speed depends on water and disease as much as on
  temperature.
- On soybean the gain is larger, because canopy closure there is fast and temperature-driven.

## Checks

- **A unit test:** with the "-true" switch off, the code reproduces week-3's N3 and G predictions for one fold exactly.
- **A unit test:** with the switch on, `target_gdd` equals `target_gdd_true` for every sample.

## Practical

- **Code:**
  - `src/run_week5.py`, reusing `near.fit_predict` and `near2` with a switch for the target degree days;
  - `final2`'s heading level with an actual-temperature option;
  - `scripts/swan/w5_*.slurm`.
- **Runs:** on Swan, once. Outputs go to `outputs/week5/`.
- **Then:** the results go into this file, and one paragraph and a figure go into the report.

## Clarifications made while implementing, before any week-5 result

1. **Where the checks live.**
   - The N3 check is a unit test: the 2016 fold reproduces week 3 exactly, and with the switch on the target degree
     days equal the actual ones.
   - The G check runs inside every wheat fold on Swan, against that fold's week-3 predictions; the fold stops if they
     differ. The G check needs the week-3 fold files, which are on Swan.
2. **Code check with stand-in data, not results.** Scoring ran on week-3 predictions with random noise standing in
   for the "-true" columns.
   - Heading: the level is one value per year and cutoff, since all plots of a year share the sowing date and the
     weather. It is never the "not reached" fallback.
3. **Heading has one level per year,** so its per-year interval is wide by construction. The year count is the useful
   number.

## Results (2026-11-08)

Run once on Swan (jobs 18851739-18851741). All 27 tests passed. In every wheat fold, N3 and G with the switch off
reproduced week 3 exactly (largest difference 0.0). Outputs are in `outputs/week5/`: `near.csv`, `near_phase.csv`,
`field_split.csv`, `heading.csv` and `soy.csv`.

### The rule: not met at 28 days, so the decision is "do not fetch weather forecasts yet"

N3-true vs N3, pooled over the six held-out years. The difference is in canopy-cover points or cm; "gain" is the
difference relative to N3's MAE.

| Forecast | N3 | N3-true | Difference [95%] | Gain | Years better | Rule |
|---|---|---|---|---|---|---|
| Canopy cover +14 d | 8.69 | 8.57 | -0.12 [-0.14, -0.09] | 1% | 3 of 6 | no |
| Canopy cover +28 d | 10.96 | 10.36 | -0.60 [-0.66, -0.54] | 5% | 4 of 6 | no |
| Height +14 d | 4.87 | 4.29 | -0.58 [-0.61, -0.55] | 12% | 5 of 6 | **yes** |
| Height +28 d | 7.63 | 6.75 | -0.88 [-0.95, -0.82] | 12% | 4 of 6 | no (years) |

The rule holds only for height at 14 days. Height at 28 days has the gain but is better in only 4 of 6 years: 2016
(+0.62 cm) and 2019 (+0.02) are worse. By the rule fixed in advance, the shared shift is not mainly a temperature effect
for the near-future forecasts, and the archived-forecast step is not taken on this evidence.

### What else the run shows (reported without a rule)

- **Temperature matters where expected and not elsewhere.**
  - Height in spring, the stem-extension period: N3-true gains 16-24% (14 d: 5.84 to 4.42 cm; 28 d: 9.08 to 7.07).
  - Canopy cover in autumn-winter: 7-22%.
  - Canopy cover in spring and from heading on: about 0 (-3% to +2%).
  - Senescence, ripening and canopy closure are not driven by degree days on our definition.
- **Canopy cover in 2019 gets worse with the actual temperature** (+1.31 points at 28 days). Matching the temperature
  does not match how cover develops in every year.
- **Field-wide vs plot-specific:**
  - The plot-specific error is unchanged by N3-true, as it should be.
  - The field-wide part falls in most years and horizons. Canopy cover +28 d: 2017 12.8 to 8.7, 2016 8.7 to 6.8.
  - It rises in some, for example 2019 3.0 to 4.7.
  - So temperature explains part of the shared shift, in some years.
- **A learned model uses the actual temperature better than N3 does.** G-true vs N3 at 28 days: canopy cover -2.0
  points (-18%, 5 of 6 years), height -1.2 cm (-16%, 3 of 6). G-true vs G: -0.5 to -0.9 at every horizon for cover.
- **Heading level: the clearest effect.** Heading day from the actual degree days (L1-true) vs the training mean (F0):
  5.8 vs 9.7 days, better in 5 of 6 years. 2016 is the exception: 10.8 vs 10.1. Compared with L1 (average weather after
  the cutoff), L1-true is better in all 6 years at every cutoff, by 4.4-5.5 days.
  - So the year-to-year shift of heading, which no model in weeks 1-4 could forecast, is mostly a temperature effect.
  - L1-true is the same at every cutoff, because with the actual weather the cutoff no longer matters.
- **Soybean:** N3-true beats N3 by 3-6% (-0.30, -0.71 and -1.04 points at 7, 14 and 28 days; 5, 6 and 7 of 7 seasons).
  This is consistent but small.

### Against the expectation stated before results

- **"Helps most at 28 days and in spring":** right for height, wrong for canopy cover, where the gain is in
  autumn-winter.
- **"Does not fix 2016's ripening-period loss":** mixed. Canopy cover in 2016 improved (-1.6 at 28 days); height
  in 2016 got worse.
- **"Soybean gains more":** wrong. The relative gain is smaller than for wheat height.

### What this means

- **Near future:** a perfect temperature forecast would remove only a modest part of the shared error. The exception is
  height in spring. Since a real forecast gets only part of this bound, fetching archived forecasts for the near-future
  task is not worth it on this evidence. The rule says so, and the result agrees.
- **End of season:** heading level is different. Here the actual temperature cuts the error of a new year from 9.7 to
  5.8 days.
  - A real 2-6 week temperature forecast from the cutoff would get some of that. Seasonal forecasts have little skill,
    so likely only a small part.
  - This is a new hypothesis, found in a part reported without a rule. Testing it needs its own design.
