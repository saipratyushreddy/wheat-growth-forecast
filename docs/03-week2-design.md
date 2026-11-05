# Week 2: learned models on measurements, genetics and weather

[Back to the plan](01-plan.md) · [Week 1 baselines and results](02-baselines-design.md)

Written and committed before any week-2 code is run. Anything changed after results are seen is marked as a change,
with the reason. Everything in week 1's "What the forecaster may use" still holds; this document lists only what is new.

## What week 1 told us

1. **Near-future bar: N3**, the average degree-day curve shifted to the plot's current level. The linear model N5 beat it
   in only about half of the held-out years.
2. **End-of-season ranking comes from the variety:** its history (F1) or its markers (F2). Measurements up to day 210 added
   little in linear form.
3. **No baseline forecasts a new year's level.** F3 learned to recognise the year from degree days at the cutoff, which
   does not carry over to an unseen year.
4. **2019 is the easiest year**, so leave-one-year-out is the headline.

Week 2 therefore:
- predicts **deviations**: from N3 (near future) and from the year mean (end of season), so a model cannot memorise
  years;
- combines **genetics with in-season measurements**;
- **chooses settings by leaving out whole years** inside the training data, not on the official validation split (see
  "Choosing settings").

## Near-future growth

Same samples, horizons, targets, metrics and season phases as week 1 (`data/derived/near_samples.parquet`, 649,991 rows).

**Target:** the residual `target - N3`, in trait units. The forecast is `N3 + predicted residual`, clipped to the physical
range. A model that learns nothing reproduces N3.

### Inputs

Every input is dated on or before the issue date.

**History (from the plot's own measurements):**
- Anomaly now: `a = last - N2(issue degree days)`.
- Anomaly 7 and 14 days earlier (interpolated), and its slope over 14 days.
- Days since the previous measurement; number of measurements so far.
- Rise of the average curve over the forecast: `N2(target) - N2(issue)`.
- The other trait's latest anomaly on or before the issue date, and its age in days. Missing if there is none.

**Timing:**
- Issue day after sowing, degree days so far.
- Climatological degree-day increment to the target (week-1 rule: training years' average temperature).
- Days to the target, nominal horizon.

**Genetics.** Variety features for a plot are always computed from training plots of **other years** only, the same rule
for training and test rows:
- Variety's mean N3 residual for this trait and horizon, and the number of plots behind it.
- Variety's deviation from the year mean for final height and heading (F1-style).
- Marker prediction of those two deviations: ridge on markers (F2-style, ridge strength frozen from week 1's
  `official_alphas.json`), fit on other years' training plots.
- Top 10 principal components of the markers, fit on training varieties only; a genotyped flag.
- Varieties without markers or history get missing values. The models handle missing values natively.

**Weather to date** (arm G+W only, not in the main models):
- Over the 14 days up to the issue date, relative to the training years' average for those calendar days:
  - mean air temperature at 2 m;
  - precipitation sum;
  - mean solar irradiance;
  - mean air humidity at 2 m.
- Requires extending `src/series.py` to keep these four daily series (read by row group, as now).
- **Why it is only an ablation:** weather is the same for every plot of a year. With five training years, a model can use
  it to recognise the year. G+W is reported as a check of whether recent weather adds anything; it is not chosen for the
  headline unless it beats G under the success rule below.

### Models

| Id | Model | Notes |
|---|---|---|
| G | Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, absolute-error loss) on the inputs above, minus weather | one model per trait, all horizons pooled |
| G+W | G plus weather to date | ablation |
| S | Sequence model (GRU), below | the plan's "sequence model" |
| E | Mean of G and S | fixed in advance, not tuned |

**Settings for G:**
- 8 configurations: max leaf nodes {15, 63} × iterations {200, 600} × min samples per leaf {100, 1000}.
- Fixed: learning rate 0.05, L2 regularisation 1.0, no early stopping.

**S, the sequence model** (PyTorch, CPU):
- **Input sequence.** One sequence per plot-year: every canopy-cover and height measurement in order of day after
  sowing. Each step carries:
  - trait (one-hot);
  - anomaly against the training average curve (canopy cover as a fraction, height in m);
  - day after sowing / 300;
  - degree days / 3000;
  - log(1 + days since the previous step).
- **Starting state.** The GRU's initial state is a linear map of the plot's static genetics: marker PCs, genotyped flag,
  variety features.
- **Prediction.** For each sample, the hidden state is taken after the last step dated on or before the issue date.
  - The head is a 2-layer MLP (width 64) that receives the hidden state, the trait, the horizon inputs (degree-day
    increment, curve rise, days to target) and the static genetics.
  - It predicts the residual from N3.
- **Training settings:**
  - 1-layer GRU, hidden size 64, dropout 0.1;
  - AdamW, learning rate 1e-3, weight decay 1e-4;
  - batches of 64 plot-years, L1 loss;
  - ensemble of 5 seeds.
  - Number of epochs chosen from {10, 25, 50}; nothing else is tuned.
- **Causality.** Because the hidden state at a step depends only on earlier steps, later measurements cannot leak into a
  forecast. A unit test checks this.

**Not used**, as in week 1:
- photos (week 3);
- field position;
- `*_blue`, `*_heritability`, `harvest_date`, `spatial_check`;
- any heading information as an input;
- future weather.

## End of season

Same targets, cutoffs and metrics as week 1. Heading is forecast only from cutoffs up to day 180.

**Decomposition:** forecast = **level** (the year's mean) + **deviation** (the plot's place within its year).
- MAE judges both together.
- Within-year Spearman judges the deviation only.

### Level

| Id | Level forecast |
|---|---|
| L0 | F0, the training years' mean |
| L1 | heading only: the day on which degree days reach the training years' mean degree days at heading |

**How L1 works:**
- Degree days come from our definition, the same in every year, so the year offset noted in the plan does not apply.
- After the cutoff, degree days are extrapolated with climatology (week-1 rule).
- L1 has one fitted number. It asks whether a warm or cold season so far moves heading, which is the one level question
  physics answers without needing more than five training years.

No other level model is tried: with five training years, a learned year-level model has five data points.

### Deviation model D

**Target:** the plot's value minus its year's mean over training plots.

**Features at the cutoff, each centred within its year:**
- **Canopy cover:**
  - anomaly of the last value against the training average curve at its degree days;
  - maximum anomaly so far;
  - slope over the last 14 days.
- **Height:** anomaly of the last value; slope over the last 14 days; a missing flag.
- **Genetics:** as in the near-future model:
  - variety deviation for this target, from other years only, and its plot count;
  - marker prediction of it, from other years only;
  - marker PCs; genotyped flag.

**Within-year centring:**
- Each in-season feature has the mean over all plots of that year subtracted. Only values dated on or before the cutoff
  are used, and test plots are included.
- This is what a breeder can do on the day of the forecast: the whole trial has been measured up to then.
- It removes the year signal that F3 exploited. A leakage test checks that the centring never uses a value after the
  cutoff or any target.

**Models:**
- D-lin: ridge, strengths {0.1, 1, 10, 100, 1000}.
- D-gbm: gradient-boosted trees, squared error.
  - max leaf nodes {7, 15} × iterations {100, 300};
  - min samples per leaf 50;
  - learning rate 0.05.
- **Primary D:** one per target and cutoff, chosen by within-year error (below).

**Forecasts reported:** L0 + D for all targets; L1 + D for heading.

## Choosing settings

**What is chosen:**
- G's configuration and S's epoch count, per trait;
- D-lin vs D-gbm and their settings, per target and cutoff.

**How:**
- Chosen once in protocol 1 by **leave-one-year-out inside the official train split** (its five years: 2016-18, 2021,
  2022). Each training year in turn is held out; the rest train.
- Score: mean over held-out years of MAE. For D, within-year MAE: each held-out year's mean error is removed, as in
  week-1 change 1.
- Settings are then frozen for the official test and for leave-one-year-out (protocol 2).

**Why not the official validation split:**
- Validation plots share their years with training.
- Week 1 showed that selecting on them rewards year lookup.
- The validation split is still scored once, as a sanity check, before the test.

## Protocols and runs

**Runner:** `src/run_week2.py`, writing to `outputs/week2/`. Same protocols as week 1.

**Steps, in order:**
1. **Tuning:** `--protocol official --stage tune`. Inner leave-one-year-out on train; writes
   `outputs/week2/settings.json`. Can be rerun until the code is right. Any rerun after a change of design is recorded
   here.
2. **Validation sanity check:** `--protocol official --eval validation`. Fit on train, score validation.
3. **Official test, once:** `--protocol official --eval test`. Refuses to run if its output exists.
4. **Leave-one-year-out:** `--protocol loyo`, settings frozen. **This is the headline.**
   - Refuses to run before step 3.
   - Fits every model, including S's five seeds, once per held-out year.

**Week-1 baselines** (N0-N5, F0-F3) are recomputed in the same runs on the same samples, so every comparison is paired.

## Metrics, intervals and what counts as a win

**Metrics:** as in week 1. Additionally:
- the paired MAE difference against N3 (near future);
- the paired difference in within-year Spearman against the better of F1 and F2 (end of season; genotyped subset when F2
  is involved).

**Intervals:**
- 95% bootstrap intervals, 2,000 resamples of varieties.
- For Spearman: varieties are resampled within each year, Spearman is computed per year, then averaged over years.

**Success rules**, fixed now, judged on leave-one-year-out:

| Task | A model "beats the bar" for a trait and horizon (or target and cutoff) when |
|---|---|
| Near future | the paired MAE difference vs N3, pooled over the six years, has its 95% interval below 0 **and** the model has lower MAE than N3 in at least 4 of 6 years |
| End-of-season ranking | the within-year Spearman difference vs max(F1, F2) has its 95% interval above 0, pooled over years **and** in `test_genotype` (new varieties) |
| End-of-season level | L1 has lower heading MAE than F0 in at least 4 of 6 years |

- The official unseen year (2019) is reported beside the headline, never instead of it.
- If no model beats N3, the plan's fallback applies: week 3 starts with understanding why.

## Leakage checks added to `tests/test_leakage.py`

1. **Variety features:**
   - a training row's variety features never use a plot of its own year;
   - a test row's never use a test plot.
2. **Weather to date** uses only days on or before the issue date.
3. **GRU causality:** perturbing measurements after the issue date leaves the forecast unchanged.
4. **Model fit ignores test plots:**
   - G and D predictions for training and validation plots are unchanged when test plots are removed.
   - Removing test plots' targets and post-cutoff measurements does not change D's predictions for any plot. This is the
     check for the within-year centring, which uses test plots' pre-cutoff values.
5. **Marker PCs** are fit on training varieties only.

## Practical

- New dependency: `torch` (CPU wheel), added to the README's setup line.
- Memory: the machine has about 3 GB free. G is fit on float32 arrays per trait, and weather is read by row group.
- Files:
  - `src/features2.py`: variety, marker and weather features;
  - `src/near2.py`: G, G+W;
  - `src/seq.py`: S;
  - `src/final2.py`: L1, D;
  - `src/run_week2.py`: the runner.

## Clarifications made while implementing, before tuning or any result

These settle points the design left open. None was prompted by a result.

1. **Variety heading from other years is allowed as an input.**
   - The design lists it under genetics, but its "not used" list repeats week 1's "any heading information as an
     input".
   - Resolved the way week 1's F1 already works:
     - the plot's own heading, or any heading in the forecast season, is never an input;
     - the variety's heading in other years is variety history and is allowed.
2. **G+W uses G's tuned configuration.** It is not tuned separately, so the ablation differs from G only in its inputs.
3. **S's epoch count per trait comes from one network.** S is trained on both traits at once, so "per trait" means: one
   training run, with the checkpoint for each trait chosen from {10, 25, 50}.
4. **Variety features use slightly fewer years for training rows.**
   - "Other years" for a training row means the other training years: four in leave-one-year-out.
   - For a test row it means all five training years.
   - Test rows' variety features are therefore slightly less noisy than those the model was trained on. This is
     inherent to the other-years rule.
5. **Missing values in D-lin:**
   - missing features are set to 0, which is the year mean after centring;
   - flags mark missing canopy cover, height and variety history;
   - D-gbm takes the missing values natively.
6. **Tuning uses only official-train plots' samples.** Other plots are never scored in tuning, so this changes no fit.
7. **The runner enforces the order of steps:**
   - the official test refuses to run before the validation sanity check, or a second time;
   - leave-one-year-out refuses to run before the official test, or a second time;
   - an interrupted leave-one-year-out resumes per held-out year. Settings are frozen and every fit is seeded, so a
     resumed fold is identical.
8. **Validation numbers were seen during a timing check.**
   - A timing check of G on the official split printed validation MAE for two configurations from the grid:
     - canopy cover 5.0 and 4.0 points vs N3 7.4;
     - height 2.9 and 2.2 cm vs N3 4.7.
   - Validation shares its years with training, so this is the year-lookup optimism this design avoids selecting on.
   - It was not used for any choice and is recorded here only because it was seen.
9. **Fix after a crash in tuning, on Swan.**
   - What happened: the tuning fold that holds out 2017 crashed in D-gbm.
   - Cause: at cutoff day 120, height has been measured only in 2017 and 2019. Tuning never trains on 2019, so with 2017
     held out the two height features are missing in every training row, and the tree model cannot be fit on an
     all-missing column.
   - Fix: D-gbm now drops any feature that is missing in all training rows. Such a feature carries no information, and
     D-lin already sees it as a constant 0.
   - Effect on the other folds: they all trained on 2017, so their tuning scores are unchanged, and only the 2017 fold
     is rerun.
   - This was prompted by the crash, not by any score. Tuning scores were visible in the logs, as tuning scores are
     meant to be.
10. **The sequence model is not bit-for-bit repeatable.**
    - Evidence: the 2017 tuning fold ran twice, before and after fix 9.
      - G's scores were identical.
      - S's held-out MAE differed in the third decimal, for example height after 10 epochs: 5.807 vs 5.834 cm.
    - Cause: CPU training of S on different Swan nodes or thread counts.
    - What this corrects: clarification 7 says a resumed leave-one-year-out fold is identical. That holds for every
      model except S, which agrees only to about 0.03 MAE units.
    - Settings were taken from the second run. Both runs pick the same epoch counts.

## Settings chosen by tuning (2026-11-05)

From leave-one-year-out inside official train, all five held-out years. Frozen in `outputs/week2/settings.json`, which
is committed.

- **G:**
  - canopy cover: 63 leaves, 200 iterations, 100 per leaf;
  - height: 63 leaves, 600 iterations, 100 per leaf.
- **S:** 25 epochs (canopy cover), 50 epochs (height).
- **D:**
  - heading: D-gbm (7 leaves, 100 iterations) at every cutoff;
  - final height: D-gbm (7 leaves, 100 iterations), except D-lin (strength 0.1) at day 180;
  - yield and adjusted yield: D-lin (strength 1000) at every cutoff.

## Validation sanity check (2026-11-05)

Run once on Swan, fit on train, scored on the official validation split. It was used only to look for bugs; nothing was
changed because of it.

- **The week-1 baselines recomputed in the week-2 run reproduce week 1's validation results:**
  - near future: same rows, MAE identical to 1e-12;
  - end of season: same rows, MAE within 2e-5 and Spearman within 1e-4, which is floating-point differences in the
    marker ridge on Swan.
- **The new predictions are well formed:**
  - every method has a prediction for every sample;
  - all predictions lie in the physical range;
  - L1 heading days lie between days 217 and 229.
- **The validation errors are not a result.** Validation plots share their years with training, so the learned models'
  much lower validation error (for example, canopy cover +7 d: G 3.6 vs N3 6.1 points) may come from same-season
  information. That is the reason this design selects on held-out years. The headline is leave-one-year-out.

## Results (2026-11-05)

**How the runs were done:**
- On Swan, with the settings above frozen.
- The official test splits were scored once (`outputs/week2/official_test_*.csv`, SLURM job 18824359).
- Then leave-one-year-out (`outputs/week2/loyo_*.csv`).
- The week-1 baselines were recomputed in the same fits, so every comparison is paired. N3's leave-one-year-out errors
  equal week 1's.

**Reading the tables:**
- Errors are MAE. Intervals are 95% and resample varieties.
- "Mean of years" averages the six held-out years' MAE, as in week 1.
- Paired differences pool all plots of the six years.
- The variety bootstrap does not capture year-to-year variation. The year count is the year-level check.

### Near-future growth

Leave-one-year-out (headline). The difference and year count are for E against N3; the last column applies the success
rule to every model.

| Forecast | N3 | G | S | E | E - N3, pooled [95%] | Years E < N3 | Models that beat the bar |
|---|---|---|---|---|---|---|---|
| Canopy cover +7 d (points) | 6.12 | 6.23 | 6.18 | 6.01 | -0.18 [-0.22, -0.14] | 5 of 6 | E |
| Canopy cover +14 d | 7.58 | 7.31 | 7.60 | 7.15 | -0.75 [-0.82, -0.69] | 4 | E, G, G+W, N5 |
| Canopy cover +28 d | 9.89 | 9.21 | 10.21 | 9.30 | -1.17 [-1.27, -1.07] | 3 | G, G+W |
| Height +7 d (cm) | 2.99 | 3.03 | 3.11 | 2.96 | -0.09 [-0.10, -0.08] | 4 | E |
| Height +14 d | 4.84 | 4.53 | 5.00 | 4.59 | -0.30 [-0.33, -0.27] | 5 | E, G, N5 |
| Height +28 d | 7.47 | 7.38 | 7.53 | 7.20 | -0.59 [-0.68, -0.50] | 3 | none |

**The learned models beat N3, but only by a little.**
- Under the success rule, some learned model beats N3 for 5 of the 6 forecasts.
- The gain is at most about 10%: canopy cover +28 d, 9.9 to 9.2 points.
- The gains are concentrated after day 150 (spring and heading to ripening). For example, canopy cover +28 d from
  heading on: N3 15.1, E 13.1 points.
- No model wins in every year. In 2016, G and E are worse than N3 at every horizon.
- At 14 days the week-1 linear model N5 also passes the rule, so part of the gain needs no learning beyond a linear
  fit.

**Recent weather adds nothing reliable.** G+W is better than G only for canopy cover +28 d pooled (9.75 vs 9.85), and
not in the mean of years.

**The sequence model alone never beats the bar.** It helps only inside the ensemble E.

**In the official unseen year (2019, `test_environment`), the learned models lose at 28 days:**
- G vs N3: canopy cover +1.1 points, height +1.4 cm.
- E vs N3: canopy cover +1.2 points, height +0.7 cm.
- At 7-14 days, E roughly ties N3 (canopy cover +14 d: -0.18 [-0.34, -0.02]).

**In known seasons the same models look far better.**
- On `test_plot` and `test_genotype`, G roughly halves N3's error. For example, height +14 d on `test_plot`: N3 4.54,
  G 1.82 cm.
- These splits share their seasons with training. The contrast with leave-one-year-out shows that most of a learned
  model's apparent skill on them is knowledge of how that season went, which a real forecast does not have. It does not
  carry over to a new year.

### End of season

**Ranking (within-year Spearman, genotyped varieties):** no target passes the success rule.

| Target | Cutoff | F1 / F2 (LOYO) | D (LOYO) | D - F2, LOYO [95%] | F2 / D (`test_genotype`) | D - F2, `test_genotype` [95%] |
|---|---|---|---|---|---|---|
| Heading | 180 | 0.815 / 0.817 | 0.816 | -0.001 [-0.016, +0.014] | 0.479 / 0.572 | +0.093 [-0.037, +0.239] |
| Final height | 180 | 0.913 / 0.914 | 0.922 | +0.008 [+0.002, +0.020] | 0.793 / 0.753 | -0.040 [-0.114, +0.068] |
| Final height | 210 | 0.913 / 0.914 | 0.933 | +0.019 [+0.010, +0.039] | 0.793 / 0.834 | +0.041 [-0.022, +0.136] |
| Adjusted yield | 210 | 0.249 / 0.230 | 0.218 | -0.030 vs F1 [-0.080, +0.025] | 0.290 / 0.278 | -0.012 [-0.162, +0.141] |

- **Final height:** measurements up to days 180-210 improve the ranking slightly in unseen years. Over all plots,
  including varieties without markers, the gain at day 210 is large: D 0.856 vs F1 0.739 (+0.117 [+0.100, +0.137]).
  For new varieties (`test_genotype`) the intervals include 0.
- **Heading:** D ties the variety baselines in unseen years. For new varieties it is higher but not significantly.
- **Yield:** D is worse than the variety history F1 at every cutoff (by 0.03-0.09). This is a negative result: in-season
  canopy and height do not help rank yield.

**Level of heading (L1, degree days):**
- **The rule is met at cutoffs 150 and 180, not at 120.** L1 has lower heading MAE than F0 in 4 of 6 years at 150 and
  180, and in 3 of 6 at 120.
- **Pooled, L1 + D is worse than F0 at every cutoff** (10.1-11.2 vs 9.7 days). It wins clearly in 2019 (5.7 vs 10.0
  days at cutoff 180) and 2021, but loses badly in 2016 and 2018 (2018: 12.5 vs 4.6 days).
- **The mean over years is a tie** at cutoff 180: 9.4 vs 9.5.
- A warm or cold season so far moves heading in the right direction in some years. A single degree-day threshold is
  too crude to rely on.

**Level of final height:**
- D does not forecast the level by design, and L0 + D (7.3-8.4 cm) is better than F0 (10.0 cm) only through D.
- The week-1 in-season linear model F3 has the lowest final-height MAE at cutoffs 120-180 (5.7-6.2 cm).
- So for final height, the season's measurements carry level information that transfers to a new year. This
  decomposition discards it.

### What this means

1. **Near future:**
   - In a new year, learning on top of N3 gains up to about 10%, mainly late in the season.
   - E is the best single choice.
   - The gain is not reliable year by year, and is negative in the official unseen year at 28 days.
   - Most of the skill that learned models show in known seasons is season information.
2. **End of season:** variety history and markers still carry the ranking. Measurements add only for final height late
   in the season, and nothing for yield.
3. **Plan fallback:** the plan's fallback ("if the week-2 model does not beat the baselines, week 3 starts with
   understanding why") applies in part. Week 3 should first look at where the learned models lose: 2016, and 2019 at
   28 days.
