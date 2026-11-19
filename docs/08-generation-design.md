# Week 6: generating the future photo of a plot

[Back to the report](06-report.md) · [Week 4 (retrieval)](05-week4-design.md) · [Week 5](07-weather-upper-bound-design.md)

Written and committed before any code or result. Anything changed after results are seen is marked as a change, with
the reason. All earlier input rules still hold.

## Why

The professor's ask was "predict the future of the plant". Weeks 1-5 forecast numbers. Week 4 showed a picture only by
retrieving a real photo of another plot that matches the forecast. That picture matches the stage of growth but not the
plot's own soil, stones and lighting, and at 14 days "today's photo of the same plot" is far closer to the future photo
than any retrieved one (distance 0.429 vs 0.565).

This week asks whether a **generated** picture does better: start from the plot's own latest photo and change it to
match the forecast.

**What we do and do not claim.**
- A generator gives a plausible plot at the forecast state. It will not reproduce the exact leaves; cauliflower showed
  that a loss matching one real future photo averages over leaf layout.
- What can be accurate is the plot-level state: canopy cover, height, colour and growth stage. The tests below check
  that, not leaf-level detail.

## Data and the question

- **Photos:** the 240x160 JPEG thumbnails made in week 3 (`data/photos/thumbs_<year>.parquet`, 2.1 GB, 12 files, 151,150
  photos, six years). Nothing is re-streamed. The resolution is the limit: a sharper version would need the photos
  re-streamed at a larger size, which is not part of this design.
- **Question:** given a plot's latest photo on or before the issue date and the week-3 forecast for 14 or 28 days ahead,
  does a generated photo describe the future plot better than the alternatives?

## Task

- **Input at forecast time:**
  - the plot's latest photo on or before the issue date;
  - the forecast canopy cover (E+P) and height (E) for the target day, and a missing-flag for height where there is none
    (week 3's forecasts, made without any future information);
  - the planned target day: days ahead, and days after sowing.
- **Output:** one generated photo for the target day.
- **Queries:** the week-4 retrieval queries (`outputs/week4/retrieval_queries.parquet`), unchanged: one per plot and
  horizon, 14 and 28 days, leave-one-year-out, 8,113 queries, each with a true future photo within 3 days of the
  target day. Using the same queries makes every comparison paired with week 4.

## Model

- **Start from** `timbrooks/instruct-pix2pix` (MIT licence, open; built on Stable Diffusion 1.5, CreativeML OpenRAIL-M).
  The revision is pinned in the code. This model already takes an input image and produces an edited image.
- **Changes:**
  - the text conditioning is replaced by a small learned network that turns the numbers (cover, height, missing flag,
    days ahead, days after sowing) into a few conditioning tokens;
  - the image encoder (VAE) is frozen. The UNet and the number network are fine-tuned.
- **Resolution:** the thumbnails' own, 240x160, resized to the nearest size the network accepts. No output is upscaled
  for scoring.
- **Not used:** the text encoder, any text prompt, any other pretrained weights.

### Training pairs (training years only)

- For each plot-year in the fold's training years, pairs of photos (a, b) of the same plot with b 7 to 35 days after a.
- Condition: the **measured** cover and height at b (the nearest measurement within 1 day of b; none: missing),
  days from a to b, and b's days after sowing.
- 10% of the training plots are kept out of training and used only to watch the loss, never to choose settings.
- At forecast time the measured values are replaced by the forecasts (the standard way to train a conditional model).

## Protocol

- **Leave-one-year-out**, the same six folds as before: train on five years' photos, generate for the sixth year's
  queries. The official test splits are not used.
- **Fixed budget, no tuning:** the same number of training steps and the same settings in every fold. The number of
  steps is fixed from a one-hour smoke test that measures speed (a clarification, before any fold is trained). No
  setting is chosen by looking at held-out-year images or scores.
- **One sample per query,** fixed seed. Each fold's generations are scored once.
- **Sample noise check:** on 500 queries, three samples per query, to report how much the read-out varies between
  samples of the same query.

## The cover read-out

To check that a picture matches a state, we need to measure canopy cover from a photo, including a generated one.

- **Estimator:** a small pretrained image network (ResNet-18, ImageNet weights) fine-tuned to predict the measured cover
  from a thumbnail. It is trained on the fold's training years' photos paired with a cover measurement within 1 day,
  and applied to the held-out year.
- **Gate, fixed now:** on **real** held-out-year photos its R² is at least 0.90 and its MAE at most 5 points of cover.
  If either fails in a fold, the read-out metrics for that fold are labelled unreliable and carry no claim; the
  distance metric and the figures remain.
- **Why not the week-3 DINOv2 regression:** its R² was 0.67, too low to judge small differences.

## Methods compared

| Id | The picture |
|---|---|
| B-now | the plot's own latest photo on or before the issue date, unchanged (copy forward) |
| R-E+P | the week-4 retrieved photo (nearest measured cover to the E+P forecast) |
| Gen | generated from the plot's latest photo and the E+P / E forecast |
| Gen-persist | generated with the condition set to "no change": the cover and height of the latest measurement. Tests whether the picture follows the numbers |
| Gen-oracle | generated with the **true** cover and height at the target. A ceiling only; uses the target |

## Metrics and rules

All paired, pooled over six years, with 95% intervals resampling varieties (2,000 resamples) and per-year counts.
"Better" means the pooled interval is below 0 **and** it holds in at least 4 of 6 years.

1. **Cover read-out error** (primary, only in folds that pass the estimator gate): the absolute difference between the
   estimator's cover on the picture and the **true** cover at the target.
   - **Gen vs B-now,** and **Gen vs R-E+P:** does the generated picture show the right state?
2. **Distance to the true future photo** (as in week 4: 1 - cosine on DINOv2 features, standardised with training years'
   photos):
   - **Gen vs B-now,** and **Gen vs R-E+P.**
3. **Faithfulness** (descriptive, with a stated bar): the absolute difference between the estimator's cover on Gen and
   the **conditioning** cover. Faithful when its mean is at most 1.5 times the estimator's MAE on real held-out photos.
4. **Does the picture follow the numbers:** Gen-persist vs Gen on read-out error, and Gen vs Gen-oracle. Reported
   without a rule. If Gen and Gen-persist read out the same, the condition is being ignored.

**Reported separately, per horizon (14 and 28 days).** A pass at one horizon is not a pass at both.

**What "the generation works" means, fixed now:**
- **Pass:** rules 1 and 2 are both met against B-now and against R-E+P at 28 days, and rule 1 against B-now and R-E+P
  at 14 days, and faithfulness is met.
- **Partial:** any other combination of passes. The report lists which.
- **Fail:** none met. This is reported as a negative result, not hidden.

## Expectations, stated before results

- **Read-out:** Gen beats B-now at 28 days (the plot has grown) and at 14 days less clearly. A forecast that beats
  persistence on cover (19.1 vs 9.3 points at 28 days) should carry into the picture if the model follows the numbers.
- **Distance:** at 14 days Gen probably beats R-E+P, since it keeps the plot's own soil and light. Whether it beats
  B-now (0.429) is uncertain: the true photo differs from today's mostly in leaf layout, which no model can predict.
- **Faithfulness:** probably the weakest part. A generator fine-tuned on 150k photos with a few numeric conditions can
  learn the stage of growth and may ignore fine differences in cover.
- **Overall:** plausible images that follow the forecast on cover, with no pixel-level accuracy. A partial result is the
  likeliest outcome.

## Figures

- Eight random queries (seed 0, from 2019 at 28 days, as in week 4), in a row each: today, Gen, R-E+P, the true photo,
  with the forecast and true cover and the estimator's cover on Gen.
- A "dose" strip for four random queries: Gen at cover set to today's value, the forecast, and the true value.
- A scatter of the estimator's cover on Gen against the conditioning cover.

## Leakage checks added to `tests/test_leakage.py`

1. Training pairs come only from the fold's training years, and a pair's target photo is later than its input photo.
2. At forecast time, the input photo is the latest on or before the issue date and the condition values equal the
   week-3 forecasts for that query (never the measured target).
3. The cover estimator is fit on the fold's training years only.
4. The fold's held-out year contributes no photo to training, to the loss monitoring split, or to the standardisation
   used by the distance.
5. The query set is exactly week 4's (same ids).

## Practical

- **Where:** everything on Swan. The `gpu` partition has L40S, A30, V100 and T4 cards, open to the account.
  - The existing environment has the CPU build of PyTorch. A separate `.venv-gpu` is built with the CUDA build of
    PyTorch and `diffusers`, so the CPU runs of earlier weeks stay reproducible.
  - Training data: 2.1 GB of thumbnails, 12 files, within the file quota.
- **Code:**
  - `src/gen_data.py`: pairs, conditions, queries;
  - `src/gen_model.py`: model, training, sampling;
  - `src/readout.py`: the cover estimator;
  - `src/run_week6.py`: folds, scoring, figures;
  - `scripts/swan/w6_*.slurm`.
- **Order:**
  1. tests;
  2. smoke test on one fold: a few hundred steps, to measure speed and check that the code runs. Only runtime, memory
     and "the loss is finite" are looked at;
  3. a clarification fixing the number of steps and the time request;
  4. the cover estimator for each fold, and its gate on real held-out photos;
  5. six folds trained, in parallel;
  6. generation for the queries and the noise check;
  7. scoring, once;
  8. figures and the write-up into this file and the report.
- **Known unknowns, to be settled in clarifications before any fold is trained:** the GPU time limit of the account's
  quality of service, the number of training steps, and whether the 240x160 size fits the network's size rules without
  padding.

## After scoring

A gallery and a small demo page built from the saved generations only, with the metric results beside the pictures. The
demo makes no claim beyond what is scored here.
