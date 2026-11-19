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

## Clarifications made while implementing, before any week-6 run

Everything below was decided before any model was trained or any real generation was scored.

1. **Distance features are recomputed from the thumbnails.** The week-4 distance used DINOv2 features of the full-size
   photos. A generated photo only exists at 240x160, so comparing it with full-size features would favour real
   photos. DINOv2 features are therefore recomputed, with the same weights and preprocessing, from the 240x160
   thumbnail of **every** picture compared (today's photo, the retrieved photo, the true future photo, the generated
   photos) and of all photos used for the standardisation. The absolute distances are not comparable with week 4's;
   the paired comparisons are on one footing.
2. **The estimator is `microsoft/resnet-18`** (Apache-2.0, pinned revision in `src/readout.py`) through the
   `transformers` library, not torchvision. It is the same architecture with ImageNet weights. Fixed settings: 3
   epochs, batch 128, AdamW at 3e-4 with a one-cycle schedule, L1 loss, horizontal flips. It is trained on every photo of
   the five training years with a measured cover within 1 day, and read out on every photo of every year.
3. **R-E+P's read-out is in-sample for the estimator.** The retrieved photos come from the training years, which the
   estimator was trained on. Its read-out there is close to the measured cover, which is what retrieval matched. It is
   scored as designed (the estimator's cover on the picture). This favours R-E+P slightly in rule 1; a read-out win
   over R-E+P is therefore the harder one.
4. **Height in the condition.**
   - A height forecast exists for only 56% of the queries (week-3 forecasts of the height sample issued up to 3 days
     before the query's issue day, never later). For the rest the missing flag is set, in training as in sampling.
   - Gen-persist uses the latest measured height on or before the issue day. Gen-oracle uses the height measurement
     nearest the target day within 3 days.
5. **Settings fixed now, not tuned:**
   - 8 conditioning tokens from the numbers;
   - 10% of training steps drop the numbers (for guidance), and sampling uses guidance 3.0, a conventional value;
   - 30 DDIM steps, one seed per query (the query id plus 1,000,000 times the variant's index);
   - batch 32, AdamW at 5e-5 for the UNet and 5e-4 for the number network, 200 warm-up steps, gradient clipping at 1;
   - mixed precision (bf16, or fp16 where bf16 is unsupported), weights kept in fp32.
6. **The noise check** uses 84 random queries per fold (504 in all), with three extra samples of Gen each.
7. **The verdict function:** the folds whose estimator passes the gate decide rule 1, and "at least 4 of 6 years" then
   counts those folds only; with fewer than four passing folds, rule 1 cannot be met. Faithfulness must hold at both
   horizons.
8. **Code checks, not results.**
   - Six tests added, run on random tiny networks and toy data. They found and fixed one bug before any run: after
     resuming from a checkpoint, the learning rate was read back from the warmed-up optimiser, so a resumed run was not
     the same run. A resumed run now equals an uninterrupted one.
   - The scoring step ran end to end on stand-in generations (random numbers). It wrote every output; the numbers mean
     nothing. All 8,113 queries have a retrieved photo.
9. **GPU environment:** a separate `.venv-gpu` (`scripts/swan/setup_gpu.sh`, CUDA PyTorch and `diffusers` 0.43.0). The jobs
   are pinned to L40S or A30 cards, as the cauliflower project's were.
10. **The smoke test** trains 300 steps on fold 2019 and generates 16 queries, in separate directories
    (`WEEK6_TAG=smoke`). The scoring step refuses to run with that tag.

## Clarifications made while implementing, before any week-6 run

1. **Where code is checked.** From 2026-11-19 nothing is run on the laptop (the user's instruction); every check runs
   on Swan. The first job of `submit_week6_part1.sh` is the full test suite.
   - **Before that instruction,** the new tests and a dry run of the scoring code on random stand-in data were run on the
     laptop. Nothing real was run. One bug was found and fixed that way: after resuming from a checkpoint, the learning
     rates were read back from the saved optimiser, which had already been warmed up. The generator code now uses the
     fixed rates, and a test checks that a resumed run equals an uninterrupted one.
2. **Height is often missing.** Week 3's height forecasts exist for only 56% of the queries, because height and cover
   are measured on different dates. The height forecast used is the latest height sample issued on or before the query's
   issue day and at most 3 days before it; never a later one. Otherwise the height input is flagged as missing, as
   designed. So the generator is conditioned on cover alone for about 44% of queries.
3. **The distance is recomputed from thumbnails.**
   - DINOv2 features are computed again for **every** picture compared (today's photo, the retrieved photo, the true
     future photo and the generated photos), all from 240x160 images resized to 420x280 with the week-3 preprocessing.
     Real and generated pictures are then treated identically.
   - So absolute distances are **not comparable** with week 4's (0.43 and 0.56, from full-size photos). Only the paired
     differences between methods within this week are.
   - The standardisation still uses the fold's training years' photos only.
4. **The read-out network.** `microsoft/resnet-18` (Apache-2.0, pinned revision in the code) loaded through
   `transformers`, not `torchvision`, to avoid another dependency. Same architecture and ImageNet pre-training. Settings
   fixed: 3 epochs, batch 128, AdamW with a one-cycle rate of 3e-4, L1 loss, horizontal flips.
5. **Generator settings, fixed in code and not tuned:**
   - UNet rate 5e-5, number network rate 5e-4, 200 warm-up steps, batch 32, gradient clipping 1.0;
   - numbers replaced by a learned "no condition" in 10% of steps, so sampling can use guidance;
   - sampling: 30 deterministic DDIM steps, guidance 3.0 (a conventional value), 8 conditioning tokens, one seed per query.
   - **Only the number of steps is open.** It is set after the smoke test.
6. **The read-out of retrieved photos is in-sample.** R-E+P photos come from the training years, which the fold's
   estimator was trained on, so its read-out of them is probably more accurate than on new photos. This favours R-E+P in
   the read-out comparison. It is a known limitation, reported as such.
7. **The rule's year count.** The read-out rule counts years among the folds that pass the estimator gate, and needs at
   least 4. If fewer than 4 folds pass, it cannot be met. Queries with no retrieved photo would be dropped from every
   comparison, so all methods use the same queries.
8. **The smoke test** is a code check on fold 2019 in separate directories (`WEEK6_TAG=smoke`): 300 training steps, 16
   generated queries. Only runtime, memory and a finite loss are looked at. Its images are not scored and never used.
