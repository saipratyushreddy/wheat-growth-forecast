# Project plan (one month)

[Back to the project overview](../README.md)

## Question

From what has been seen of a wheat plot so far in the season (measurements, photos, weather, genetics), how well can we
forecast its near future (canopy cover and height 7, 14 and 28 days ahead) and its end of season (heading date, final
height, yield), how sure should we be, and does it still work for plots, varieties and years the model has never seen?

This is the professor's "predict the future of the plant", made testable. The closest published work, Plant Time Bender
(Roth et al., bioRxiv 2025, doi:10.1101/2025.05.12.653430), forecasts yield only, from full-season inputs, without
prediction ranges, and did not beat a random-regression baseline on FIP 1.0. Early-season cutoffs, forecasts of the
growth traits themselves, and calibrated ranges are open.

## Data

FIP 1.0 (Roth et al., *GigaScience* 2025, giaf051): ETH Zurich winter-wheat trials at one site, 4,058 plot-years,
904 varieties, harvest years 2016-19, 2021, 2022. Official splits (plot-years per split):

| Split | What is new to the model | Plot-years | Years |
|---|---|---|---|
| train | - | 2,930 | all but 2019 |
| validation | the plot | 380 | all but 2019 |
| test_plot | the plot | 250 | all but 2019 |
| test_genotype | the variety | 246 | all but 2019 |
| test_environment | the year (2019) | 190 | 2019 |
| test_genotype_environment | variety and year | 62 | 2019 |

Leave-one-year-out (train on five years, test on the sixth) is added because the official unseen-year test is a single
year. Optional second dataset: FIP1SOY (soybean, same group, same site, same table layout), as a replication.

## Weeks

| Week | Work | Result |
|---|---|---|
| 1 | Evaluation design, simple baselines ([design](02-baselines-design.md)) | Error vs lead time per split; how hard the problem is |
| 2 | Sequence model on measurements + weather + genetics | Does learning beat the baselines, where |
| 3 | Frozen photo features (DINOv2); conformal prediction ranges | Do photos add anything beyond the measurements; calibrated ranges |
| 4 | Future-photo demo by retrieval; soybean replication if its data check passes; write-up | Report, figures, future directions |

Fallbacks fixed in advance: if photos are not on Swan by week 3, week 3 is ranges only; if the week-2 model does not beat
the baselines, week 3 starts with understanding why.

## Rules

Each design is written and committed before its results exist. Settings are chosen on validation only; each test set is
evaluated once per design. Differences are reported with 95% bootstrap intervals (resampling varieties). Negative results
are reported.

## Data check findings that shape the design (2026-11-05)

- **Storage:** all values are float16; they are converted to float32 before any arithmetic. Canopy cover is stored as a 0-1
  fraction although its unit field says %. No inf or NaN values in any series.
- **Coverage per plot-year:** canopy cover 19-59 dates (from about day 20-57 after sowing to harvest); height 21-47 dates,
  mostly from day 120-180 after sowing (2019 from day 19); head counts from day 185-210; senescence 2016-18 only.
- **Heading:** within a year it spans only 12-16 days, rated on 11-16 distinct dates, so the bar to beat is the variety's
  average from other years. Earliest heading in any year is day 205 after sowing.
- **Degree days:** our degree days (base 0 °C, daily mean air temperature at 2 m, from sowing) do not reproduce the
  dataset's heading values in °C·d: the offset is constant within a year but differs between years (-63 to +100 °C·d;
  2017 is the largest). No common definition matched. We use our own definition everywhere and forecast heading in days.
- **Yield:** 12 plot-years are more than 4 robust SD below their year's median (7 in 2021), with very short final height
  (0.14-0.42 m): damaged plots, kept, reported with and without. `yield_adjusted` (rescaled to the 2019 plot size) is the
  primary yield target because it is comparable across years.
- **Genetics:** markers for 312 of 904 varieties (53% of plot-years). 2017 heading heritability is 0 because only 3
  varieties had two scored plots.
- **Weather:** hourly, identical for all plots within a year.
