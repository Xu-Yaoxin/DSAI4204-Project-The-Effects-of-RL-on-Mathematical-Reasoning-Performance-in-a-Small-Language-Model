# Paired RLVR evaluation

**SYNTHETIC PIPELINE TEST — not model results.**

Split: `test`. Questions: 8. Independent responses per question and model: 16.
95% intervals: 2000 paired bootstrap replicates, resampling questions; seed 42.

| k | Base pass@k | RL pass@k | RL − Base (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.2109 | 0.2891 | +0.0781 [-0.0078, +0.1797] |
| 2 | 0.2667 | 0.3552 | +0.0885 [-0.0156, +0.2281] |
| 4 | 0.3327 | 0.4014 | +0.0687 [-0.0625, +0.2231] |
| 8 | 0.4083 | 0.4375 | +0.0292 [-0.1583, +0.2125] |
| 16 | 0.5000 | 0.5000 | +0.0000 [-0.3750, +0.3750] |

## Primary contrasts

Synthetic example; these fixtures cannot support or challenge the scientific hypothesis.

D = delta1 − delta16: +0.0781; 95% CI [-0.2891, +0.3984].

Observed coverage across all 16 samples: both 3; Base only 1; RL only 1; neither 3.

## Diagnostics

| Model | Format compliance | Extraction failure | Ambiguous answer | Truncation | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| base | 81.25% | 0.00% | 0.00% | 0.00% | 4.0 |
| rl | 81.25% | 0.00% | 0.00% | 0.00% | 4.0 |

## Interpretation limits

- SYNTHETIC FIXTURE ONLY: these values test the pipeline and are not measurements of model capability.
- pass@k estimates success within k independent draws; it does not measure whether a model can select a correct answer without a verifier.
- Coverage categories use all n collected samples; with n greater than 16 they describe observed pass@n, not estimated pass@16.
- Intervals are marginal 95% paired percentile bootstrap intervals; they quantify evaluation-question uncertainty conditional on this training run, not variation across training seeds.
- An interval crossing zero is inconclusive evidence about a difference, not proof of no change.
- Correct final values do not establish valid reasoning; inspect retained outputs manually.

## Manual inspection

`manual_inspection.csv` includes up to 200 deterministic candidate outputs with diagnostic failures, model coverage disagreements, differing correctness, or a correct response among the first two draws. This convenience sample is not a random audit and cannot estimate invalid-reasoning prevalence. Review `text` against `question` and `answer`, and record a reasoning judgment separately.

The observed score uses every valid generated sample in the denominator; format failures and ambiguous extractions do not disappear from the analysis. Infrastructure errors and invalid gold answers are rejected before computing scores.
