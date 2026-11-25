# Paired RLVR evaluation

Recorded model evaluation; interpretation is conditional on this training run.

Split: `test`. Questions: 200. Independent responses per question and model: 16.
95% intervals: 2000 paired bootstrap replicates, resampling questions; seed 42.

| k | Base pass@k | RL pass@k | RL − Base (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.4534 | 0.7906 | +0.3372 [+0.3075, +0.3666] |
| 2 | 0.6565 | 0.8533 | +0.1968 [+0.1711, +0.2225] |
| 4 | 0.8141 | 0.8975 | +0.0834 [+0.0600, +0.1077] |
| 8 | 0.8975 | 0.9321 | +0.0346 [+0.0108, +0.0602] |
| 16 | 0.9400 | 0.9550 | +0.0150 [-0.0100, +0.0400] |

## Primary contrasts

Both directional contrasts have positive marginal 95% intervals in this run; this supports the finite-sample hypothesis conditional on this protocol.

D = delta1 − delta16: +0.3222; 95% CI [+0.2812, +0.3585].

Observed coverage across all 16 samples: both 186; Base only 2; RL only 5; neither 7.

## Diagnostics

| Model | Format compliance | Extraction failure | Ambiguous answer | Truncation | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| base | 40.81% | 36.50% | 0.25% | 4.28% | 173.9 |
| rl | 97.09% | 2.88% | 0.00% | 2.84% | 187.5 |

## Interpretation limits

- pass@k estimates success within k independent draws; it does not measure whether a model can select a correct answer without a verifier.
- Coverage categories use all n collected samples; with n greater than 16 they describe observed pass@n, not estimated pass@16.
- Intervals are marginal 95% paired percentile bootstrap intervals; they quantify evaluation-question uncertainty conditional on this training run, not variation across training seeds.
- An interval crossing zero is inconclusive evidence about a difference, not proof of no change.
- Correct final values do not establish valid reasoning; inspect retained outputs manually.

## Manual inspection

`manual_inspection.csv` includes up to 200 deterministic candidate outputs with diagnostic failures, model coverage disagreements, differing correctness, or a correct response among the first two draws. This convenience sample is not a random audit and cannot estimate invalid-reasoning prevalence. Review `text` against `question` and `answer`, and record a reasoning judgment separately.

The observed score uses every valid generated sample in the denominator; format failures and ambiguous extractions do not disappear from the analysis. Infrastructure errors and invalid gold answers are rejected before computing scores.
