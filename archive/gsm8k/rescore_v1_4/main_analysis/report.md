# Paired RLVR evaluation

**POSTHOC OFFLINE RESCORING — original generation, training rewards and checkpoint selection retained.**

Scorer SHA-256: `0e0121cb6994f1a5206e4d1d85fa560dbf350e4fbb9b5904f218ef2d13475ae1`. This analysis cannot retroactively repair training or establish valid reasoning.

Recorded model evaluation; interpretation is conditional on this training run.

Split: `test`. Questions: 200. Independent responses per question and model: 16.
95% intervals: 2000 paired bootstrap replicates, resampling questions; seed 42.

| k | Base pass@k | RL pass@k | RL − Base (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.6481 | 0.7906 | +0.1425 [+0.1197, +0.1644] |
| 2 | 0.7907 | 0.8533 | +0.0626 [+0.0433, +0.0829] |
| 4 | 0.8706 | 0.8975 | +0.0269 [+0.0094, +0.0459] |
| 8 | 0.9196 | 0.9321 | +0.0124 [-0.0057, +0.0317] |
| 16 | 0.9500 | 0.9550 | +0.0050 [-0.0150, +0.0300] |

## Primary contrasts

Both directional contrasts have positive marginal 95% intervals in this run; this supports the finite-sample hypothesis conditional on this protocol.

D = delta1 − delta16: +0.1375; 95% CI [+0.1050, +0.1672].

Observed coverage across all 16 samples: both 188; Base only 2; RL only 3; neither 7.

## Diagnostics

| Model | Format compliance | Extraction failure | Ambiguous answer | Truncation | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| base | 47.12% | 9.03% | 0.25% | 4.28% | 173.9 |
| rl | 97.09% | 2.88% | 0.00% | 2.84% | 187.5 |

## Interpretation limits

- POSTHOC OFFLINE RESCORING: the original saved responses and frozen checkpoint are retained. Generation protocol_id describes the original generation; scoring_revision identifies the new scorer separately. This does not rerun generation, correct past training rewards, or reselect a checkpoint.
- pass@k estimates success within k independent draws; it does not measure whether a model can select a correct answer without a verifier.
- Coverage categories use all n collected samples; with n greater than 16 they describe observed pass@n, not estimated pass@16.
- Intervals are marginal 95% paired percentile bootstrap intervals; they quantify evaluation-question uncertainty conditional on this training run, not variation across training seeds.
- An interval crossing zero is inconclusive evidence about a difference, not proof of no change.
- Correct final values do not establish valid reasoning; inspect retained outputs manually.
- The scorer was revised after inspecting this run. Results are a scoring-sensitivity analysis, not an independently confirmed experiment under a prospectively fixed scorer. Exact final-value scoring does not assess reasoning validity.

## Manual inspection

`manual_inspection.csv` includes up to 200 deterministic candidate outputs with diagnostic failures, model coverage disagreements, differing correctness, or a correct response among the first two draws. This convenience sample is not a random audit and cannot estimate invalid-reasoning prevalence. Review `text` against `question` and `answer`, and record a reasoning judgment separately.

The observed score uses every valid generated sample in the denominator; format failures and ambiguous extractions do not disappear from the analysis. Infrastructure errors and invalid gold answers are rejected before computing scores.
