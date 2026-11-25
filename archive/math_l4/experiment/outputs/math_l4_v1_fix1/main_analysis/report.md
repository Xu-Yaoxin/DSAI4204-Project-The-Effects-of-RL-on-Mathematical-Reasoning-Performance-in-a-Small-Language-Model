# MATH Level4 Base/RL comparison

Questions: 200; samples/model/question: 16; selected optimizer step: 384.
Primary scores use the fixed300-second and repetition policy; all requested attempts remain in the denominator.

| Metric | Base | RL | Paired difference | 95% interval for difference |
|---|---:|---:|---:|---:|
| Operational pass@1 | 43.53% | 60.94% | +17.41pp | [+14.47, +20.38]pp |
| Operational pass@4 | 69.63% | 81.45% | +11.81pp | [+8.44, +15.20]pp |
| Operational pass@16 | 85.50% | 93.50% | +8.00pp | [+3.50, +12.50]pp |

Delta pass@1 minus delta pass@16: +9.41pp; 95% interval [+4.31, +14.69]pp.

Observed coverage counts: {"base_only": 4, "both": 167, "neither": 9, "rl_only": 20}

## Completion diagnostics

- base: {"completed_count": 2770, "completed_only_accuracy": 0.5028880866425993, "correct_count": 1393, "mean_output_tokens": 1441.933125, "requested_count": 3200, "status_counts": {"ambiguous": 41, "correct": 1393, "extraction_failed": 176, "incorrect": 1160, "timeout": 219, "unscored": 211}, "summed_answer_seconds": 113641.05081334338, "termination_counts": {"answer_timeout": 219, "eos": 2770, "repetition_guard": 211}, "unknown_count": 211}
- rl: {"completed_count": 2742, "completed_only_accuracy": 0.7111597374179431, "correct_count": 1950, "mean_output_tokens": 1403.3596875, "requested_count": 3200, "status_counts": {"correct": 1950, "extraction_failed": 92, "incorrect": 700, "timeout": 320, "unscored": 138}, "summed_answer_seconds": 183040.58411095105, "termination_counts": {"answer_timeout": 320, "eos": 2742, "repetition_guard": 138}, "unknown_count": 138}

Unknown-outcome bounds (not confidence intervals):

- pass@1: Base [0.4353125, 0.50125]; RL [0.609375, 0.6525]; difference [0.108125, 0.2171875].
- pass@4: Base [0.6963296703296704, 0.8141840659340663]; RL [0.8144725274725279, 0.8889313186813195]; difference [0.0002884615384615241, 0.19260164835164836].
- pass@16: Base [0.855, 0.965]; RL [0.935, 0.995]; difference [-0.03, 0.14].

## Interpretation limits

- Primary estimates are operational success under the fixed native-context/300-second/repetition policy. Unknown stopped attempts contribute no confirmed success; this does not label them mathematical errors.
- Answer timeouts are zero by the prespecified policy. Exact repetition, no-token stalls and context stops retain unknown correctness; unknown bounds are not confidence intervals.
- Base and RL use the same questions, deterministic sample seeds, scorer, prompt, dtype, backend and generation settings; the only intended model difference is the selected LoRA adapter.
- Checkpoint selection used all predeclared epoch candidates on development only; test outcomes never choose a checkpoint.
- Confidence intervals use paired whole-question-family bootstrap; repeated answers are not independent questions. Intervals are marginal, not multiplicity adjusted.
- Zero observed success in16 attempts does not establish zero capability; pass@16 is limited-sample coverage, not a complete reasoning boundary.
- This is one training seed and one source-filtered Level4 subset. It does not establish universal RL effects or validate every reasoning step.
