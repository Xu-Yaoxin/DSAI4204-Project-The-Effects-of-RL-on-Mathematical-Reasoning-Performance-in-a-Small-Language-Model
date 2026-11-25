# Guarded baseline calibration / 带重复防护的基线评估

**Exploratory baseline-only results under the recorded guard protocol.**

Questions: 200; attempts/question: 4; handled attempts: 800; evaluable: 736; unknown: 31; answer-timeout policy failures: 33.
EOS-completed answers receive verifier scores. Answer timeout at 300 seconds is a user-defined zero-score policy failure; other guard-aborted answers retain null scores.

| Group | Questions | Completion | Correct/request | 95% interval for observed success | Incomplete-answer bounds (not CI) | Completed-only accuracy | pass@4 bounds (not CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM-Symbolic main confirmation | 100 | 99.5% | 64.0% | [55.2%, 72.7%] | [64.0%, 64.5%] | 64.3% | [85.0%, 86.0%] |
| MATH Level4 scalar confirmation | 100 | 84.5% | 43.8% | [37.0%, 50.7%] | [43.8%, 51.0%] | 51.8% | [68.0%, 82.0%] |

40–50% target decisions remain unresolved for groups with any unknown answer. Completed-only accuracy must not be used to select easier-looking data.

## Stop diagnostics / 终止原因

- GSM-Symbolic main confirmation: {"eos": 398, "repetition_guard": 2}; unknown=2; answer-timeout failures=0; target assessment=unresolved_incomplete_answers.
- MATH Level4 scalar confirmation: {"answer_timeout": 33, "eos": 338, "repetition_guard": 29}; unknown=29; answer-timeout failures=33; target assessment=unresolved_incomplete_answers.

## Interpretation limits / 解释边界

- Exploratory baseline screening under a new guarded stopping protocol; no formal Base/RL comparison.
- Only natural-EOS answers are scoring-evaluable. User-defined answer_timeout at the recorded deadline is a policy failure scored zero, not evidence of a mathematical error. Repetition, no-token timeout and context-limit stops remain unscored unknowns.
- Observed success per request is completed-and-correct/n. Its 95% interval concerns operational success under this guard, not the unknown mathematical correctness of aborted answers.
- Incomplete-answer bounds [c/n,(c+u)/n] are worst/best-case assignments for unknown answers, not confidence intervals or guarantees about generation without the guard.
- Completed-only accuracy excludes answer timeouts and every other non-EOS stop; it conditions on finishing and can be selection-biased. It is not the target metric or ranking criterion.
- Pass@4 bounds assign unknown answers as failures/successes; they are diagnostic bounds, not confidence intervals. Success within four attempts is not pass@1.
- Repeated answers and template instances are dependent. Intervals resample entire families, are marginal rather than simultaneous, and do not remove missing-answer uncertainty.
- A degenerate bootstrap interval reflects zero empirical variation, not certainty about future performance. Small family counts give coarse precision.
- Source/level labels do not establish a global model difficulty order. MATH rows are scalar/text-only subsets; answer correctness does not prove reasoning validity.
- No question is selected using its observed correctness. Fresh confirmation/comparison must exclude all calibration template families.

`per_question.jsonl` preserves counts and unknown bounds for every question. `calibration_candidates.json` ranks groups descriptively by observed success per request, never completed-only accuracy; it does not select questions or claim the unknown outcomes are errors.
