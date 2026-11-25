# Baseline difficulty calibration / 基线难度摸底

**Exploratory baseline-only screening. These are not formal held-out comparison results.**

Questions: 100. Samples per question: 4. Saved answers: 400.
Target: 40–50% single-attempt success (pass@1). Dataset labels describe sources and levels; measured difficulty is reported below.

| Group | Questions | pass@1 | 95% interval | pass@4 | Target point? | Truncated | Extraction failure |
|---|---:|---:|---:|---:|---|---:|---:|
| GSM-Symbolic main | 20 | 51.2% | [38.8%, 63.7%] | 85.0% | no | 6.2% | 11.2% |
| GSM-Symbolic P1 | 20 | 30.0% | [15.0%, 45.0%] | 55.0% | no | 3.8% | 12.5% |
| GSM-Symbolic P2 | 20 | 23.8% | [12.5%, 36.2%] | 60.0% | no | 3.8% | 13.8% |
| MATH-500 scalar Level1-3 | 20 | 68.8% | [55.0%, 82.5%] | 95.0% | no | 10.0% | 16.2% |
| MATH-500 scalar Level4-5 | 20 | 25.0% | [13.8%, 37.5%] | 55.0% | no | 28.7% | 30.0% |

## Group ranking / 题组排序

Groups are ranked by distance of pass@1 from 45%. A point inside 40–50% is a candidate for fresh confirmation, not an established ability boundary. No questions or models are automatically selected.

1. GSM-Symbolic main: pass@1 51.2%; interval wholly in target: False; length-budget concern: False.
2. GSM-Symbolic P1: pass@1 30.0%; interval wholly in target: False; length-budget concern: False.
3. MATH-500 scalar Level4-5: pass@1 25.0%; interval wholly in target: False; length-budget concern: True.
4. GSM-Symbolic P2: pass@1 23.8%; interval wholly in target: False; length-budget concern: False.
5. MATH-500 scalar Level1-3: pass@1 68.8%; interval wholly in target: False; length-budget concern: True.

## Interpretation limits / 解释边界

- Exploratory difficulty screening; no claim that a selected group is confirmed to have 40-50% success.
- Each group contains few questions. Repeated answers do not turn 20 questions into 80 independent questions; intervals are marginal, not simultaneous.
- A degenerate bootstrap interval reflects zero empirical variation in the collected question scores, not certainty about future performance.
- The target applies to pass@1, not success within four attempts. No individual questions are selected using model outcomes.
- Dataset/level labels are source-defined strata, not a demonstrated total ordering of difficulty for this model.
- MATH rows are text-only scalar-answer subsets, not full MATH benchmark scores. Final-value correctness does not establish reasoning validity.
- Review extraction failures and truncation before interpreting low scores as model inability. Results are conditional on the recorded generation budget.
- Choose future source strata using baseline calibration, then evaluate Base/RL on fresh questions. For symbolic variants, keep entire original-template families out of confirmation data.

With one response per question, intervals use Wilson when families are unique. With repeated responses or shared families, intervals bootstrap whole families, retaining all responses within each question. Twenty families give coarse uncertainty; more draws from the same questions cannot replace fresh independent questions.

`summary.json` retains interval methods, diagnostics and protocol identities. `per_question.jsonl` retains all question scores, without selecting favorable examples. `calibration_candidates.json` ranks only source groups for a subsequent independent confirmation.
