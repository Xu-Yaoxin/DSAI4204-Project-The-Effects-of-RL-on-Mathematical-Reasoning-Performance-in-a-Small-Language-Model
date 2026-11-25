# Answer quality and verifier audit

This review inspects the downloaded 2B run. It preserves the original outputs, primary scores, source code, and gold answers. It is a Codex-assisted qualitative audit, not an independent human adjudication.

## Confirmed parser interaction

The archived verifier accepts both `#### 5` and explicit `Answer: 5` forms. When an output contains `Answer: #### 5`, however, the two extraction rules overlap:

1. The hash-marker rule contributes the valid candidate `5`.
2. The explicit-answer rule contributes the candidate `#### 5`.
3. The scalar parser cannot parse that second candidate, marks the response invalid, and overrides the valid candidate from step 1.

This is an interaction between supported extraction rules, not simply a failure to emit a correct number. The verifier records format compliance separately, so this overlap also contaminates the nominal numerical-correctness reward.

One real example is `gsm8k-test-001010`, sample 0. The Base response correctly computes Bethany 10 laps, Trey 14, Shaelyn 7, Quinn 5, and the difference 5, then ends `Answer: #### 5`. It receives `extraction_failed`. The independent numerical answer is 5.

## Narrow symmetric diagnostic

The separate script [answer_quality_review.py](answer_quality_review.py) copies the archived verifier and changes exactly one behavior: an explicit-answer candidate starting with literal `####` is skipped because the existing hash-marker rule already extracts it. Malformed marked values and contradictory values remain rejected. No number is selected using the gold answer. Six small diagnostic cases check valid, wrong, contradictory, malformed, and unchanged forms.

This rule is applied to all 3,200 Base and 3,200 RL outputs. The original verifier is replayed first and agrees with the archived scores. The patched copy and diagnostic scores are kept under [nested_marker_diagnostic](nested_marker_diagnostic/). This is a post-hoc sensitivity analysis, not a replacement primary result.

| Measure | Original Base | Diagnostic Base | Original/diagnostic RL |
|---|---:|---:|---:|
| Correct responses | 1,451 | 1,885 | 2,530 |
| pass@1 | 45.34375% | 58.90625% | 79.06250% |
| pass@16 | 94.0% | 94.5% | 95.5% |

The change makes 601 previously unparseable Base responses parseable: 434 correct and 167 incorrect. No RL response changes status. The RL-minus-Base pass@1 difference decreases from 33.71875 to 20.15625 percentage points. The pass@16 difference decreases from 1.5 to 1.0 percentage point. Thus 13.5625 percentage points of the original single-draw gap are sensitive to this one parser interaction. This does not establish that all remaining improvement is mathematical reasoning.

The completed policy was trained using the original verifier, and checkpoint selection also used it. Evaluation-only rescoring cannot retrospectively produce a run trained with a corrected numerical reward. The diagnostic also retains the originally selected checkpoint; it does not select a model based on final-test outcomes.

## Qualitative inspection

A fixed `random.Random(20261031)` sample includes 24 Base extraction failures, 12 RL extraction failures, and 8 randomly selected question/sample-index pairs inspected for both models. Full records are in [answer_quality_review.sample.jsonl](answer_quality_review.sample.jsonl); judgments and traceable text hashes are in [answer_quality_review.json](answer_quality_review.json). This is a stratified diagnostic sample, not a prevalence estimate for all outputs.

- Among the 24 sampled Base extraction failures, 16 contain a correct calculation and numerical value. Eight use the overlapping explicit/hash-marker form; two use literal XML answer wrappers; one uses `The answer is: 160`; the other five continue after a correct value with meta discussion, a new prompt, or an empty marker. Some continuations are truncated. These categories should not all be treated as clean final answers automatically.
- Seven sampled Base failures contain incorrect mathematics or an unfinished incorrect loop. The remaining sample has a plausible interpretation disagreement with the supplied gold answer.
- All 12 sampled RL extraction failures are truncated and contain substantive reasoning errors or unresolved loops. For example, `gsm8k-test-000186` invents 20 rabbits each receiving 20 pieces of straw although the question gives 20 total rabbit straw pieces. This is not a cosmetic output-format failure.
- In the eight paired examples, three pairs are mathematically correct in both outputs; two additional pairs contain correct Base calculations but rejected Base formatting and correctly scored RL outputs; three pairs show mathematical errors in both outputs. These eight pairs illustrate failure modes and do not estimate the overall treatment effect.

Additional cases requiring explicit future scoring decisions include `The final answer is: 5` followed by a valid `#### 5` (the explicit candidate becomes `: 5`), literal `<answer>5</answer>` wrappers, and empty markers emitted after a valid answer. They are outside the narrow diagnostic and remain unchanged.

For `gsm8k-test-001048`, the supplied answer assumes straight-line annual depreciation of 21% of the initial car price, yielding 7,400; one Base response interprets 21% per year as compound depreciation and computes 9,860.78. The official raw GSM8K rationale was checked. This ambiguity is flagged, not used to alter the test label or exclude the question.

## Consequences for the research claim

The primary report is reproducible under its original scorer, but its 33.72-point improvement cannot be presented as an isolated mathematical-reasoning gain. Output regularization and a confirmed parser interaction contribute to the observed result. A correct final value also does not by itself validate every reasoning step.

The current run should be retained and identified by its original verifier hash. Before a new controlled run or the proposed 4B extension, specify and test a revised answer-extraction policy independently of final-test outcomes, apply it consistently to reward, development selection, and evaluation, and keep the old and new protocols separate. A broader secondary scoring analysis should state its rules in advance and be symmetric for Base and RL, rather than accepting any number matching the gold somewhere in the response.
