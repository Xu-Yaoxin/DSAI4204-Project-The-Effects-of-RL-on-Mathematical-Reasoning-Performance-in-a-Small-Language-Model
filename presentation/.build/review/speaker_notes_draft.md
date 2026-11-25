# Does RL Make a 2B Model More Reliable?

**English presentation script.** Twelve main slides, 600 seconds total. Source notes and the reference appendix are not spoken. Rehearse at approximately 120 words per minute, allowing brief pauses to point to charts.

| Presenter role | Slides | Time |
|---|---|---:|
| Speaker 1 | 1–4 | 3:15 |
| Speaker 2 | 5–8 | 3:20 |
| Speaker 3 | 9–12 | 3:25 |
| Total | 1–12 | 10:00 |

## 1. Does RL Make a 2B Model More Reliable?

**Time: 0:00–0:30 (30 seconds). Speaker 1.**

Our project asks whether reinforcement learning can make a small language model more reliable at solving mathematics. We compare the same two-billion-parameter base model before and after training with verifiable answer rewards. On our selected MATH Level 4 test set, single-answer success increased from 43.53 percent to 61.50 percent. We will explain how we measured that improvement and what it tells us about reasoning.

*Source notes: `results/final/summary.json`; `code/math_l4_v1/configs/math_l4.json`.*

## 2. Reliability and Observed Coverage

**Time: 0:30–1:15 (45 seconds). Speaker 1.**

Our starting point is the question raised by Yue and colleagues: does reinforcement learning expand a model's reasoning capacity, or make successful answers more likely? These possibilities require different measurements. Imagine a student who occasionally solves a difficult problem. Training might help that student succeed more consistently, even if the set of problems they can solve barely changes. We therefore measure both success on one sampled answer and coverage across sixteen attempts. This is a small-scale adaptation of the paper's research question. Our model, training budget, and evaluation settings differ from the original study.

*Source notes: Yue et al. 2025, arXiv:2504.13837v5; `ANALYSIS.md`, sections 1 and 3.*

## 3. Independent Data and a Paired Comparison

**Time: 1:15–2:10 (55 seconds). Speaker 1.**

We use Qwen3.5-2B-Base and a selected subset of MATH Level 4 with scalar numerical answers. The dataset contains 512 training questions, 128 development questions, and 200 independent test questions. Training updates the model, development chooses the checkpoint, and testing measures the final comparison. Each model produces sixteen independently seeded answers for each test question. That gives 3,200 answers per model and 6,400 in total. Both models receive the same prompts and sampling settings, including temperature 0.6 and top-p 0.95. Every requested attempt stays in the denominator, including timeouts. Comparing the same questions helps separate model differences from differences in question difficulty.

*Source notes: `code/math_l4_v1/configs/math_l4.json`; `results/final/summary.json`; `results/final/per_question.jsonl`.*

## 4. Learning from Verifiable Rewards

**Time: 2:10–3:15 (65 seconds). Speaker 1.**

The training loop combines GRPO with LoRA. For each question, the model samples eight responses. A numerical checker gives a reward of one for a confirmed correct answer and zero otherwise. GRPO compares rewards within that group. For example, if four answers are correct, the average reward is one half. Correct answers are above that average, while incorrect answers are below it. The update encourages responses with better relative rewards. Groups where every reward is identical provide no reward variation. LoRA updates about 10.9 million adapter parameters while keeping the base weights frozen. We train for three epochs, or 384 optimizer updates. There is no supervised fine-tuning stage or teacher-provided reasoning. The model learns from its own sampled answers and their verified outcomes.

*Source notes: `code/math_l4_v1/configs/math_l4.json`; `archive/math_l4/experiment/outputs/math_l4_v1_fix1/formal/report.json`; `models/rl_checkpoint384/adapter_model.safetensors`. The four-correct example illustrates the method; it is not a reported training batch.*

## 5. Checkpoint Selection Used Development Data

**Time: 3:15–3:55 (40 seconds). Speaker 2.**

We evaluated checkpoints at steps 128, 256, and 384. Each answered the same 128 development questions four times, giving 512 responses per checkpoint. The confirmed-correct rate increased from 49.80 to 58.40 and then 60.35 percent. We selected checkpoint 384 using this development score. Test answers played no part in that decision. Training and selection used scorer version three. We later improved the final evaluation scorer, but kept the selected checkpoint and every generated answer unchanged. This keeps the selection decision separate from the final test analysis.

*Source notes: `archive/math_l4/experiment/outputs/math_l4_v1_fix1/selection.json`; `ANALYSIS.md`, section 2.*

## 6. What pass@1, pass@4, and pass@16 Mean

**Time: 3:55–4:50 (55 seconds). Speaker 2.**

Pass at one estimates the chance that one sampled answer is correct. Pass at four estimates the chance of at least one correct answer in four attempts. Pass at sixteen measures whether any of our sixteen answers is correct. Consider an actual test question where Base answered eight times correctly and eight incorrectly. Its pass at one estimate is fifty percent. If we choose four answers from those sixteen, only seventy of the 1,820 possible groups contain four failures. Estimated pass at four is therefore 96.15 percent. Pass at sixteen is one hundred percent. We calculate these quantities per question and then average across the two hundred questions.

*Source notes: `results/final/per_question.jsonl`, question `math-l4-test-algebra-0029`; `ANALYSIS.md`, section 3.*

## 7. RL Improves All Three Main Metrics

**Time: 4:50–5:45 (55 seconds). Speaker 2.**

The main results show improvement at every tested sampling budget. Pass at one rises from 43.53 to 61.50 percent. Pass at four rises from 69.63 to 81.93 percent. Pass at sixteen rises from 85.50 to 94 percent. These differences are 17.97, 12.29, and 8.50 percentage points respectively. In raw answer counts, Base produces 1,393 confirmed-correct responses, while RL produces 1,968, an increase of 575 out of the same 3,200 requested attempts. The largest improvement appears when we ask for only one answer. That pattern suggests a stronger gain in reliability than in coverage within our sixteen-attempt budget.

*Source notes: `results/final/summary.json`, main estimates and score totals.*

## 8. Paired Uncertainty Estimates

**Time: 5:45–6:35 (50 seconds). Speaker 2.**

We estimated uncertainty with two thousand paired bootstrap resamples. Each resample keeps the Base and RL results for a question together. We resample question families, rather than treating repeated answers as independent questions. All three main gain intervals lie above zero. For pass at one, the 95 percent interval is approximately fifteen to twenty-one percentage points. For pass at sixteen, it is four to thirteen points. The difference between those two gains is 9.47 points, with an interval from 4.37 to 14.75. These intervals describe question-sampling variation for this trained model. They do not measure variation across training seeds.

*Source notes: `results/final/summary.json`, bootstrap configuration, paired deltas, and D interval.*

## 9. Seventeen More Questions Covered

**Time: 6:35–7:25 (50 seconds). Speaker 3.**

The overlap diagram shows which questions each model covers within sixteen attempts. Both models solve 167 questions. RL alone solves twenty-one, Base alone solves four, and neither solves eight. Base therefore covers 171 questions, compared with 188 for RL, a net increase of seventeen. The four Base-only questions also show that improvement is not uniform. However, the twenty-one RL-only questions are not proof of entirely new capabilities. A model with a small but nonzero success probability can fail sixteen times. Our evidence concerns observed coverage at this sampling budget, rather than the full set of problems the base model could ever solve.

*Source notes: `results/final/summary.json`, coverage overlap; `results/final/per_question.jsonl`.*

## 10. Scoring and Stopping Rules Matter

**Time: 7:25–8:20 (55 seconds). Speaker 3.**

We preserved the raw answers and applied one final scorer to both models. Version four recovered eighteen additional correct RL answers through extraction fixes; Base's correct count did not change. We did not regenerate any answers. Responses stopped by the guard retained their original failure or unknown status. Each answer had a five-minute deadline and repetition monitoring. RL produced fewer repetition stops, 138 compared with 211 for Base, but more timeouts, 320 compared with 219. The checker still has unresolved extraction or ambiguity cases. It verifies numerical final answers, so a correct score does not establish that every reasoning step is valid.

*Source notes: `results/final/summary.json`, rescoring changes and diagnostics; `results/final/changes.jsonl`; `ANALYSIS.md`, sections 5–6.*

## 11. Limits of the Comparison

**Time: 8:20–9:25 (65 seconds). Speaker 3.**

Three limits affect interpretation. First, we trained one seed and tested a selected scalar-answer subset, so we cannot generalize directly to all MATH problems or all training runs. Second, all machines used RTX 4090 GPUs, but Base generated about forty-one tokens per second, while the two RL workers generated about twenty and twenty-eight. The same five-minute deadline therefore allowed different generation opportunities. Third, unknown outcomes affect coverage. Assigning extreme alternative labels to those outcomes gives a pass-at-sixteen gain range from minus 2.5 to plus fourteen percentage points. This is a sensitivity range, not a confidence interval, and does not cover possible timeout or extraction errors. The positive main estimate holds under our scoring policy, but broader capacity claims require stronger evidence.

*Source notes: `ANALYSIS.md`, sections 5–7; archived Base and RL shard manifests; `results/final/summary.json`, unknown-outcome bounds.*

## 12. Higher Success Within the Tested Budget

**Time: 9:25–10:00 (35 seconds). Speaker 3.**

Our experiment shows that answer-reward RL can improve this small model under the tested protocol. Single-answer success increased by about eighteen percentage points, and observed coverage increased by seventeen questions. The reliability gain was larger than the sixteen-attempt coverage gain. A stronger follow-up would repeat training across several seeds, match token budgets, and independently audit scoring. For now, our conclusion is better observed performance, with the complete reasoning frontier still unresolved. Thank you.

*Source notes: `results/final/summary.json`; `ANALYSIS.md`.*

## Reference Appendix — Not Part of the Timed Talk

Yue, Yang, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?” *arXiv preprint*, arXiv:2504.13837v5. Retrieved November 7, 2026 (https://arxiv.org/abs/2504.13837v5).

Qwen Team. n.d. “Qwen3.5-2B-Base.” *Hugging Face model repository*. Model revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (https://huggingface.co/Qwen/Qwen3.5-2B-Base).

DSAI4207 Group Project. 2026. “MATH Level 4 Base/RL Experiment.” Internal frozen experiment records: `results/final/summary.json`, `results/final/per_question.jsonl`, `code/math_l4_v1/configs/math_l4.json`, and `archive/math_l4/experiment/outputs/math_l4_v1_fix1/selection.json`.
