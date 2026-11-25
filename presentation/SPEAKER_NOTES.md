# Does RL Make a 2B Model More Reliable?

**Canva presentation:** https://www.canva.com/d/LOOPQ54pGMZKgqb

**Delivery:** English, 12 main slides in 10 minutes, followed by an untimed reference appendix. Timings include brief pauses to point to charts. Source notes and references are not spoken. The role labels below divide the talk without assigning group members.

| Presenter role | Slides | Time |
|---|---|---:|
| Speaker 1 | 1–4 | 3:00 |
| Speaker 2 | 5–8 | 3:25 |
| Speaker 3 | 9–12 | 3:35 |
| Total | 1–12 | 10:00 |

## 1. Does RL Make a 2B Model More Reliable?

**Time: 0:00–0:30 (30 seconds). Speaker 1.**

Our project asks whether reinforcement learning can make a small language model more reliable at solving mathematics. We compare the same two-billion-parameter base model before and after training with verifiable answer rewards. We will explain our experiment, quantify the changes, and distinguish stronger performance on this test from broader claims about reasoning capacity.

*Source notes: `code/math_l4_v1/configs/math_l4.json`; `ANALYSIS.md`.*

## 2. Higher Reliability and Observed Coverage

**Time: 0:30–1:20 (50 seconds). Speaker 1.**

Here are the two main findings. Single-answer success increased from 43.53 percent to 61.50 percent, a gain of 17.97 percentage points. The proportion of questions solved at least once within sixteen attempts increased from 85.50 to 94 percent, a gain of 8.50 points. These metrics describe different improvements. The first tells us how reliably one sampled response succeeds. The second tells us how many questions are covered when we allow repeated attempts. Both improved, but the larger gain was in single-answer reliability. All results come from the same two hundred held-out test questions.

*Source notes: `results/final/summary.json`, final v4 estimates. Percentage-point differences are absolute differences, not relative percentage increases.*

## 3. The Research Question

**Time: 1:20–2:05 (45 seconds). Speaker 1.**

Our starting point is the question raised by Yue and colleagues: does reinforcement learning expand reasoning capacity, or make successful answers more likely? Imagine a student who occasionally solves a difficult problem. Training might help that student succeed more consistently, even if the set of problems they can solve changes little. We therefore examine both reliability and coverage. This is a small-scale adaptation of the paper's research question. Our model, training budget, and sampling range differ from the original study, so we do not claim an exact replication.

*Source notes: Yue et al. 2025, arXiv:2504.13837v5; `ANALYSIS.md`, section 1.*

## 4. Independent Data and a Paired Comparison

**Time: 2:05–3:00 (55 seconds). Speaker 1.**

We use Qwen3.5-2B-Base and a selected subset of MATH Level 4 with scalar numerical answers. There are 512 training questions, 128 development questions, and 200 independent test questions. Training updates the model, development chooses the checkpoint, and testing measures the final comparison. Each model produces sixteen independently seeded answers per test question. That gives 3,200 answers per model and 6,400 in total. Both models receive the same prompts and sampling settings, including temperature 0.6 and top-p 0.95. Every requested attempt stays in the denominator. Comparing the same questions helps separate model differences from differences in question difficulty.

*Source notes: `code/math_l4_v1/configs/math_l4.json`; `results/final/summary.json`; `results/final/per_question.jsonl`.*

## 5. Learning from Verifiable Rewards

**Time: 3:00–3:55 (55 seconds). Speaker 2.**

The training loop combines GRPO with LoRA. For each question, the model samples eight responses. A numerical checker gives one for a confirmed correct answer and zero otherwise. GRPO compares rewards within that group. For example, with four correct responses, the average reward is one half. Correct responses are above that average, while incorrect responses are below it. The update encourages better relative outcomes. LoRA changes about 10.9 million adapter parameters while keeping the base weights frozen. We train for three epochs, or 384 updates. There is no supervised fine-tuning stage or teacher-provided reasoning. The model learns from its own sampled answers and verified outcomes.

*Source notes: `code/math_l4_v1/configs/math_l4.json`; `archive/math_l4/experiment/outputs/math_l4_v1_fix1/formal/report.json`; `models/rl_checkpoint384/adapter_model.safetensors`. The four-correct example illustrates GRPO; it is not a reported training batch. LoRA rank 16, learning rate 0.000005, KL coefficient beta 0, training output cap 2,048 tokens.*

## 6. Development Data Selected Checkpoint 384

**Time: 3:55–4:35 (40 seconds). Speaker 2.**

We evaluated checkpoints at steps 128, 256, and 384. Each answered 128 development questions four times, giving 512 responses per checkpoint. The correct-answer rate rose from 49.80 to 58.40 and then 60.35 percent. We selected checkpoint 384 using this development score. Test answers played no part in that decision. Training and selection used scorer version three. Later improvements to final scoring did not change the selected checkpoint or generate new answers. The selection decision therefore remains separate from the final test analysis.

*Source notes: `archive/math_l4/experiment/outputs/math_l4_v1_fix1/selection.json`. Development numerators are 255, 299, and 309, all over 512. These are v3 development scores, not v4 final test scores.*

## 7. Understanding pass@1, pass@4, and pass@16

**Time: 4:35–5:30 (55 seconds). Speaker 2.**

Pass at one estimates success on one sampled answer. Pass at four estimates the chance of at least one correct answer in four attempts. Pass at sixteen checks whether any of our sixteen answers is correct. Consider an actual test question where Base answered eight times correctly and eight incorrectly. Its pass at one estimate is fifty percent. If we choose four answers from those sixteen, only seventy of the 1,820 possible groups contain four failures. Estimated pass at four is therefore 96.15 percent. Pass at sixteen is one hundred percent. We calculate these quantities per question and then average across the two hundred questions.

*Source notes: `results/final/per_question.jsonl`, question `math-l4-test-algebra-0029`; `results/final/summary.json`; `ANALYSIS.md`, section 3. The estimator is one minus the number of all-failure subsets divided by the number of possible k-answer subsets.*

## 8. RL Improves Every Tested Sampling Budget

**Time: 5:30–6:25 (55 seconds). Speaker 2.**

The bars compare all three metrics. Pass at one rises from 43.53 to 61.50 percent. Pass at four rises from 69.63 to 81.93 percent. Pass at sixteen rises from 85.50 to 94 percent. In raw answer counts, Base produces 1,393 confirmed-correct responses, while RL produces 1,968. That is 575 additional correct answers out of the same 3,200 requested attempts. These scores include the unsuccessful and interrupted attempts, rather than considering only completed responses. The shrinking gap as the sampling budget grows suggests that RL improves consistency more strongly than it expands observed coverage within sixteen attempts.

*Source notes: `results/final/summary.json`, main estimates and correct-answer totals. All figures use the unified final v4 scorer.*

## 9. Paired Confidence Intervals

**Time: 6:25–7:20 (55 seconds). Speaker 3.**

We estimated uncertainty with two thousand paired bootstrap resamples. Each resample keeps Base and RL results for the same question together. We resample question families rather than treating repeated answers as independent questions. All three main gain intervals lie above zero. For pass at one, the 95 percent interval is approximately fifteen to twenty-one percentage points. For pass at sixteen, it is four to thirteen points. The difference between those two gains is 9.47 points, with an interval from 4.37 to 14.75. These intervals describe question-sampling variation for this trained model. We used one training seed, so the intervals do not measure variation across repeated training runs.

*Source notes: `results/final/summary.json`, bootstrap configuration, paired gain intervals, and D interval. Two hundred distinct question families; 2,000 replicates; bootstrap seed 42.*

## 10. Seventeen More Questions Covered

**Time: 7:20–8:05 (45 seconds). Speaker 3.**

Both models solve 167 questions within sixteen attempts. RL alone solves twenty-one, Base alone solves four, and neither solves eight. Base therefore covers 171 questions, compared with 188 for RL, a net increase of seventeen. The four Base-only questions show that improvement is not uniform. However, the twenty-one RL-only questions do not prove entirely new capabilities. A model with a small but nonzero success probability can still fail sixteen times. Our evidence concerns observed coverage at this sampling budget, rather than everything the base model could ever solve.

*Source notes: `results/final/summary.json`, coverage overlap; `results/final/per_question.jsonl`.*

## 11. Shared Scoring and Stopping Rules

**Time: 8:05–8:50 (45 seconds). Speaker 3.**

We preserved the raw answers and applied one final scorer to both models. Version four recovered eighteen additional correct RL answers through extraction fixes; Base's correct count did not change. We did not regenerate answers or reselect the model. Stopped responses retained their original failure or unknown status. Every answer had a five-minute deadline and repetition monitoring. Natural end-of-sequence completion was required for credit. Unresolved extraction and ambiguity cases remain. The checker verifies numerical final answers, so a correct score does not establish that every reasoning step is valid.

*Source notes: `results/final/summary.json`, rescoring changes and diagnostics; `results/final/changes.jsonl`; `code/math_l4_v1/configs/math_l4.json`. There are 217 remaining Base EOS extraction/ambiguity cases and 67 for RL. The v4 revision changes 25 RL scoring records, with 18 becoming correct; Base has no scoring changes.*

## 12. Diagnostics, Limits, and Conclusion

**Time: 8:50–10:00 (70 seconds). Speaker 3.**

RL had fewer repetition stops: 138 versus 211. However, it had more timeouts: 320 versus 219. Although all devices were RTX 4090 GPUs, Base generated about forty-one tokens per second, while the RL workers generated twenty and twenty-eight. Equal time limits therefore allowed different generation opportunities. Unknown outcomes also affect interpretation. Assigning extreme alternative labels gives a pass-at-sixteen gain range from minus 2.5 to plus fourteen percentage points. That is a sensitivity range, not a confidence interval, and excludes possible timeout and extraction errors. Under our scoring policy, RL improved single-answer success by about eighteen points and observed coverage by seventeen questions. A stronger follow-up would use multiple training seeds, matched token budgets, and independently audited scoring. Our evidence supports improved performance under this protocol, while the complete reasoning frontier remains unresolved. Thank you.

*Source notes: `results/final/summary.json`, stops and unknown-outcome bounds; archived Base and RL shard manifests; `ANALYSIS.md`, sections 5–7. Recorded throughput is approximately 40.53 tokens/second for Base, 20.00 and 28.44 for RL workers. Hardware and task distributions affect these descriptive rates, so do not interpret them as a controlled speed benchmark.*

## 13. References — Untimed Appendix

Yue, Yang, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?” *arXiv preprint*, arXiv:2504.13837v5. Retrieved November 7, 2026 (https://arxiv.org/abs/2504.13837v5).

Qwen Team. n.d. “Qwen3.5-2B-Base.” *Hugging Face model repository*. Model revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (https://huggingface.co/Qwen/Qwen3.5-2B-Base).

DSAI4207 Group Project. 2026. “MATH Level 4 Base/RL Experiment.” Internal frozen experiment records: `results/final/summary.json`, `results/final/per_question.jsonl`, `code/math_l4_v1/configs/math_l4.json`, and `archive/math_l4/experiment/outputs/math_l4_v1_fix1/selection.json`.

The full paper PDF and its provenance are retained under `docs/papers/`. All internal source paths above are relative to the project root. Source notes are for preparation and questions, not part of the spoken script.
