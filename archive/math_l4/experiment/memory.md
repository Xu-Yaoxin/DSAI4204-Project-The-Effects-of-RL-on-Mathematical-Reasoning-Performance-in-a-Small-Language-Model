# Version history

## 2026-11-01 · math-l4-1.0.1

- Reproduced the server CPU-check failure with RLVR_CPU_THREADS=4: the generated hung-producer Torch fixture lacked both thread setters. The earlier local suite ran without the launcher variable and missed this branch. Real Torch, GPU availability and model loading were not implicated by this exception.
- Added both fixture setters and assertions for intraop=4/interop=1 in normal generation, forced timeout exit75 and infrastructure-failure exit1. The evaluation test explicitly supplies launcher thread/rank settings; the suite runner defaults RLVR_CPU_THREADS to4 so direct local runs exercise the same branch.
- Production scripts, configuration, data, scorer, sampling budgets and dependencies remain byte-identical to1.0. Bundle revision changes because tests and release evidence changed. Restart the preflight-only failure in outputs/math_l4_v1_fix1; preserve the previous directory and strict identity checks.
- Regression evidence is recorded in validation/thread_mock_fix/ and the current validation/test_report.json. New GPU training remains untested locally.

## 2026-11-01 · math-l4-1.0

- User chose MATH Level4 and retained Qwen3.5-2B-Base after guarded baseline calibration. User explicitly confirmed16 samples/question for Base/RL despite the multi-day evaluation budget.
- Created a separate package and output namespace. Officialtrain supplies512train+128dev; officialtest supplies200test. Engineering32 andpilot128 nest inside training. Frozen source revision, SHA-256, normalization exclusion ledger and proportional subject sampling preserve provenance without selecting by observed scores.
- Scorer3.0.0 repairs clear presentation losses while preserving exact scalar comparisons and public-answer conflict rules. Replayed old736EOS records:4MATH+3GSM newly correct, no previous correct answers lost. Two percent-convention cases remain separate from formatting bugs. Old scores unchanged.
- Shared scorer is used by GRPO rewards and both evaluation arms. Training uses2048-token rollouts,8generations/group,32responses/update,3epochs and zero reward unlessEOS. Formal384updates starts afresh fromBase after engineering/pilot acceptance.
- Reused sealed checkpoint recovery with optimizer/scheduler/RNG/sampler and reward-audit integrity. Recovery every16updates; epoch128/256/384 candidates are compared on128dev×4 only. TestRL requires the immutable selectedadapter.
- Native-context serial evaluation retains300-second timeoutzero, exactrepeatguard,120-secondno-token diagnosis, live traces, per-answer resume and bounded durable-stop recovery. Test200×16×2=6400responses; reporting includespass@1/4/16, paireduncertainty, coverage and unknown bounds.
- Added bundle/environment/data/tokenizer preflight, globalsingleGPU runlock, stageprogress, SIGTERMchildcleanup, same-command resume and existingenvironment reuse. No new packages or model download required on the current server.
- Actual new2048-token MATH GPU training remains unrun locally; CPU fixtures and plans are evidence for control and integrity only. Hardwareengineering/pilot gates precedeformaltraining.
- Serverport42172; remote/root/code/math_l4_v1; commands inCOMMANDS.md. Final test and release reports invalidation/.

- Release validation:11 CPU suites,658 scenarios passed in73.80s. Actual-data fullplan passed; shell syntax and upload-filter relocation verified. Old v2/v3 packages86 bound files remainunchanged. New GPU training not executed locally.

- 1.0.1 validation:11 CPU suites/661 scenarios passed in73.11s with launcher thread settings; targeted evaluation62 checks passed. No production script or scientific setting changed.
